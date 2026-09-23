"""FastAPI entrypoint: health, chat API, uploads, noVNC proxy, media, and static chat UI."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import websockets
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.websockets import WebSocketState

from app import __version__
from app.agent import chat as agent_chat
from app.attachments import MAX_UPLOAD_BYTES, resolve_upload, save_upload, uploads_root
from app.config import get_settings
from app import jobs as jobstore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("nitc.main")

STATIC_DIR = Path(__file__).parent / "static"
_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # Ensure job/session dirs exist and are writable by the agent user
    for root in (jobstore.jobs_root(), jobstore.sessions_root()):
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
    try:
        from app import bots_store
        bots_store.bots_path().parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    await jobstore.recover_queued_jobs()
    yield


app = FastAPI(
    title="Nitc Agent",
    description="Self-hosted AI agent with shell, files, interactive desktop, browser, and GitHub tools.",
    version=__version__,
    lifespan=_lifespan,
)


class ChatMessage(BaseModel):
    role: str
    content: Any = ""


class AttachmentRef(BaseModel):
    id: str | None = None
    path: str | None = None
    name: str | None = None
    mime: str | None = None
    size: int | None = None
    url: str | None = None


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(default_factory=list)
    message: str | None = Field(
        default=None,
        description="Convenience: single user message (appended if messages empty or in addition)",
    )
    attachments: list[AttachmentRef] = Field(
        default_factory=list,
        description="Files previously uploaded via /api/uploads; applied to the latest user turn",
    )
    session_id: str | None = Field(
        default=None,
        description="Client bot/session id for durable history mirroring",
    )
    bot_id: str | None = Field(
        default=None,
        description="Alias for session_id (multi-bot UI)",
    )
    instructions: str | None = Field(
        default=None,
        max_length=4000,
        description="Optional per-bot system prompt stub from the client",
    )


class JobCreateResponse(BaseModel):
    job_id: str
    status: str = "queued"
    session_id: str | None = None
    bot_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    tool_rounds: int = 0
    messages: list[dict[str, Any]] = Field(default_factory=list)
    images: list[str] = Field(default_factory=list)


def _novnc_upstream() -> str:
    return get_settings().novnc_upstream.rstrip("/")


def _novnc_embed_url() -> str:
    # Page lives at /novnc/vnc.html — path must be relative "websockify" (NOT "novnc/websockify"),
    # otherwise noVNC resolves to /novnc/novnc/websockify and fails with "Failed to connect".
    return "/novnc/vnc.html?autoconnect=1&resize=scale&path=websockify"


@app.get("/health")
async def health() -> dict[str, Any]:
    settings = get_settings()
    return {
        "status": "ok",
        "version": __version__,
        "model": settings.model,
        "workspace": str(settings.workspace_path),
        "api_key_set": bool(settings.openai_api_key),
        "desktop_api_url": settings.desktop_api_url,
        "novnc_public_url": settings.novnc_public_url,
        "novnc_embed_url": _novnc_embed_url(),
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "jobs": True,
        "async_chat": True,
        "job_pool": jobstore.pool_stats(),
    }


@app.get("/api/config")
async def api_config() -> dict[str, Any]:
    """Public UI config (no secrets)."""
    settings = get_settings()
    embed = _novnc_embed_url()
    # Same-origin by default so mobile carriers that block :6080 still work.
    # Keep configured NOVNC_PUBLIC_URL as optional direct/external link.
    direct = settings.novnc_public_url.rstrip("/")
    return {
        "version": __version__,
        "novnc_public_url": embed,
        "novnc_direct_url": f"{direct}/vnc.html?autoconnect=1&resize=scale" if direct else embed,
        "novnc_embed_url": embed,
        "model": settings.model,
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "desktop_proxy": "/api/desktop",
        "screen": {"width": 1280, "height": 800},
    }


@app.post("/api/uploads")
async def api_upload(file: UploadFile = File(...)) -> dict[str, Any]:
    """Save an uploaded file under workspace/uploads and return metadata."""
    raw = await file.read()
    try:
        meta = save_upload(raw, file.filename or "upload.bin", file.content_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        logger.exception("upload write failed")
        raise HTTPException(status_code=500, detail=f"Could not save upload: {exc}") from exc
    return meta


@app.get("/api/media/uploads/{name}")
async def media_upload(name: str) -> FileResponse:
    """Serve a previously uploaded file from workspace/uploads/ (path-safe)."""
    if not name or name != Path(name).name or ".." in name or "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="Invalid upload name")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        raise HTTPException(status_code=400, detail="Invalid upload name")

    uploads = uploads_root()
    path = (uploads / name).resolve()
    if not str(path).startswith(str(uploads.resolve()) + "/") and path != uploads.resolve():
        raise HTTPException(status_code=403, detail="Path escape blocked")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Upload not found")

    import mimetypes

    media, _ = mimetypes.guess_type(name)
    return FileResponse(path, media_type=media or "application/octet-stream", filename=name)



@app.get("/api/desktop/view.png")
async def desktop_view_png() -> Response:
    """Fresh desktop screenshot as PNG — mobile compat view when noVNC stalls."""
    settings = get_settings()
    upstream = f"{settings.desktop_api_url.rstrip('/')}/screenshot"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=8.0)) as client:
            resp = await client.post(upstream)
            if resp.status_code >= 400:
                raise HTTPException(status_code=502, detail="screenshot failed")
            data = resp.json()
            rel = str(data.get("path") or "")
            name = Path(rel).name
            if not name or ".." in name:
                raise HTTPException(status_code=502, detail="bad screenshot path")
            shot = (settings.workspace_path / "screenshots" / name).resolve()
            root = (settings.workspace_path / "screenshots").resolve()
            if not str(shot).startswith(str(root)) or not shot.is_file():
                raise HTTPException(status_code=404, detail="screenshot missing")
            return FileResponse(shot, media_type="image/png", headers={"Cache-Control": "no-store"})
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("desktop view png failed: %s", exc)
        raise HTTPException(status_code=502, detail="desktop view unavailable") from exc


@app.get("/api/media/screenshots/{name}")
async def media_screenshot(name: str) -> FileResponse:
    """Serve a PNG from workspace/screenshots/ (path-safe)."""
    if not name or name != Path(name).name or ".." in name or "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="Invalid screenshot name")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        raise HTTPException(status_code=400, detail="Invalid screenshot name")
    if not name.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".gif")):
        raise HTTPException(status_code=400, detail="Unsupported media type")

    settings = get_settings()
    shot_root = (settings.workspace_path / "screenshots").resolve()
    shot_root.mkdir(parents=True, exist_ok=True)
    path = (shot_root / name).resolve()
    if not str(path).startswith(str(shot_root) + "/") and path != shot_root:
        raise HTTPException(status_code=403, detail="Path escape blocked")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Screenshot not found")

    media = "image/png"
    lower = name.lower()
    if lower.endswith((".jpg", ".jpeg")):
        media = "image/jpeg"
    elif lower.endswith(".webp"):
        media = "image/webp"
    elif lower.endswith(".gif"):
        media = "image/gif"
    return FileResponse(path, media_type=media)






# —— Local auth + runtime settings (Profile) ——
import hashlib
import hmac
import secrets as _secrets
import time as _time

_AUTH_USERS_FILE = "auth_users.json"
_SESSIONS: dict[str, dict[str, Any]] = {}


def _auth_users_path() -> Path:
    return get_settings().workspace_path / _AUTH_USERS_FILE


def _load_users() -> dict[str, Any]:
    path = _auth_users_path()
    if not path.is_file():
        # seed default local account matching Profile defaults
        users = {
            "polycapsamuel5@gmail.com": {
                "email": "polycapsamuel5@gmail.com",
                "name": "chisom odoh",
                "initials": "CO",
                # password: nitc1234 (sha256)
                "password_sha256": hashlib.sha256(b"nitc1234").hexdigest(),
                "pin": "1234",
            }
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(users, indent=2) + "\n", encoding="utf-8")
        return users
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_users(users: dict[str, Any]) -> None:
    path = _auth_users_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(users, indent=2) + "\n", encoding="utf-8")


def _hash_pw(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _make_session(email: str, user: dict[str, Any]) -> str:
    token = _secrets.token_urlsafe(24)
    _SESSIONS[token] = {
        "email": email,
        "name": user.get("name") or email.split("@")[0],
        "initials": user.get("initials") or "CO",
        "exp": _time.time() + 60 * 60 * 24 * 30,
    }
    return token


def _session_from_request(request: Request) -> dict[str, Any] | None:
    auth = request.headers.get("authorization") or ""
    token = ""
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
    if not token:
        token = request.cookies.get("nitc_session") or ""
    if not token:
        return None
    sess = _SESSIONS.get(token)
    if not sess:
        return None
    if float(sess.get("exp") or 0) < _time.time():
        _SESSIONS.pop(token, None)
        return None
    return {"token": token, **sess}


class AuthLoginBody(BaseModel):
    email: str = Field(..., min_length=1, max_length=320)
    password: str | None = Field(default=None, max_length=200)
    pin: str | None = Field(default=None, max_length=32)


class SettingsPatch(BaseModel):
    autoReview: bool | None = None
    autoReviewRules: list[str] | None = None
    tzAuto: bool | None = None
    timeZone: str | None = None
    notifications: bool | None = None
    appearance: str | None = None
    language: str | None = None
    haptics: bool | None = None
    plugins: dict[str, bool] | None = None


class AllowOnceBody(BaseModel):
    tool: str = Field(..., min_length=1, max_length=64)
    args: dict[str, Any] = Field(default_factory=dict)




@app.get("/api/media/files/{name}")
async def media_file(name: str) -> FileResponse:
    """Serve a downloadable artifact from workspace/downloads/ (path-safe)."""
    if not name or name != Path(name).name or ".." in name or "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="Invalid file name")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        raise HTTPException(status_code=400, detail="Invalid file name")

    settings = get_settings()
    root = (settings.workspace_path / "downloads").resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = (root / name).resolve()
    if not str(path).startswith(str(root) + "/") and path != root:
        raise HTTPException(status_code=403, detail="Path escape blocked")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    import mimetypes

    media, _ = mimetypes.guess_type(name)
    return FileResponse(
        path,
        media_type=media or "application/octet-stream",
        filename=name,
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )



@app.get("/api/settings")
async def api_get_settings() -> dict[str, Any]:
    from app import runtime_settings as rs
    data = rs.load_settings()
    return {k: v for k, v in data.items() if not str(k).startswith("_")}


@app.put("/api/settings")
async def api_put_settings(body: SettingsPatch) -> dict[str, Any]:
    from app import runtime_settings as rs
    patch = body.model_dump(exclude_none=True)
    data = rs.save_settings(patch)
    return {k: v for k, v in data.items() if not str(k).startswith("_")}


@app.post("/api/settings/allow-once")
async def api_allow_once(body: AllowOnceBody) -> dict[str, Any]:
    """Grant a one-shot approval for a risky tool (Auto-review)."""
    from app import runtime_settings as rs
    token = rs.grant_allow_once(body.tool, body.args or {})
    return {"ok": True, "token": token, "tool": body.tool}



class ApprovalDecisionBody(BaseModel):
    decision: str | None = Field(default=None, max_length=32)


@app.get("/api/approvals/{approval_id}")
async def api_get_approval(approval_id: str) -> dict[str, Any]:
    from app import approvals as approval_store
    rec = approval_store.get_approval(approval_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Approval not found")
    return approval_store.public_approval(rec)


@app.post("/api/approvals/{approval_id}/approve")
async def api_approve(approval_id: str) -> dict[str, Any]:
    """Approve a paused Auto-review tool; resumes the SAME job from the waiting tool."""
    from app import approvals as approval_store
    try:
        rec = approval_store.resolve_approval(approval_id, "approve")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not rec:
        raise HTTPException(status_code=404, detail="Approval not found")
    # Reflect on job events
    job_id = rec.get("job_id")
    if job_id:
        job = jobstore.load_job(str(job_id))
        if job:
            jobstore.append_event(
                job,
                "approval",
                approval_id=approval_id,
                tool=rec.get("tool"),
                status="approved",
                message="User approved — resuming tool",
                args_summary=rec.get("args_summary"),
                risk=rec.get("risk"),
                args=rec.get("args") or {},
            )
            if job.get("status") == "awaiting_approval":
                job["status"] = "running"
            jobstore.save_job(job)
    return {"ok": True, **approval_store.public_approval(rec)}


@app.post("/api/approvals/{approval_id}/decline")
async def api_decline(approval_id: str) -> dict[str, Any]:
    """Decline a paused tool; the agent continues with a declined tool result."""
    from app import approvals as approval_store
    try:
        rec = approval_store.resolve_approval(approval_id, "decline")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not rec:
        raise HTTPException(status_code=404, detail="Approval not found")
    job_id = rec.get("job_id")
    if job_id:
        job = jobstore.load_job(str(job_id))
        if job:
            jobstore.append_event(
                job,
                "approval",
                approval_id=approval_id,
                tool=rec.get("tool"),
                status="declined",
                message="User declined tool",
                args_summary=rec.get("args_summary"),
                risk=rec.get("risk"),
            )
            if job.get("status") == "awaiting_approval":
                job["status"] = "running"
            jobstore.save_job(job)
    return {"ok": True, **approval_store.public_approval(rec)}


@app.post("/api/approvals/{approval_id}/allow-once")
async def api_allow_once_approval(approval_id: str) -> dict[str, Any]:
    from app import approvals as approval_store
    try:
        rec = approval_store.resolve_approval(approval_id, "allow_once")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not rec:
        raise HTTPException(status_code=404, detail="Approval not found")
    job_id = rec.get("job_id")
    if job_id:
        job = jobstore.load_job(str(job_id))
        if job:
            jobstore.append_event(
                job,
                "approval",
                approval_id=approval_id,
                tool=rec.get("tool"),
                status="approved",
                message="User allowed once — resuming tool",
                args_summary=rec.get("args_summary"),
                risk=rec.get("risk"),
                args=rec.get("args") or {},
            )
            if job.get("status") == "awaiting_approval":
                job["status"] = "running"
            jobstore.save_job(job)
    return {"ok": True, **approval_store.public_approval(rec)}


@app.post("/api/auth/login")
async def api_auth_login(body: AuthLoginBody, response: Response) -> dict[str, Any]:
    email = body.email.strip().lower()
    users = _load_users()
    user = users.get(email)
    if not user:
        # allow first-time register with password or pin
        if not (body.password or body.pin):
            raise HTTPException(status_code=401, detail="Unknown account — provide password or PIN to create one")
        name = email.split("@")[0].replace(".", " ").replace("_", " ")
        parts = [p for p in name.split() if p]
        initials = ((parts[0][0] if parts else "U") + (parts[1][0] if len(parts) > 1 else "N")).upper()
        user = {
            "email": email,
            "name": name.title(),
            "initials": initials,
            "password_sha256": _hash_pw(body.password) if body.password else None,
            "pin": (body.pin or "").strip() or None,
        }
        users[email] = user
        _save_users(users)
    else:
        ok = False
        if body.password and user.get("password_sha256"):
            ok = hmac.compare_digest(user["password_sha256"], _hash_pw(body.password))
        if body.pin and user.get("pin"):
            ok = ok or hmac.compare_digest(str(user["pin"]), str(body.pin).strip())
        # demo convenience: default account accepts nitc1234 / 1234 even if file drifted
        if email == "polycapsamuel5@gmail.com":
            if body.password == "nitc1234" or body.pin == "1234":
                ok = True
        if not ok:
            raise HTTPException(status_code=401, detail="Invalid email, password, or PIN")
    token = _make_session(email, user)
    response.set_cookie(
        key="nitc_session",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=60 * 60 * 24 * 30,
    )
    return {
        "ok": True,
        "token": token,
        "user": {
            "email": email,
            "name": user.get("name"),
            "initials": user.get("initials") or "CO",
        },
    }


@app.post("/api/auth/logout")
async def api_auth_logout(request: Request, response: Response) -> dict[str, Any]:
    sess = _session_from_request(request)
    if sess and sess.get("token"):
        _SESSIONS.pop(sess["token"], None)
    response.delete_cookie("nitc_session")
    return {"ok": True}


@app.get("/api/auth/me")
async def api_auth_me(request: Request) -> dict[str, Any]:
    sess = _session_from_request(request)
    if not sess:
        return {"authenticated": False}
    return {
        "authenticated": True,
        "user": {
            "email": sess.get("email"),
            "name": sess.get("name"),
            "initials": sess.get("initials") or "CO",
        },
    }


class FeedbackBody(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    email: str | None = None
    name: str | None = None
    version: str | None = None


@app.post("/api/feedback")
async def api_feedback(body: FeedbackBody) -> dict[str, Any]:
    """Append user feedback as a JSON line under workspace/feedback.jsonl."""
    settings = get_settings()
    path = settings.workspace_path / "feedback.jsonl"
    import time as _time
    rec = {
        "ts": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        "message": body.message.strip(),
        "email": (body.email or "").strip() or None,
        "name": (body.name or "").strip() or None,
        "version": body.version or __version__,
    }
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.exception("feedback write failed")
        raise HTTPException(status_code=500, detail=f"Could not save feedback: {exc}") from exc
    return {"ok": True}


@app.post("/api/chat", response_model=ChatResponse)
async def api_chat(body: ChatRequest) -> ChatResponse:
    messages: list[dict[str, Any]] = []
    for m in body.messages:
        messages.append({"role": m.role, "content": m.content})
    if body.message:
        messages.append({"role": "user", "content": body.message})
    if not messages:
        raise HTTPException(status_code=400, detail="Provide messages or message")

    cleaned: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if role not in ("user", "assistant"):
            continue
        if isinstance(content, str):
            cleaned.append({"role": role, "content": content})
        elif isinstance(content, list):
            # multimodal parts from client (rare); keep as-is
            cleaned.append({"role": role, "content": content})
        else:
            cleaned.append({"role": role, "content": str(content or "")})

    if not cleaned or cleaned[-1]["role"] != "user":
        raise HTTPException(status_code=400, detail="Last message must be from user")

    att_dicts = [a.model_dump(exclude_none=True) for a in body.attachments]
    # Drop dangling refs early
    if att_dicts:
        valid = []
        for a in att_dicts:
            if resolve_upload(a) is not None or a.get("path") or a.get("id"):
                valid.append(a)
        att_dicts = valid

    try:
        result = await agent_chat(cleaned, attachments=att_dicts or None)
    except Exception as exc:  # noqa: BLE001
        logger.exception("chat failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatResponse(
        reply=result["reply"],
        tool_rounds=result.get("tool_rounds", 0),
        messages=result.get("messages", []),
        images=result.get("images", []),
    )


def _clean_messages(body: ChatRequest) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for m in body.messages:
        messages.append({"role": m.role, "content": m.content})
    if body.message:
        messages.append({"role": "user", "content": body.message})
    if not messages:
        raise HTTPException(status_code=400, detail="Provide messages or message")

    cleaned: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if role not in ("user", "assistant"):
            continue
        if isinstance(content, str):
            cleaned.append({"role": role, "content": content})
        elif isinstance(content, list):
            cleaned.append({"role": role, "content": content})
        else:
            cleaned.append({"role": role, "content": str(content or "")})

    if not cleaned or cleaned[-1]["role"] != "user":
        raise HTTPException(status_code=400, detail="Last message must be from user")
    return cleaned


def _valid_attachments(body: ChatRequest) -> list[dict[str, Any]]:
    att_dicts = [a.model_dump(exclude_none=True) for a in body.attachments]
    if not att_dicts:
        return []
    valid = []
    for a in att_dicts:
        if resolve_upload(a) is not None or a.get("path") or a.get("id"):
            valid.append(a)
    return valid


@app.post("/api/jobs", response_model=JobCreateResponse)
@app.post("/api/chat/async", response_model=JobCreateResponse)
async def api_create_job(body: ChatRequest) -> JobCreateResponse:
    """Accept a chat turn immediately; agent runs on the server in the background."""
    cleaned = _clean_messages(body)
    att_dicts = _valid_attachments(body)
    sid = (body.bot_id or body.session_id or "default").strip() or "default"
    job = jobstore.create_job(
        messages=cleaned,
        attachments=att_dicts or None,
        session_id=sid,
        bot_id=sid,
        instructions=body.instructions,
    )
    jobstore.enqueue_job(job["id"])
    return JobCreateResponse(
        job_id=job["id"],
        status=job["status"],
        session_id=sid,
        bot_id=sid,
    )


@app.get("/api/jobs")
async def api_list_jobs(
    session_id: str | None = None,
    bot_id: str | None = None,
    status: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    sid = bot_id or session_id
    items = jobstore.list_jobs(session_id=sid, limit=min(max(limit, 1), 100), status=status)
    return {"jobs": items}


@app.get("/api/jobs/{job_id}")
async def api_get_job(job_id: str) -> dict[str, Any]:
    job = jobstore.load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobstore.public_job(job)


@app.get("/api/jobs/{job_id}/events")
async def api_job_events(job_id: str, request: Request) -> StreamingResponse:
    """SSE stream of job status until completed/failed/cancelled."""
    job = jobstore.load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_gen():
        last_status = None
        last_reply = None
        last_events = -1
        for _ in range(3600):  # ~30 min at 0.5s
            if await request.is_disconnected():
                break
            current = jobstore.load_job(job_id)
            if not current:
                yield "event: error\ndata: {\"detail\":\"missing\"}\n\n"
                break
            status = current.get("status")
            reply = current.get("reply") or current.get("partial_reply") or ""
            n_events = len(current.get("events") or [])
            if status != last_status or reply != last_reply or n_events != last_events:
                payload = json.dumps(jobstore.public_job(current), ensure_ascii=False)
                yield f"event: job\ndata: {payload}\n\n"
                last_status = status
                last_reply = reply
                last_events = n_events
            if status in ("completed", "failed", "cancelled"):
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/jobs/{job_id}/retry", response_model=JobCreateResponse)
async def api_retry_job(job_id: str) -> JobCreateResponse:
    job = await jobstore.retry_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobCreateResponse(
        job_id=job["id"],
        status=job["status"],
        session_id=job.get("session_id"),
        bot_id=job.get("bot_id") or job.get("session_id"),
    )


@app.get("/api/sessions/{session_id}")
async def api_get_session(session_id: str) -> dict[str, Any]:
    data = jobstore.load_session(session_id)
    if not data:
        return {"session_id": session_id, "messages": [], "jobs": [], "active_jobs": []}
    # Attach recent/active jobs for reconnect
    active = jobstore.list_jobs(session_id=session_id, limit=20)
    running = [j for j in active if j.get("status") in ("queued", "running", "awaiting_approval")]
    return {
        "session_id": session_id,
        "messages": data.get("messages") or [],
        "jobs": data.get("jobs") or [],
        "updated_at": data.get("updated_at"),
        "active_jobs": running,
        "recent_jobs": active[:10],
    }


class BotsPutBody(BaseModel):
    bots: list[dict[str, Any]] = Field(default_factory=list)


@app.get("/api/bots")
async def api_get_bots() -> dict[str, Any]:
    """Durable bot registry (survives refresh / Safari private wipe of localStorage)."""
    from app import bots_store
    return bots_store.public_bots_payload()


@app.put("/api/bots")
async def api_put_bots(body: BotsPutBody) -> dict[str, Any]:
    from app import bots_store
    saved = bots_store.save_bots(body.bots or [])
    return bots_store.public_bots_payload(saved)


# ---------------------------------------------------------------------------
# Desktop-api reverse proxy (clipboard / mouse / type) → desktop:7090
_DESKTOP_PROXY_ALLOW = frozenset({
    "health",
    "clipboard",
    "click",
    "type",
    "hotkey",
    "scroll",
    "mouse",
    "screenshot",
})


@app.get("/api/desktop/screenshot")
async def api_desktop_live_screenshot() -> FileResponse:
    """Capture desktop now and return PNG (compat live-view for mobile when RFB fails)."""
    settings = get_settings()
    upstream = f"{settings.desktop_api_url.rstrip('/')}/screenshot"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=8.0)) as client:
            resp = await client.post(upstream)
    except httpx.ConnectError as exc:
        raise HTTPException(status_code=502, detail="desktop-api unreachable") from exc
    if resp.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"screenshot failed: HTTP {resp.status_code}")
    try:
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail="screenshot returned non-JSON") from exc
    rel = str(data.get("path") or "").lstrip("/")
    if not rel.startswith("screenshots/") or ".." in rel:
        raise HTTPException(status_code=502, detail="invalid screenshot path")
    shot = (settings.workspace_path / rel).resolve()
    shot_root = (settings.workspace_path / "screenshots").resolve()
    if not str(shot).startswith(str(shot_root)) or not shot.is_file():
        raise HTTPException(status_code=404, detail="screenshot file missing")
    return FileResponse(
        shot,
        media_type="image/png",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@app.api_route(
    "/api/desktop/{full_path:path}",
    methods=["GET", "POST"],
)
async def desktop_api_proxy(full_path: str, request: Request) -> Response:
    """Same-origin proxy to the desktop container control API (limited routes)."""
    path = full_path.strip("/")
    root = path.split("/", 1)[0] if path else ""
    if root not in _DESKTOP_PROXY_ALLOW:
        raise HTTPException(status_code=404, detail="Unknown desktop route")

    settings = get_settings()
    upstream = f"{settings.desktop_api_url.rstrip('/')}/{path}"
    if request.url.query:
        upstream = f"{upstream}?{request.url.query}"

    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in _HOP_BY_HOP and k.lower() not in {"content-length"}
    }
    body = await request.body()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=8.0)) as client:
            resp = await client.request(
                request.method,
                upstream,
                headers=headers,
                content=body if body else None,
                follow_redirects=False,
            )
    except httpx.ConnectError as exc:
        logger.warning("desktop-api unreachable: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="desktop-api unreachable. Is the desktop container running?",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("desktop-api proxy error")
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    out_headers = {
        k: v
        for k, v in resp.headers.items()
        if k.lower() not in _HOP_BY_HOP and k.lower() not in {"content-encoding", "content-length"}
    }
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=out_headers,
        media_type=resp.headers.get("content-type"),
    )


# noVNC reverse proxy (HTTP + WebSocket) → desktop:6080
# Mobile carriers often block :6080; same-origin /novnc/ on :8080 fixes that.
# ---------------------------------------------------------------------------


@app.api_route("/novnc", methods=["GET", "HEAD"])
@app.api_route("/novnc/", methods=["GET", "HEAD"])
async def novnc_root_redirect() -> Response:
    return Response(status_code=307, headers={"Location": "/novnc/vnc.html"})


@app.api_route(
    "/novnc/{full_path:path}",
    methods=["GET", "HEAD", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
)
async def novnc_http_proxy(full_path: str, request: Request) -> Response:
    upstream = f"{_novnc_upstream()}/{full_path}"
    if request.url.query:
        upstream = f"{upstream}?{request.url.query}"

    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in _HOP_BY_HOP
    }
    body = await request.body()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
            resp = await client.request(
                request.method,
                upstream,
                headers=headers,
                content=body,
                follow_redirects=False,
            )
    except httpx.ConnectError as exc:
        logger.warning("novnc upstream unreachable: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="noVNC upstream (desktop:6080) unreachable. Is the desktop container running?",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("novnc proxy error")
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    out_headers = {
        k: v
        for k, v in resp.headers.items()
        if k.lower() not in _HOP_BY_HOP and k.lower() not in {"content-encoding"}
    }
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=out_headers,
        media_type=resp.headers.get("content-type"),
    )


@app.websocket("/novnc/{full_path:path}")
async def novnc_ws_proxy(websocket: WebSocket, full_path: str) -> None:
    """Binary-only RFB proxy. Avoid ping frames / dual subprotocols that break mobile RFB."""
    base = _novnc_upstream().replace("https://", "wss://").replace("http://", "ws://")
    # Client may request /novnc/websockify — upstream websockify listens at /websockify
    upstream_path = full_path
    if full_path.startswith("novnc/"):
        upstream_path = full_path[len("novnc/"):]
    if upstream_path in ("", "websockify", "websockify/"):
        upstream_path = "websockify"
    target = f"{base}/{upstream_path}"
    if websocket.scope.get("query_string"):
        qs = websocket.scope["query_string"].decode("utf-8", errors="replace")
        if qs:
            target = f"{target}?{qs}"

    # Match binary only — offering base64+binary makes some clients dual-negotiate and corrupt RFB
    proto_header = websocket.headers.get("sec-websocket-protocol")
    client_protos = [p.strip() for p in proto_header.split(",") if p.strip()] if proto_header else []
    if client_protos and "binary" not in client_protos:
        logger.warning("novnc client subprotocols without binary: %s", client_protos)
    chosen = "binary"

    try:
        async with websockets.connect(
            target,
            subprotocols=["binary"],
            open_timeout=15,
            max_size=8 * 1024 * 1024,
            ping_interval=None,
            ping_timeout=None,
            compression=None,
            max_queue=None,
        ) as upstream:
            # Accept only after upstream is ready (single connection; binary subprotocol)
            await websocket.accept(subprotocol=chosen)

            async def client_to_upstream() -> None:
                try:
                    while True:
                        msg = await websocket.receive()
                        if msg["type"] == "websocket.disconnect":
                            break
                        data = msg.get("bytes")
                        if data is None and msg.get("text") is not None:
                            # RFB is binary; coerce any text frames to bytes without reinterpret
                            data = msg["text"].encode("latin-1", errors="replace")
                        if data is not None:
                            await upstream.send(data)
                except WebSocketDisconnect:
                    pass
                except Exception:  # noqa: BLE001
                    logger.debug("client_to_upstream closed", exc_info=True)

            async def upstream_to_client() -> None:
                try:
                    async for message in upstream:
                        if websocket.client_state != WebSocketState.CONNECTED:
                            break
                        if isinstance(message, str):
                            message = message.encode("latin-1", errors="replace")
                        await websocket.send_bytes(message)
                except Exception:  # noqa: BLE001
                    logger.debug("upstream_to_client closed", exc_info=True)

            await asyncio.gather(client_to_upstream(), upstream_to_client())
    except Exception as exc:  # noqa: BLE001
        logger.warning("novnc websocket proxy failed: %s", exc)
        if websocket.client_state.name == "CONNECTING":
            try:
                await websocket.close(code=1011)
            except Exception:  # noqa: BLE001
                pass
    finally:
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.close()
            except Exception:  # noqa: BLE001
                pass


@app.get("/")
async def index() -> FileResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(
        index_path,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        },
    )


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
