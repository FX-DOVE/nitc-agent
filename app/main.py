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
from fastapi.responses import FileResponse, Response, StreamingResponse
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
        for _ in range(1800):  # ~30 min at 1s
            if await request.is_disconnected():
                break
            current = jobstore.load_job(job_id)
            if not current:
                yield "event: error\ndata: {\"detail\":\"missing\"}\n\n"
                break
            status = current.get("status")
            reply = current.get("reply") or current.get("partial_reply") or ""
            if status != last_status or reply != last_reply:
                payload = json.dumps(jobstore.public_job(current), ensure_ascii=False)
                yield f"event: job\ndata: {payload}\n\n"
                last_status = status
                last_reply = reply
            if status in ("completed", "failed", "cancelled"):
                break
            await asyncio.sleep(1.0)

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
    running = [j for j in active if j.get("status") in ("queued", "running")]
    return {
        "session_id": session_id,
        "messages": data.get("messages") or [],
        "jobs": data.get("jobs") or [],
        "updated_at": data.get("updated_at"),
        "active_jobs": running,
        "recent_jobs": active[:10],
    }


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
})


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

    subprotocols: list[str] = []
    proto_header = websocket.headers.get("sec-websocket-protocol")
    if proto_header:
        subprotocols = [p.strip() for p in proto_header.split(",") if p.strip()]
    chosen = subprotocols[0] if subprotocols else None

    try:
        async with websockets.connect(
            target,
            subprotocols=subprotocols or None,
            open_timeout=15,
            max_size=8 * 1024 * 1024,
        ) as upstream:
            # Accept only after upstream is up, with matching subprotocol for noVNC.
            await websocket.accept(subprotocol=chosen)

            async def client_to_upstream() -> None:
                try:
                    while True:
                        msg = await websocket.receive()
                        if msg["type"] == "websocket.disconnect":
                            break
                        data = msg.get("bytes")
                        if data is not None:
                            await upstream.send(data)
                            continue
                        text = msg.get("text")
                        if text is not None:
                            await upstream.send(text)
                except WebSocketDisconnect:
                    pass
                except Exception:  # noqa: BLE001
                    logger.debug("client_to_upstream closed", exc_info=True)

            async def upstream_to_client() -> None:
                try:
                    async for message in upstream:
                        if websocket.client_state != WebSocketState.CONNECTED:
                            break
                        if isinstance(message, (bytes, bytearray)):
                            await websocket.send_bytes(message)
                        else:
                            await websocket.send_text(str(message))
                except Exception:  # noqa: BLE001
                    logger.debug("upstream_to_client closed", exc_info=True)

            await asyncio.gather(client_to_upstream(), upstream_to_client())
    except Exception as exc:  # noqa: BLE001
        logger.warning("novnc websocket proxy failed: %s", exc)
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
