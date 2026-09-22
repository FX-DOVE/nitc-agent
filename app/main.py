"""FastAPI entrypoint: health, chat API, noVNC proxy, media, and static chat UI."""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any

import httpx
import websockets
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.websockets import WebSocketState

from app import __version__
from app.agent import chat as agent_chat
from app.config import get_settings

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

app = FastAPI(
    title="Nitc Agent",
    description="Self-hosted AI agent with shell, files, interactive desktop, browser, and GitHub tools.",
    version=__version__,
)


class ChatMessage(BaseModel):
    role: str
    content: str = ""


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(default_factory=list)
    message: str | None = Field(
        default=None,
        description="Convenience: single user message (appended if messages empty or in addition)",
    )


class ChatResponse(BaseModel):
    reply: str
    tool_rounds: int = 0
    messages: list[dict[str, Any]] = Field(default_factory=list)
    images: list[str] = Field(default_factory=list)


def _novnc_upstream() -> str:
    return get_settings().novnc_upstream.rstrip("/")


def _novnc_embed_url() -> str:
    # path= tells noVNC to open the WebSocket under /novnc/websockify (same origin).
    return "/novnc/vnc.html?autoconnect=1&resize=scale&path=novnc/websockify"


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
    }


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


@app.post("/api/chat", response_model=ChatResponse)
async def api_chat(body: ChatRequest) -> ChatResponse:
    messages: list[dict[str, Any]] = [m.model_dump() for m in body.messages]
    if body.message:
        messages.append({"role": "user", "content": body.message})
    if not messages:
        raise HTTPException(status_code=400, detail="Provide messages or message")

    cleaned = []
    for m in messages:
        if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str):
            cleaned.append({"role": m["role"], "content": m["content"]})

    if not cleaned or cleaned[-1]["role"] != "user":
        raise HTTPException(status_code=400, detail="Last message must be from user")

    try:
        result = await agent_chat(cleaned)
    except Exception as exc:  # noqa: BLE001
        logger.exception("chat failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatResponse(
        reply=result["reply"],
        tool_rounds=result.get("tool_rounds", 0),
        messages=result.get("messages", []),
        images=result.get("images", []),
    )


# ---------------------------------------------------------------------------
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
    await websocket.accept()
    base = _novnc_upstream().replace("https://", "wss://").replace("http://", "ws://")
    target = f"{base}/{full_path}"
    if websocket.scope.get("query_string"):
        qs = websocket.scope["query_string"].decode("utf-8", errors="replace")
        if qs:
            target = f"{target}?{qs}"

    # Prefer binary subprotocol used by websockify / noVNC when offered.
    subprotocols: list[str] = []
    proto_header = websocket.headers.get("sec-websocket-protocol")
    if proto_header:
        subprotocols = [p.strip() for p in proto_header.split(",") if p.strip()]

    try:
        async with websockets.connect(
            target,
            subprotocols=subprotocols or None,
            open_timeout=15,
            max_size=8 * 1024 * 1024,
        ) as upstream:
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
    return FileResponse(index_path)


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
