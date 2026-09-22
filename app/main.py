"""FastAPI entrypoint: health, chat API, and static chat UI."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import __version__
from app.agent import chat as agent_chat
from app.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("nitc.main")

STATIC_DIR = Path(__file__).parent / "static"

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
    }


@app.get("/api/config")
async def api_config() -> dict[str, Any]:
    """Public UI config (no secrets)."""
    settings = get_settings()
    novnc = settings.novnc_public_url.rstrip("/")
    return {
        "version": __version__,
        "novnc_public_url": novnc,
        "novnc_embed_url": f"{novnc}/vnc.html?autoconnect=1&resize=scale",
        "model": settings.model,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def api_chat(body: ChatRequest) -> ChatResponse:
    messages: list[dict[str, Any]] = [m.model_dump() for m in body.messages]
    if body.message:
        messages.append({"role": "user", "content": body.message})
    if not messages:
        raise HTTPException(status_code=400, detail="Provide messages or message")

    # Only pass user/assistant text turns from the client (strip tool noise)
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
    )


@app.get("/")
async def index() -> FileResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(index_path)


# Mount static assets (css/js if added later)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
