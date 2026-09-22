"""Upload storage helpers and attachment → model-context enrichment."""

from __future__ import annotations

import base64
import logging
import mimetypes
import re
import uuid
from pathlib import Path
from typing import Any

import httpx

from app.config import get_settings

logger = logging.getLogger("nitc.attachments")

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB
MAX_TEXT_EXCERPT = 24_000
MAX_IMAGE_BYTES_FOR_VISION = 4 * 1024 * 1024  # 4 MB inline

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def uploads_root() -> Path:
    root = get_settings().workspace_path / "uploads"
    root.mkdir(parents=True, exist_ok=True)
    return root


def safe_filename(name: str) -> str:
    base = Path(name or "file").name
    base = _SAFE_NAME.sub("_", base).strip("._") or "file"
    return base[:180]


def save_upload(data: bytes, filename: str, content_type: str | None = None) -> dict[str, Any]:
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(f"File too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)}MB)")
    if not data:
        raise ValueError("Empty file")

    uid = uuid.uuid4().hex[:12]
    clean = safe_filename(filename)
    stored = f"{uid}_{clean}"
    path = uploads_root() / stored
    path.write_bytes(data)

    mime = (content_type or "").split(";")[0].strip().lower()
    if not mime or mime == "application/octet-stream":
        guessed, _ = mimetypes.guess_type(clean)
        mime = (guessed or "application/octet-stream").lower()

    rel = f"uploads/{stored}"
    return {
        "id": uid,
        "path": rel,
        "name": clean,
        "mime": mime,
        "size": len(data),
        "url": f"/api/media/uploads/{stored}",
    }


def resolve_upload(ref: dict[str, Any]) -> Path | None:
    """Resolve an attachment ref to an absolute path under workspace/uploads."""
    workspace = get_settings().workspace_path
    uploads = uploads_root()

    raw = (ref.get("path") or "").strip()
    if not raw and ref.get("id"):
        # find by id prefix
        prefix = str(ref["id"])
        for child in uploads.iterdir():
            if child.is_file() and child.name.startswith(prefix + "_"):
                return child
        return None

    if not raw:
        return None

    # Accept "uploads/..." or bare stored filename
    name = Path(raw).name
    if ".." in name or "/" in name or "\\" in name:
        return None
    candidate = (uploads / name).resolve()
    try:
        candidate.relative_to(uploads.resolve())
    except ValueError:
        return None
    if not candidate.is_file():
        # also try relative to workspace
        alt = (workspace / raw).resolve()
        try:
            alt.relative_to(workspace.resolve())
        except ValueError:
            return None
        return alt if alt.is_file() else None
    return candidate


def _is_image(mime: str, name: str) -> bool:
    if mime.startswith("image/"):
        return True
    return Path(name).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


def _is_audio(mime: str, name: str) -> bool:
    if mime.startswith("audio/"):
        return True
    return Path(name).suffix.lower() in {".mp3", ".wav", ".m4a", ".ogg", ".webm", ".aac", ".flac"}


def _is_textish(mime: str, name: str) -> bool:
    if mime.startswith("text/"):
        return True
    if mime in {
        "application/json",
        "application/xml",
        "application/javascript",
        "application/x-javascript",
        "application/typescript",
        "application/x-yaml",
        "application/yaml",
        "application/csv",
        "application/sql",
    }:
        return True
    return Path(name).suffix.lower() in {
        ".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".xml", ".yml", ".yaml",
        ".py", ".js", ".ts", ".tsx", ".jsx", ".html", ".css", ".scss", ".sh", ".bash",
        ".rs", ".go", ".java", ".c", ".cpp", ".h", ".hpp", ".rb", ".php", ".sql",
        ".toml", ".ini", ".cfg", ".conf", ".env", ".log", ".r", ".swift", ".kt",
    }


async def try_transcribe(path: Path, mime: str) -> str | None:
    """Best-effort OpenAI-compatible /audio/transcriptions. Returns None on failure."""
    settings = get_settings()
    if not settings.openai_api_key:
        return None
    url = settings.openai_base_url.rstrip("/") + "/audio/transcriptions"
    try:
        data = path.read_bytes()
        if len(data) > MAX_UPLOAD_BYTES:
            return None
        files = {
            "file": (path.name, data, mime or "application/octet-stream"),
        }
        form = {"model": "whisper-1"}
        headers = {"Authorization": f"Bearer {settings.openai_api_key}"}
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, headers=headers, data=form, files=files)
        if resp.status_code >= 400:
            logger.info("transcription unavailable (%s): %s", resp.status_code, resp.text[:200])
            return None
        payload = resp.json()
        text = payload.get("text") if isinstance(payload, dict) else None
        if isinstance(text, str) and text.strip():
            return text.strip()
    except Exception as exc:  # noqa: BLE001
        logger.info("transcription skipped: %s", exc)
    return None


def _image_data_url(path: Path, mime: str) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) > MAX_IMAGE_BYTES_FOR_VISION:
        return None
    mt = mime if mime.startswith("image/") else (mimetypes.guess_type(path.name)[0] or "image/png")
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mt};base64,{b64}"


async def enrich_user_message(
    text: str,
    attachments: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """
    Build a user message (possibly multimodal) that includes attachment context.

    Returns a message dict suitable for the chat completions API.
    """
    text = (text or "").strip()
    atts = [a for a in (attachments or []) if isinstance(a, dict)]
    if not atts:
        return {"role": "user", "content": text or "(empty)"}

    notes: list[str] = []
    image_parts: list[dict[str, Any]] = []

    for att in atts:
        name = att.get("name") or "file"
        mime = (att.get("mime") or "").lower()
        path = resolve_upload(att)
        rel = att.get("path") or (f"uploads/{path.name}" if path else "")
        size = att.get("size")
        size_s = f", {size} bytes" if isinstance(size, int) else ""

        if path is None:
            notes.append(f"- Missing attachment: {name} ({mime or 'unknown'}) path={rel or '?'}")
            continue

        if _is_image(mime, name):
            notes.append(
                f"- Image attached: **{name}** ({mime or 'image'}{size_s}) at workspace path `{rel}`. "
                "Use tools if you need the file on disk; a preview may also be provided as multimodal input."
            )
            data_url = _image_data_url(path, mime)
            if data_url:
                image_parts.append(
                    {"type": "image_url", "image_url": {"url": data_url}}
                )
            continue

        if _is_audio(mime, name):
            transcript = await try_transcribe(path, mime or "audio/webm")
            if transcript:
                notes.append(
                    f"- Audio attached: **{name}** ({mime or 'audio'}{size_s}) at `{rel}`.\n"
                    f"  Transcript:\n\"\"\"\n{transcript}\n\"\"\""
                )
            else:
                notes.append(
                    f"- Audio attached: **{name}** ({mime or 'audio'}{size_s}) at workspace path `{rel}`. "
                    "No transcript available; acknowledge the audio file and use tools if needed."
                )
            continue

        if _is_textish(mime, name):
            try:
                raw = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                notes.append(f"- Could not read text file **{name}**: {exc}")
                continue
            excerpt = raw[:MAX_TEXT_EXCERPT]
            truncated = len(raw) > MAX_TEXT_EXCERPT
            notes.append(
                f"- Text file attached: **{name}** ({mime or 'text'}{size_s}) at `{rel}`"
                + (" (truncated)" if truncated else "")
                + f":\n```\n{excerpt}\n```"
            )
            continue

        if Path(name).suffix.lower() == ".pdf" or mime == "application/pdf":
            notes.append(
                f"- PDF attached: **{name}** ({mime or 'application/pdf'}{size_s}) at workspace path `{rel}`. "
                "Use shell/tools (e.g. pdftotext if available) to extract text if needed."
            )
            continue

        notes.append(
            f"- File attached: **{name}** ({mime or 'application/octet-stream'}{size_s}) at workspace path `{rel}`. "
            "Use read_file / shell tools to inspect it."
        )

    header = "The user attached the following file(s) (under the agent workspace):\n" + "\n".join(notes)
    body = text if text else "Please review the attachment(s) and help me with them."
    combined = f"{body}\n\n{header}"

    if image_parts:
        parts: list[dict[str, Any]] = [{"type": "text", "text": combined}]
        parts.extend(image_parts)
        return {"role": "user", "content": parts}

    return {"role": "user", "content": combined}


def user_content_as_text(content: Any) -> str:
    """Flatten user message content (str or multimodal list) to plain text for heuristics."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        bits: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                bits.append(str(part.get("text") or ""))
        return "\n".join(bits)
    return str(content or "")
