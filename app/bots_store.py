"""Server-side bot registry: durable list in workspace/bots.json."""

from __future__ import annotations

import json
import logging
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import get_settings

logger = logging.getLogger("nitc.bots")

_SAFE_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_lock = threading.RLock()

_BOT_FIELDS = (
    "id",
    "name",
    "color",
    "shape",
    "kind",
    "pinned",
    "pinnedAt",
    "unread",
    "section",
    "hidden",
    "systemPrompt",
    "instructions",
    "snippet",
    "updatedAt",
    "updated_at",
    "last_snippet",
    "activeJobId",
    "activeJobIds",
    "pendingJobId",
    "lastJobId",
)


def bots_path() -> Path:
    root = get_settings().workspace_path
    root.mkdir(parents=True, exist_ok=True)
    return root / "bots.json"


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_bot(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    bid = str(raw.get("id") or "").strip()
    if not bid:
        return None
    if not _SAFE_ID.match(bid):
        bid = re.sub(r"[^A-Za-z0-9_.:-]+", "_", bid)[:128] or "bot"
    out: dict[str, Any] = {"id": bid}
    for key in _BOT_FIELDS:
        if key == "id":
            continue
        if key in raw and raw[key] is not None:
            out[key] = raw[key]
    # Normalize timestamps
    updated = out.get("updatedAt") or out.get("updated_at")
    if isinstance(updated, str):
        try:
            # ISO → ms
            updated = int(datetime.fromisoformat(updated.replace("Z", "+00:00")).timestamp() * 1000)
        except Exception:
            updated = _now_ms()
    if not isinstance(updated, (int, float)):
        updated = _now_ms()
    out["updatedAt"] = int(updated)
    out["updated_at"] = _now_iso()
    if "name" not in out or not str(out.get("name") or "").strip():
        out["name"] = "Bot"
    out["name"] = str(out["name"])[:80]
    snippet = out.get("last_snippet") or out.get("snippet") or ""
    out["snippet"] = str(snippet).replace("\n", " ")[:120]
    out["last_snippet"] = out["snippet"]
    # Never store full message history in bots.json (sessions hold that)
    out.pop("messages", None)
    return out


def load_bots() -> list[dict[str, Any]]:
    path = bots_path()
    if not path.is_file():
        return []
    try:
        with _lock:
            data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("bots"), list):
            items = data["bots"]
        elif isinstance(data, list):
            items = data
        else:
            return []
        out: list[dict[str, Any]] = []
        for item in items:
            bot = _sanitize_bot(item)
            if bot:
                out.append(bot)
        return out
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("load_bots failed: %s", exc)
        return []


def save_bots(bots: list[Any]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in bots or []:
        bot = _sanitize_bot(item)
        if not bot:
            continue
        if bot["id"] in seen:
            continue
        seen.add(bot["id"])
        cleaned.append(bot)
    # Cap registry size
    cleaned = cleaned[:200]
    path = bots_path()
    payload = {
        "bots": cleaned,
        "updated_at": _now_iso(),
    }
    tmp = path.with_suffix(".tmp")
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    with _lock:
        tmp.write_text(text + "\n", encoding="utf-8")
        tmp.replace(path)
    return cleaned


def public_bots_payload(bots: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    items = bots if bots is not None else load_bots()
    return {
        "bots": items,
        "updated_at": _now_iso(),
        "count": len(items),
    }
