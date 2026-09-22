"""Persisted UI/runtime settings under workspace/settings.json (auto-review, etc.)."""

from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from pathlib import Path
from typing import Any

from app.config import get_settings

logger = logging.getLogger("nitc.settings")
_lock = threading.RLock()

# shell + interactive desktop + browser navigate (network) — screenshots allowed
RISKY_TOOLS = frozenset({
    "shell",
    "desktop_click",
    "desktop_type",
    "desktop_hotkey",
    "desktop_scroll",
    "desktop_open_browser",
    "browser_navigate",
    "github_run",
    "write_file",
})

DEFAULTS: dict[str, Any] = {
    "autoReview": False,
    "autoReviewRules": [],
    "tzAuto": True,
    "timeZone": "Africa/Lagos",
    "notifications": True,
    "appearance": "Black",
    "language": "System",
    "haptics": True,
    "plugins": {
        "browser": True,
        "desktop": True,
        "files": True,
        "shell": True,
        "github": True,
    },
}


def _path() -> Path:
    return get_settings().workspace_path / "settings.json"


def load_settings() -> dict[str, Any]:
    with _lock:
        path = _path()
        data = dict(DEFAULTS)
        if path.is_file():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    data.update(raw)
            except Exception:  # noqa: BLE001
                logger.exception("failed reading settings.json")
        if not isinstance(data.get("autoReviewRules"), list):
            data["autoReviewRules"] = []
        if not isinstance(data.get("plugins"), dict):
            data["plugins"] = dict(DEFAULTS["plugins"])
        # allow-once tokens live in memory + optional file field
        data.setdefault("_allow_once", [])
        return data


def save_settings(patch: dict[str, Any]) -> dict[str, Any]:
    with _lock:
        cur = load_settings()
        for k, v in patch.items():
            if k.startswith("_"):
                continue
            cur[k] = v
        # preserve allow list in file lightly
        path = _path()
        to_write = {k: v for k, v in cur.items() if not k.startswith("_") or k == "_allow_once"}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(to_write, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return cur


def grant_allow_once(tool: str, args: dict[str, Any] | None = None, ttl_sec: int = 120) -> str:
    """Permit one matching risky tool call within ttl_sec. Returns token id."""
    token = secrets.token_urlsafe(12)
    with _lock:
        cur = load_settings()
        allows = list(cur.get("_allow_once") or [])
        allows.append({
            "id": token,
            "tool": tool,
            "args": args or {},
            "expires": time.time() + ttl_sec,
        })
        # prune expired
        now = time.time()
        allows = [a for a in allows if float(a.get("expires") or 0) > now]
        cur["_allow_once"] = allows
        path = _path()
        to_write = {k: v for k, v in cur.items() if not k.startswith("_") or k == "_allow_once"}
        path.write_text(json.dumps(to_write, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return token


def consume_allow_once(tool: str, args: dict[str, Any]) -> bool:
    """Return True and consume if a matching allow-once grant exists."""
    with _lock:
        cur = load_settings()
        allows = list(cur.get("_allow_once") or [])
        now = time.time()
        kept: list[dict[str, Any]] = []
        matched = False
        for a in allows:
            if float(a.get("expires") or 0) <= now:
                continue
            if matched:
                kept.append(a)
                continue
            if a.get("tool") != tool:
                kept.append(a)
                continue
            # match — consume
            matched = True
        if matched:
            cur["_allow_once"] = kept
            path = _path()
            to_write = {k: v for k, v in cur.items() if not k.startswith("_") or k == "_allow_once"}
            path.write_text(json.dumps(to_write, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return matched


def auto_review_enabled() -> bool:
    return bool(load_settings().get("autoReview"))


def auto_review_rules() -> list[str]:
    rules = load_settings().get("autoReviewRules") or []
    return [str(r) for r in rules if str(r).strip()]
