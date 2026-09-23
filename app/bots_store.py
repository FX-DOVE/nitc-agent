"""Server-side bot registry backed by SQLite (workspace/nitc.db) with bots.json mirror."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import get_settings
from app import store


def bots_path() -> Path:
    root = get_settings().workspace_path
    root.mkdir(parents=True, exist_ok=True)
    return root / "bots.json"


def load_bots() -> list[dict[str, Any]]:
    return store.load_bots()


def save_bots(bots: list[Any]) -> list[dict[str, Any]]:
    return store.save_bots(bots)


def public_bots_payload(bots: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return store.public_bots_payload(bots)
