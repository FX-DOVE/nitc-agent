"""Workspace-scoped filesystem tools."""

from __future__ import annotations

import json
from pathlib import Path

from app.config import get_settings


def _resolve_safe(path: str) -> Path:
    """Resolve path under workspace; raise ValueError on escape attempts."""
    workspace = get_settings().workspace_path
    # Disallow absolute paths that are not under workspace
    candidate = (workspace / path).resolve()
    try:
        candidate.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"Path escapes workspace: {path}") from exc
    return candidate


def read_file(path: str) -> str:
    target = _resolve_safe(path)
    if not target.exists():
        return json.dumps({"ok": False, "error": f"File not found: {path}"})
    if not target.is_file():
        return json.dumps({"ok": False, "error": f"Not a file: {path}"})
    if target.stat().st_size > 2_000_000:
        return json.dumps({"ok": False, "error": "File too large (>2MB)"})
    text = target.read_text(encoding="utf-8", errors="replace")
    return json.dumps({"ok": True, "path": path, "content": text})


def write_file(path: str, content: str) -> str:
    target = _resolve_safe(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return json.dumps(
        {
            "ok": True,
            "path": path,
            "bytes": len(content.encode("utf-8")),
        }
    )


def list_dir(path: str = ".") -> str:
    target = _resolve_safe(path or ".")
    if not target.exists():
        return json.dumps({"ok": False, "error": f"Directory not found: {path}"})
    if not target.is_dir():
        return json.dumps({"ok": False, "error": f"Not a directory: {path}"})

    entries = []
    for child in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        entries.append(
            {
                "name": child.name,
                "type": "dir" if child.is_dir() else "file",
                "size": None if child.is_dir() else child.stat().st_size,
            }
        )
    return json.dumps({"ok": True, "path": path or ".", "entries": entries})
