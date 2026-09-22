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


def zip_paths(paths: list[str] | None = None, output: str | None = None) -> str:
    """Zip workspace-relative paths into workspace/downloads/ and return a media URL."""
    import zipfile
    from datetime import datetime, timezone

    workspace = get_settings().workspace_path
    downloads = workspace / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)

    srcs = paths or ["."]
    if isinstance(srcs, str):
        srcs = [srcs]
    resolved: list[Path] = []
    for p in srcs:
        try:
            target = _resolve_safe(str(p))
        except ValueError as exc:
            return json.dumps({"ok": False, "error": str(exc)})
        if not target.exists():
            return json.dumps({"ok": False, "error": f"Not found: {p}"})
        resolved.append(target)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_name = (output or f"bundle-{stamp}.zip").strip()
    out_name = Path(out_name).name
    if not out_name.lower().endswith(".zip"):
        out_name += ".zip"
    out_path = downloads / out_name
    # avoid clobber
    if out_path.exists():
        out_path = downloads / f"{out_path.stem}-{stamp}.zip"
        out_name = out_path.name

    try:
        with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for target in resolved:
                if target.is_file():
                    arc = str(target.relative_to(workspace))
                    zf.write(target, arcname=arc)
                else:
                    for child in target.rglob("*"):
                        if child.is_file():
                            # skip nested downloads of previous zips optionally
                            if "downloads" in child.parts and child.suffix == ".zip":
                                continue
                            arc = str(child.relative_to(workspace))
                            zf.write(child, arcname=arc)
    except OSError as exc:
        return json.dumps({"ok": False, "error": str(exc)})

    rel = f"downloads/{out_name}"
    url = f"/api/media/files/{out_name}"
    return json.dumps(
        {
            "ok": True,
            "path": rel,
            "filename": out_name,
            "url": url,
            "bytes": out_path.stat().st_size,
            "download_markdown": f"[Download {out_name}]({url})",
        }
    )
