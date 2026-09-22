"""Desktop control API — runs inside the desktop container on DISPLAY.

Exposes screenshot / click / type / hotkey / scroll / open_browser for the
agent container to call over the Docker network (http://desktop:7090).
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

DISPLAY = os.environ.get("DISPLAY", ":99")
WORKSPACE = Path(os.environ.get("DESKTOP_WORKSPACE", "/home/desktop/workspace")).resolve()
SCREENSHOT_DIR = WORKSPACE / "screenshots"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Nitc Desktop API", version="0.2.0")


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["DISPLAY"] = DISPLAY
    env.setdefault("HOME", "/home/desktop")
    return env


async def _run(*args: str, timeout: float = 30.0) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=_env(),
    )
    try:
        out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise HTTPException(status_code=504, detail=f"Command timed out: {' '.join(args)}")
    return (
        proc.returncode or 0,
        out_b.decode("utf-8", errors="replace"),
        err_b.decode("utf-8", errors="replace"),
    )


class ClickBody(BaseModel):
    x: int
    y: int
    button: Literal[1, 2, 3] = 1
    clicks: int = Field(default=1, ge=1, le=3)


class TypeBody(BaseModel):
    text: str = ""
    key: str | None = Field(
        default=None,
        description="Optional single key name after typing text (Return, Tab, Escape, …)",
    )


class HotkeyBody(BaseModel):
    keys: list[str] = Field(
        ...,
        min_length=1,
        description="Key combination, e.g. ['ctrl','c'] or ['alt','F4']",
    )


class ScrollBody(BaseModel):
    x: int
    y: int
    direction: Literal["up", "down", "left", "right"] = "down"
    amount: int = Field(default=3, ge=1, le=50)


class OpenBrowserBody(BaseModel):
    url: str


def _screen_size() -> tuple[int, int]:
    w = int(os.environ.get("SCREEN_WIDTH", "1280") or 1280)
    h = int(os.environ.get("SCREEN_HEIGHT", "800") or 800)
    return w, h


@app.get("/health")
async def health() -> dict[str, Any]:
    code, out, err = await _run("xdpyinfo", timeout=5.0)
    sw, sh = _screen_size()
    # Prefer live geometry from xdpyinfo when available
    if code == 0 and out:
        m = re.search(r"dimensions:\s+(\d+)x(\d+)", out)
        if m:
            sw, sh = int(m.group(1)), int(m.group(2))
    return {
        "status": "ok" if code == 0 else "degraded",
        "display": DISPLAY,
        "workspace": str(WORKSPACE),
        "xdpyinfo_ok": code == 0,
        "xdpyinfo_err": err.strip()[:200] if code != 0 else "",
        "screen": {"width": sw, "height": sh},
    }


@app.post("/screenshot")
async def screenshot(filename: str | None = None) -> dict[str, Any]:
    """Capture the X display; save under workspace/screenshots/."""
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    if filename:
        safe = re.sub(r"[^a-zA-Z0-9._-]", "_", filename)
        if not safe.lower().endswith(".png"):
            safe += ".png"
    else:
        safe = f"desktop_{int(time.time() * 1000)}.png"
    dest = SCREENSHOT_DIR / safe

    # Prefer scrot; fall back to ImageMagick import
    if shutil.which("scrot"):
        code, _, err = await _run("scrot", "-o", str(dest), timeout=15.0)
    elif shutil.which("import"):
        code, _, err = await _run("import", "-window", "root", str(dest), timeout=15.0)
    else:
        raise HTTPException(status_code=500, detail="Neither scrot nor import available")

    if code != 0 or not dest.exists():
        raise HTTPException(status_code=500, detail=f"Screenshot failed: {err[:500]}")

    rel = f"screenshots/{safe}"
    size = dest.stat().st_size
    return {
        "ok": True,
        "path": rel,
        "absolute": str(dest),
        "bytes": size,
        "note": f"Screenshot saved to workspace/{rel} ({size} bytes). Shared with the agent.",
    }


@app.post("/click")
async def click(body: ClickBody) -> dict[str, Any]:
    # Move then click (xdotool)
    move_code, _, move_err = await _run("xdotool", "mousemove", str(body.x), str(body.y))
    if move_code != 0:
        raise HTTPException(status_code=500, detail=f"mousemove failed: {move_err[:300]}")

    btn = str(body.button)
    args = ["xdotool", "click", "--repeat", str(body.clicks), btn]
    code, _, err = await _run(*args)
    if code != 0:
        raise HTTPException(status_code=500, detail=f"click failed: {err[:300]}")
    return {"ok": True, "x": body.x, "y": body.y, "button": body.button, "clicks": body.clicks}


@app.post("/type")
async def type_text(body: TypeBody) -> dict[str, Any]:
    if body.text:
        # --clearmodifiers avoids stuck modifiers; type literal text
        code, _, err = await _run(
            "xdotool", "type", "--clearmodifiers", "--delay", "12", "--", body.text
        )
        if code != 0:
            raise HTTPException(status_code=500, detail=f"type failed: {err[:300]}")
    if body.key:
        key = body.key.strip()
        code, _, err = await _run("xdotool", "key", "--clearmodifiers", key)
        if code != 0:
            raise HTTPException(status_code=500, detail=f"key failed: {err[:300]}")
    return {"ok": True, "text_len": len(body.text), "key": body.key}


@app.post("/hotkey")
async def hotkey(body: HotkeyBody) -> dict[str, Any]:
    # Normalize common aliases
    alias = {
        "control": "ctrl",
        "ctl": "ctrl",
        "cmd": "super",
        "win": "super",
        "option": "alt",
        "return": "Return",
        "enter": "Return",
        "esc": "Escape",
        "escape": "Escape",
        "space": "space",
        "tab": "Tab",
        "backspace": "BackSpace",
        "delete": "Delete",
    }
    parts: list[str] = []
    for k in body.keys:
        k2 = k.strip()
        low = k2.lower()
        parts.append(alias.get(low, k2 if len(k2) == 1 else (k2.capitalize() if low in ("return", "tab") else k2)))
    combo = "+".join(parts)
    code, _, err = await _run("xdotool", "key", "--clearmodifiers", combo)
    if code != 0:
        raise HTTPException(status_code=500, detail=f"hotkey failed: {err[:300]}")
    return {"ok": True, "keys": parts, "combo": combo}


@app.post("/scroll")
async def scroll(body: ScrollBody) -> dict[str, Any]:
    await _run("xdotool", "mousemove", str(body.x), str(body.y))
    # xdotool button 4=up 5=down 6=left 7=right
    button_map = {"up": "4", "down": "5", "left": "6", "right": "7"}
    btn = button_map[body.direction]
    code, _, err = await _run(
        "xdotool", "click", "--repeat", str(body.amount), "--delay", "30", btn
    )
    if code != 0:
        raise HTTPException(status_code=500, detail=f"scroll failed: {err[:300]}")
    return {
        "ok": True,
        "x": body.x,
        "y": body.y,
        "direction": body.direction,
        "amount": body.amount,
    }


@app.post("/open_browser")
async def open_browser(body: OpenBrowserBody) -> dict[str, Any]:
    url = body.url.strip()
    if not re.match(r"^https?://", url, re.I):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")

    chromium = None
    for cand in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable"):
        if shutil.which(cand):
            chromium = cand
            break
    if not chromium:
        raise HTTPException(status_code=500, detail="No Chromium/Chrome binary found")

    # Launch detached so the API returns immediately
    proc = await asyncio.create_subprocess_exec(
        chromium,
        "--no-sandbox",
        "--disable-gpu",
        "--disable-dev-shm-usage",
        "--start-maximized",
        url,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        env=_env(),
        start_new_session=True,
    )
    return {
        "ok": True,
        "url": url,
        "browser": chromium,
        "pid": proc.pid,
        "note": "Browser launched on the shared desktop. Watch/take over via noVNC.",
    }


class MouseBody(BaseModel):
    x: int
    y: int
    action: Literal["move", "down", "up", "click"] = "move"
    button: Literal[1, 2, 3] = 1


class ClipboardBody(BaseModel):
    text: str = ""


@app.post("/mouse")
async def mouse(body: MouseBody) -> dict[str, Any]:
    """Move / press / release / click mouse on the shared display."""
    move_code, _, move_err = await _run("xdotool", "mousemove", "--sync", str(body.x), str(body.y))
    if move_code != 0:
        raise HTTPException(status_code=500, detail=f"mousemove failed: {move_err[:300]}")
    btn = str(body.button)
    if body.action == "move":
        return {"ok": True, "x": body.x, "y": body.y, "action": "move"}
    if body.action == "down":
        code, _, err = await _run("xdotool", "mousedown", btn)
        if code != 0:
            raise HTTPException(status_code=500, detail=f"mousedown failed: {err[:300]}")
        return {"ok": True, "x": body.x, "y": body.y, "action": "down", "button": body.button}
    if body.action == "up":
        code, _, err = await _run("xdotool", "mouseup", btn)
        if code != 0:
            raise HTTPException(status_code=500, detail=f"mouseup failed: {err[:300]}")
        return {"ok": True, "x": body.x, "y": body.y, "action": "up", "button": body.button}
    # click
    code, _, err = await _run("xdotool", "click", btn)
    if code != 0:
        raise HTTPException(status_code=500, detail=f"click failed: {err[:300]}")
    return {"ok": True, "x": body.x, "y": body.y, "action": "click", "button": body.button}


@app.get("/clipboard")
async def get_clipboard() -> dict[str, Any]:
    """Read the X11 clipboard (CLIPBOARD selection)."""
    text_out = ""
    err_msg = ""
    if shutil.which("xclip"):
        code, out, err = await _run("xclip", "-selection", "clipboard", "-o", timeout=5.0)
        if code == 0:
            text_out = out
        else:
            err_msg = err.strip()[:300]
            # Fallback primary selection
            code2, out2, err2 = await _run("xclip", "-selection", "primary", "-o", timeout=5.0)
            if code2 == 0 and out2:
                text_out = out2
            else:
                err_msg = err_msg or err2.strip()[:300]
    elif shutil.which("xsel"):
        code, out, err = await _run("xsel", "--clipboard", "--output", timeout=5.0)
        if code == 0:
            text_out = out
        else:
            err_msg = err.strip()[:300]
    else:
        raise HTTPException(status_code=500, detail="Neither xclip nor xsel available")
    return {"ok": True, "text": text_out, "error": err_msg or None}


async def _pipe_clipboard(cmd: list[str], payload: str, *, allow_hang: bool = False) -> None:
    """Write text to an xclip/xsel process. xclip may keep running as clipboard owner."""
    env = _env()
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    data = payload.encode("utf-8", errors="replace")
    try:
        _, err_b = await asyncio.wait_for(proc.communicate(data), timeout=2.0)
    except asyncio.TimeoutError:
        # xclip often stays alive to serve CLIPBOARD; stdin is already consumed.
        if allow_hang:
            return
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        try:
            await proc.wait()
        except ProcessLookupError:
            pass
        raise HTTPException(status_code=504, detail=f"{cmd[0]} timed out")
    if (proc.returncode or 0) != 0:
        # Some xclip builds exit non-zero even after a successful handoff; verify below.
        err = (err_b or b"").decode("utf-8", errors="replace")[:300]
        if not allow_hang:
            raise HTTPException(status_code=500, detail=f"{cmd[0]} failed: {err}")


@app.post("/clipboard")
async def set_clipboard(body: ClipboardBody) -> dict[str, Any]:
    """Write text to the X11 clipboard (CLIPBOARD + PRIMARY)."""
    payload = body.text if body.text is not None else ""
    # Prefer xsel — it exits after setting. Fall back to xclip (may hang as owner).
    if shutil.which("xsel"):
        await _pipe_clipboard(["xsel", "--clipboard", "--input"], payload, allow_hang=False)
        try:
            await _pipe_clipboard(["xsel", "--primary", "--input"], payload, allow_hang=False)
        except HTTPException:
            pass
    elif shutil.which("xclip"):
        await _pipe_clipboard(
            ["xclip", "-selection", "clipboard", "-i"], payload, allow_hang=True
        )
        try:
            await _pipe_clipboard(
                ["xclip", "-selection", "primary", "-i"], payload, allow_hang=True
            )
        except HTTPException:
            pass
    else:
        raise HTTPException(status_code=500, detail="Neither xclip nor xsel available")
    return {"ok": True, "text_len": len(payload)}
