"""Headless Chromium tools via Playwright."""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

from app.config import get_settings

_lock = asyncio.Lock()
_playwright: Any = None
_browser: Any = None
_page: Any = None


async def _ensure_page():
    global _playwright, _browser, _page
    async with _lock:
        if _page is not None and not _page.is_closed():
            return _page

        from playwright.async_api import async_playwright

        if _playwright is None:
            _playwright = await async_playwright().start()
        if _browser is None or not _browser.is_connected():
            _browser = await _playwright.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
        context = await _browser.new_context(
            viewport={"width": 1280, "height": 720},
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 NitcAgent/0.1"
            ),
        )
        _page = await context.new_page()
        return _page


async def browser_navigate(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        return json.dumps({"ok": False, "error": "URL must start with http:// or https://"})
    page = await _ensure_page()
    response = await page.goto(url, wait_until="domcontentloaded", timeout=45_000)
    status = response.status if response else None
    return json.dumps(
        {
            "ok": True,
            "url": page.url,
            "title": await page.title(),
            "status": status,
        }
    )


async def browser_get_text(url: str | None = None, max_chars: int = 15_000) -> str:
    page = await _ensure_page()
    if url:
        nav = json.loads(await browser_navigate(url))
        if not nav.get("ok"):
            return json.dumps(nav)
    text = await page.inner_text("body")
    text = " ".join(text.split())
    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars] + "…"
    return json.dumps(
        {
            "ok": True,
            "url": page.url,
            "title": await page.title(),
            "text": text,
            "truncated": truncated,
        }
    )


async def browser_screenshot(url: str | None = None, filename: str | None = None) -> str:
    page = await _ensure_page()
    if url:
        nav = json.loads(await browser_navigate(url))
        if not nav.get("ok"):
            return json.dumps(nav)

    workspace = get_settings().workspace_path
    shot_dir = workspace / "screenshots"
    shot_dir.mkdir(parents=True, exist_ok=True)

    name = filename or f"shot-{uuid.uuid4().hex[:10]}.png"
    if not name.endswith(".png"):
        name += ".png"
    # Keep filename workspace-safe (no path traversal)
    name = Path(name).name
    out = shot_dir / name
    await page.screenshot(path=str(out), full_page=False)
    rel = f"screenshots/{name}"
    media = f"/api/media/screenshots/{name}"
    return json.dumps(
        {
            "ok": True,
            "url": page.url,
            "page_url": page.url,
            "path": rel,
            "image_url": media,
            "absolute": str(out),
        }
    )
