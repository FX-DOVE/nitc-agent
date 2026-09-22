"""Computer-use tools — drive the shared interactive desktop via desktop-api."""

from __future__ import annotations

import json
from typing import Any

import httpx

from app.config import get_settings

_TIMEOUT = httpx.Timeout(60.0, connect=10.0)


def _base_url() -> str:
    return get_settings().desktop_api_url.rstrip("/")


async def _post(path: str, payload: dict[str, Any] | None = None) -> str:
    url = f"{_base_url()}{path}"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(url, json=payload or {})
            text = resp.text
            if resp.status_code >= 400:
                return json.dumps(
                    {
                        "ok": False,
                        "error": f"desktop-api HTTP {resp.status_code}",
                        "detail": text[:1500],
                        "url": url,
                    }
                )
            try:
                return json.dumps(resp.json())
            except Exception:  # noqa: BLE001
                return json.dumps({"ok": True, "raw": text[:2000]})
    except httpx.ConnectError as exc:
        return json.dumps(
            {
                "ok": False,
                "error": (
                    "Cannot reach desktop-api. Is the `desktop` Compose service running? "
                    f"DESKTOP_API_URL={_base_url()}"
                ),
                "detail": str(exc),
            }
        )
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


async def desktop_screenshot(filename: str | None = None) -> str:
    """Capture the interactive desktop display; save under workspace/screenshots/."""
    url = f"{_base_url()}/screenshot"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(url, params={"filename": filename} if filename else None)
            if resp.status_code >= 400:
                return json.dumps(
                    {
                        "ok": False,
                        "error": f"desktop-api HTTP {resp.status_code}",
                        "detail": resp.text[:1500],
                    }
                )
            return json.dumps(resp.json())
    except httpx.ConnectError as exc:
        return json.dumps(
            {
                "ok": False,
                "error": (
                    "Cannot reach desktop-api. Is the `desktop` Compose service running? "
                    f"DESKTOP_API_URL={_base_url()}"
                ),
                "detail": str(exc),
            }
        )
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


async def desktop_click(x: int, y: int, button: int = 1, clicks: int = 1) -> str:
    return await _post(
        "/click",
        {"x": int(x), "y": int(y), "button": int(button), "clicks": int(clicks)},
    )


async def desktop_type(text: str = "", key: str | None = None) -> str:
    payload: dict[str, Any] = {"text": text or ""}
    if key:
        payload["key"] = key
    return await _post("/type", payload)


async def desktop_hotkey(keys: list[str]) -> str:
    if isinstance(keys, str):
        # Allow "ctrl+c" style from the model
        keys = [k for k in keys.replace("+", " ").split() if k]
    return await _post("/hotkey", {"keys": list(keys)})


async def desktop_scroll(
    x: int,
    y: int,
    direction: str = "down",
    amount: int = 3,
) -> str:
    return await _post(
        "/scroll",
        {
            "x": int(x),
            "y": int(y),
            "direction": direction,
            "amount": int(amount),
        },
    )


async def desktop_open_browser(url: str) -> str:
    return await _post("/open_browser", {"url": url})
