"""Tool-calling agent loop over an OpenAI-compatible chat completions API."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx

from app.config import get_settings
from app.tools import get_openai_tools, run_tool

logger = logging.getLogger("nitc.agent")

SYSTEM_PROMPT = """You are Nitc Agent, a capable self-hosted AI assistant with a sandboxed computer, an interactive Linux desktop the user can watch, a headless browser, and GitHub tools.

You can:
- Run shell commands in your sandbox workspace (`shell`)
- Read, write, and list files under the workspace
- Drive the **interactive desktop** (user-visible via noVNC): `desktop_screenshot`, `desktop_click`, `desktop_type`, `desktop_hotkey`, `desktop_scroll`, `desktop_open_browser`
- Browse quickly with headless Playwright: `browser_navigate`, `browser_get_text`, `browser_screenshot`
- Interact with GitHub via the `gh` CLI (`github_run`)

Guidelines:
- Be concise and practical.
- Prefer **desktop computer-use** (`desktop_*`) for GUI tasks, visual verification, filling forms the user should see, or anything interactive.
- Prefer **Playwright** (`browser_*`) for quick headless scrapes, fetching page text, or when you only need content — not a live desktop session.
- Prefer workspace-relative paths; never try to escape the sandbox.
- After using tools, summarize results clearly for the user.
- If a tool fails, explain briefly and try an alternative when reasonable.

Human-like computer use (Phase 3 groundwork):
- For GUI work: **screenshot first** (`desktop_screenshot`) to see the screen, then act (`click`/`type`/`hotkey`/`scroll`/`open_browser`), then **screenshot again to verify**.
- Narrate briefly what you see and what you will do next — like a careful human operator.
- If login, 2FA, CAPTCHA, payment, or other user-only steps appear: **stop and ask the user** to complete them in the Computer view, then continue after they confirm.
- Screenshots are saved under workspace/screenshots/ and **shown inline in chat** via `/api/media/screenshots/...`. After a successful screenshot, briefly mention what is visible; the UI also embeds the image automatically.
"""

_SCREENSHOT_TOOLS = frozenset({"desktop_screenshot", "browser_screenshot"})


def _image_urls_from_tool_result(name: str, result: str) -> list[str]:
    """Collect media URLs from screenshot tool JSON results."""
    if name not in _SCREENSHOT_TOOLS:
        return []
    try:
        data = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(data, dict) or not data.get("ok", True):
        # desktop/browser return ok:false on failure; skip
        if data.get("ok") is False:
            return []

    urls: list[str] = []
    for key in ("url", "image_url", "media_url"):
        val = data.get(key)
        if isinstance(val, str) and val.startswith("/api/media/screenshots/"):
            urls.append(val)

    path = data.get("path")
    if isinstance(path, str) and path:
        # path like screenshots/foo.png or absolute ending in screenshots/foo.png
        fname = Path(path).name
        if fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".gif")):
            media = f"/api/media/screenshots/{fname}"
            if media not in urls:
                urls.append(media)
    return urls


def _ensure_reply_mentions_images(reply: str, images: list[str]) -> str:
    """Append markdown image links if the model omitted them."""
    if not images:
        return reply
    out = (reply or "").rstrip()
    for img in images:
        md = f"![desktop]({img})"
        if img in out or md in out:
            continue
        out = f"{out}\n\n{md}" if out else md
    return out


async def chat(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Run the agent loop.

    Returns:
        {
          "reply": str,
          "messages": [...full conversation including tool turns...],
          "tool_rounds": int,
          "images": ["/api/media/screenshots/..."],
        }
    """
    settings = get_settings()
    if not settings.openai_api_key:
        return {
            "reply": "Configuration error: OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.",
            "messages": messages,
            "tool_rounds": 0,
            "images": [],
        }

    working: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in messages:
        role = m.get("role")
        if role in ("user", "assistant", "tool", "system"):
            working.append(m)

    tools = get_openai_tools()
    tool_rounds = 0
    collected_images: list[str] = []
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }
    if "openrouter.ai" in settings.openai_base_url:
        headers["HTTP-Referer"] = "https://github.com/FX-DOVE/nitc-agent"
        headers["X-Title"] = "Nitc Agent"

    url = settings.openai_base_url.rstrip("/") + "/chat/completions"

    async with httpx.AsyncClient(timeout=120.0) as client:
        while tool_rounds < settings.max_tool_rounds:
            payload: dict[str, Any] = {
                "model": settings.model,
                "messages": working,
                "tools": tools,
                "tool_choice": "auto",
            }
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code >= 400:
                err_text = resp.text[:2000]
                return {
                    "reply": f"LLM API error {resp.status_code}: {err_text}",
                    "messages": messages,
                    "tool_rounds": tool_rounds,
                    "images": collected_images,
                }

            data = resp.json()
            try:
                choice = data["choices"][0]
                msg = choice["message"]
            except (KeyError, IndexError, TypeError):
                return {
                    "reply": f"Unexpected API response: {json.dumps(data)[:1500]}",
                    "messages": messages,
                    "tool_rounds": tool_rounds,
                    "images": collected_images,
                }

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": msg.get("content") or "",
            }
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            working.append(assistant_msg)

            if not tool_calls:
                reply = (msg.get("content") or "").strip() or "(empty response)"
                reply = _ensure_reply_mentions_images(reply, collected_images)
                return {
                    "reply": reply,
                    "messages": [m for m in working if m.get("role") != "system"],
                    "tool_rounds": tool_rounds,
                    "images": collected_images,
                }

            tool_rounds += 1
            for tc in tool_calls:
                fn = tc.get("function") or {}
                name = fn.get("name") or ""
                raw_args = fn.get("arguments") or "{}"
                tc_id = tc.get("id") or f"call_{tool_rounds}"
                logger.info("tool_call name=%s args=%s", name, raw_args[:200])
                result = await run_tool(name, raw_args)
                for img in _image_urls_from_tool_result(name, result):
                    if img not in collected_images:
                        collected_images.append(img)
                working.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": result,
                    }
                )

        payload = {
            "model": settings.model,
            "messages": working
            + [
                {
                    "role": "user",
                    "content": "Tool round limit reached. Give your best final answer now without more tools.",
                }
            ],
        }
        resp = await client.post(url, headers=headers, json=payload)
        if resp.status_code >= 400:
            return {
                "reply": f"Reached tool limit ({settings.max_tool_rounds}) and final call failed: {resp.text[:1000]}",
                "messages": [m for m in working if m.get("role") != "system"],
                "tool_rounds": tool_rounds,
                "images": collected_images,
            }
        data = resp.json()
        reply = (
            data.get("choices", [{}])[0].get("message", {}).get("content") or ""
        ).strip() or "Stopped after maximum tool rounds."
        reply = _ensure_reply_mentions_images(reply, collected_images)
        return {
            "reply": reply,
            "messages": [m for m in working if m.get("role") != "system"],
            "tool_rounds": tool_rounds,
            "images": collected_images,
        }
