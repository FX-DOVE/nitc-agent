"""Tool-calling agent loop over an OpenAI-compatible chat completions API."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from app.attachments import enrich_user_message, user_content_as_text
from app.config import get_settings
from app.tools import get_openai_tools, run_tool

logger = logging.getLogger("nitc.agent")

JOB_STATE_FILE = ".nitc_job.json"

SYSTEM_PROMPT = """You are Nitc Agent — a capable everyday colleague with a sandboxed Linux computer the user can watch live (Computer / noVNC). Aim for Grok Bot–level usefulness: proactive, concrete, and finish the job. Prefer acting over asking when the request is clear.

## Personality & communication
- Sharp, warm, practical teammate — not robotic, not a lecture.
- When the ask is clear (e.g. "write a calculator and zip it"), **just do it**: write files → verify → zip → return a markdown download link. Do not stop after narrating intent.
- Ask clarifying questions only when requirements are genuinely ambiguous (design briefs, video edits, multi-option product work).
- For image analysis: lead with a thorough visual description, then insights — never claim blindness when an attachment was provided.
- Write for readability: short paragraphs, bullets, **bold** for paths/labels. Markdown is rendered in chat.
- Be honest about limits. Never invent tool results.

## Attachments & vision
- Attachments live under workspace (`uploads/...`). Inspect them with vision input and/or tools.
- Text/code: `read_file` / `shell`. Audio may include a transcript.

## Tools
- `shell` — non-interactive commands in the sandbox workspace
- `read_file` / `write_file` / `list_dir` — workspace files (prefer relative paths)
- `zip_paths` — zip workspace files/folders into `downloads/` and get a `/api/media/files/...` URL
- Interactive desktop: `desktop_screenshot`, `desktop_click`, `desktop_type`, `desktop_hotkey`, `desktop_scroll`, `desktop_open_browser`
- Headless browser: `browser_navigate`, `browser_get_text`, `browser_screenshot`
- GitHub via `gh`: `github_run`

## Delivery rules (critical)
- Finish multi-step jobs end-to-end. Example: create calculator → smoke-test → `zip_paths` → reply with `[Download name.zip](/api/media/files/name.zip)` so mobile shows a tappable link.
- Prefer `write_file` + `zip_paths` over asking the user to copy code out of chat.
- Prefer `shell` zip only if `zip_paths` is unavailable; still return a `/api/media/files/...` or workspace path the UI can download.
- After screenshots, briefly describe what is visible; the UI embeds `/api/media/screenshots/...` automatically.
- Verify with tools; do not claim success without checking.

## Auto-review (when enabled)
- The runtime may pause truly risky actions and emit a real Approve/Decline card in the UI.
- **Never invent or tell the user to click Approve unless an approval card was actually emitted** (tool result will mention approval_id / waiting). If a tool was declined, acknowledge and adapt.
- Everyday coding (write/read/list files, zip, screenshots, benign desktop, normal shell) is NOT paused.

## GUI loop (when visual work is needed)
1. Screenshot → 2. Plan → 3. Act → 4. Verify → 5. Narrate briefly.
6. Login/2FA/CAPTCHA: stop and ask the user to finish in Computer; resume the same job when they say **done**.

## Job continuity
- Keep goal / last step / next step in mind. Honor resume context when prepended.
"""


_SCREENSHOT_TOOLS = frozenset({"desktop_screenshot", "browser_screenshot"})
_RESUME_TOKENS = frozenset({"done", "continue"})


def _job_state_path() -> Path:
    return get_settings().workspace_path / JOB_STATE_FILE


def load_job_state() -> dict[str, Any] | None:
    path = _job_state_path()
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def save_job_state(state: dict[str, Any]) -> None:
    path = _job_state_path()
    try:
        payload = {
            **state,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning("could not save job state: %s", exc)


def clear_job_state() -> None:
    path = _job_state_path()
    try:
        if path.is_file():
            path.unlink()
    except OSError:
        pass


def _maybe_inject_resume(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """If the latest user message is exactly 'done' or 'continue', prepend job resume context."""
    if not messages:
        return messages
    last = messages[-1]
    if last.get("role") != "user":
        return messages
    raw = user_content_as_text(last.get("content")).strip()
    if raw.lower() not in _RESUME_TOKENS:
        return messages

    state = load_job_state()
    if not state:
        return messages

    summary = state.get("summary") or state.get("goal") or "(no summary)"
    last_step = state.get("last_step") or ""
    next_step = state.get("next_step") or ""
    blocker = state.get("blocker") or ""
    parts = [
        "[Resume context — user said they are ready to continue]",
        f"Goal: {summary}",
    ]
    if last_step:
        parts.append(f"Last step: {last_step}")
    if next_step:
        parts.append(f"Next step: {next_step}")
    if blocker:
        parts.append(f"Was blocked on: {blocker}")
    parts.append(
        "Continue this same job from where you left off. Screenshot the desktop if needed, then proceed."
    )
    resume_block = "\n".join(parts)
    out = list(messages)
    out[-1] = {
        "role": "user",
        "content": f"{resume_block}\n\nUser message: {raw}",
    }
    return out


def _extract_job_state_from_reply(reply: str) -> None:
    """Optionally persist a lightweight job_state fenced block the model may emit."""
    # Accept optional ```job_state ... ``` JSON for continuity; strip is handled by caller if needed.
    m = re.search(r"```job_state\s*(\{.*?\})\s*```", reply, re.DOTALL | re.IGNORECASE)
    if not m:
        return
    try:
        data = json.loads(m.group(1))
        if isinstance(data, dict) and (data.get("goal") or data.get("summary")):
            save_job_state(
                {
                    "summary": data.get("summary") or data.get("goal") or "",
                    "goal": data.get("goal") or data.get("summary") or "",
                    "last_step": data.get("last_step") or "",
                    "next_step": data.get("next_step") or "",
                    "blocker": data.get("blocker") or "",
                }
            )
    except (json.JSONDecodeError, TypeError):
        pass


def _strip_job_state_fence(reply: str) -> str:
    return re.sub(
        r"\n*```job_state\s*\{.*?\}\s*```\n*",
        "\n",
        reply,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()


def _heuristic_save_job_on_user_wait(reply: str, user_text: str) -> None:
    """If the assistant asks the user to complete login/2FA, snapshot a simple job state."""
    lower = (reply or "").lower()
    wait_markers = (
        "log in",
        "login",
        "sign in",
        "2fa",
        "two-factor",
        "captcha",
        "when you're done",
        "when you are done",
        "say done",
        "reply done",
        "computer view",
        "complete it in the computer",
    )
    if not any(m in lower for m in wait_markers):
        return
    # Keep prior state if richer; otherwise seed from recent user ask.
    prev = load_job_state() or {}
    save_job_state(
        {
            "summary": prev.get("summary")
            or prev.get("goal")
            or (user_text[:240] if user_text else "Continue desktop task"),
            "goal": prev.get("goal") or (user_text[:240] if user_text else ""),
            "last_step": prev.get("last_step") or "Paused for user action",
            "next_step": prev.get("next_step") or "Resume after user confirms done",
            "blocker": "Waiting for user (login / 2FA / confirmation)",
        }
    )


def _image_urls_from_tool_result(name: str, result: str) -> list[str]:
    """Collect media URLs from screenshot tool JSON results."""
    if name not in _SCREENSHOT_TOOLS:
        return []
    try:
        data = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(data, dict) or not data.get("ok", True):
        if data.get("ok") is False:
            return []

    urls: list[str] = []
    for key in ("url", "image_url", "media_url"):
        val = data.get(key)
        if isinstance(val, str) and val.startswith("/api/media/screenshots/"):
            urls.append(val)

    path = data.get("path")
    if isinstance(path, str) and path:
        fname = Path(path).name
        if fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".gif")):
            media = f"/api/media/screenshots/{fname}"
            if media not in urls:
                urls.append(media)
    return urls



def _download_urls_from_tool_result(name: str, result: str) -> list[str]:
    """Collect downloadable media URLs from zip/file tool results."""
    if name not in ("zip_paths", "shell"):
        return []
    urls: list[str] = []
    try:
        data = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        data = None
    if isinstance(data, dict):
        for key in ("url", "media_url", "download_url"):
            val = data.get(key)
            if isinstance(val, str) and val.startswith("/api/media/"):
                urls.append(val)
        md = data.get("download_markdown")
        if isinstance(md, str):
            for m in re.findall(r"\(/api/media/[^)]+\)", md):
                urls.append(m[1:-1])
    # also scrape raw
    for m in re.findall(r"/api/media/files/[A-Za-z0-9._-]+", result or ""):
        if m not in urls:
            urls.append(m)
    return urls


def _ensure_reply_mentions_downloads(reply: str, downloads: list[str]) -> str:
    if not downloads:
        return reply
    out = (reply or "").rstrip()
    for url in downloads:
        name = url.rsplit("/", 1)[-1]
        md = f"[Download {name}]({url})"
        if url in out:
            continue
        out = f"{out}\n\n{md}" if out else md
    return out


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



def _flatten_messages_for_text_only(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert multimodal user content parts to plain text (fallback when vision unsupported)."""
    out: list[dict[str, Any]] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            text = user_content_as_text(content)
            # note dropped images
            n_img = sum(1 for p in content if isinstance(p, dict) and p.get("type") == "image_url")
            if n_img:
                text = (text + f"\n\n({n_img} image attachment(s) were provided; model has no vision — use workspace paths above).").strip()
            nm = dict(m)
            nm["content"] = text
            out.append(nm)
        else:
            out.append(m)
    return out



_TOOL_STATUS = {
    "shell": "Using shell…",
    "run_shell": "Using shell…",
    "write_file": "Writing file…",
    "read_file": "Reading file…",
    "list_dir": "Listing directory…",
    "browser": "Using browser…",
    "browser_navigate": "Opening page…",
    "browser_click": "Clicking in browser…",
    "browser_type": "Typing in browser…",
    "desktop_screenshot": "Capturing desktop…",
    "desktop_click": "Clicking on desktop…",
    "desktop_type": "Typing on desktop…",
    "computer": "Using computer…",
    "github": "Using GitHub…",
    "zip_paths": "Zipping files…",
}


def _tool_status_message(name: str) -> str:
    n = (name or "").strip()
    if n in _TOOL_STATUS:
        return _TOOL_STATUS[n]
    if n.startswith("browser"):
        return "Using browser…"
    if n.startswith("desktop") or n.startswith("computer"):
        return "Using computer…"
    if n.startswith("github"):
        return "Using GitHub…"
    nice = n.replace("_", " ").strip() or "tool"
    return f"Using {nice}…"


def _emit(on_event: Any, payload: dict[str, Any]) -> None:
    if not on_event:
        return
    try:
        on_event(payload)
    except Exception:
        pass


async def chat(
    messages: list[dict[str, Any]],
    attachments: list[dict[str, Any]] | None = None,
    *,
    instructions: str | None = None,
    on_event: Any = None,
) -> dict[str, Any]:
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

    # Enrich the latest user turn with attachment context (text excerpts / vision / audio notes).
    if attachments and messages and messages[-1].get("role") == "user":
        last = messages[-1]
        base_text = user_content_as_text(last.get("content"))
        enriched = await enrich_user_message(base_text, attachments)
        messages = list(messages)
        messages[-1] = enriched

    messages = _maybe_inject_resume(messages)
    last_user = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user = user_content_as_text(m.get("content"))
            break

    system_content = SYSTEM_PROMPT
    if instructions and str(instructions).strip():
        system_content = (
            SYSTEM_PROMPT
            + "\n\n## Bot-specific instructions\n"
            + str(instructions).strip()[:4000]
            + "\n"
        )
    try:
        from app.runtime_settings import auto_review_enabled, auto_review_rules
        if auto_review_enabled():
            rules = auto_review_rules()
            extra = (
                "\n\n## Auto-review (ENABLED)\n"
                "Only destructive/exfil-like actions are paused. Everyday write_file, read_file, "
                "list_dir, zip_paths, screenshots, and benign desktop/shell are free.\n"
                "When a tool is paused, the UI shows a real Approve/Decline card via a structured "
                "approval event — do NOT invent Approve button text yourself. Wait for the tool "
                "result (approved execution or declined). Never claim buttons were sent if they were not.\n"
            )
            if rules:
                extra += "User auto-review rules:\n" + "\n".join(f"- {r}" for r in rules) + "\n"
            system_content = system_content + extra
    except Exception:
        pass
    working: list[dict[str, Any]] = [{"role": "system", "content": system_content}]
    for m in messages:
        role = m.get("role")
        if role in ("user", "assistant", "tool", "system"):
            working.append(m)

    tools = get_openai_tools()
    tool_rounds = 0
    collected_images: list[str] = []
    collected_downloads: list[str] = []
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
            _emit(
                on_event,
                {
                    "type": "status",
                    "message": "Thinking…" if tool_rounds == 0 else f"Planning next step (round {tool_rounds})…",
                    "partial_reply": "Thinking…" if tool_rounds == 0 else f"Working… (round {tool_rounds})",
                },
            )
            payload: dict[str, Any] = {
                "model": settings.model,
                "messages": working,
                "tools": tools,
                "tool_choice": "auto",
            }
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code >= 400:
                err_text = resp.text[:2000]
                has_multi = any(isinstance(m.get("content"), list) for m in working)
                lower_err = err_text.lower()
                vision_hint = any(
                    k in lower_err
                    for k in (
                        "image", "vision", "multimodal", "content", "invalid",
                        "unsupported", "media", "base64", "dataurl", "data_url",
                    )
                )
                # Prefer text-only retry whenever multimodal was sent and the API rejected —
                # attachment paths remain in the text so the model can still use tools.
                if has_multi and (vision_hint or resp.status_code in (400, 422)):
                    logger.info(
                        "multimodal rejected by API (%s); retrying text-only with file paths",
                        resp.status_code,
                    )
                    working[:] = _flatten_messages_for_text_only(working)
                    # Nudge the model to inspect attachments via tools rather than claiming blindness
                    working.append(
                        {
                            "role": "system",
                            "content": (
                                "Vision input was unavailable for this turn. "
                                "Attachment workspace paths are listed in the user message. "
                                "Use tools (read_file/shell/desktop_screenshot as appropriate) "
                                "to inspect what you can; describe files honestly. "
                                "Do not claim you never received an attachment."
                            ),
                        }
                    )
                    payload = {
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
                else:
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
                reply = _ensure_reply_mentions_downloads(reply, collected_downloads)
                _extract_job_state_from_reply(reply)
                reply = _strip_job_state_fence(reply)
                _heuristic_save_job_on_user_wait(reply, last_user)
                _emit(
                    on_event,
                    {
                        "type": "partial",
                        "partial_reply": reply,
                        "message": "Finalizing…",
                    },
                )
                return {
                    "reply": reply,
                    "messages": [m for m in working if m.get("role") != "system"],
                    "tool_rounds": tool_rounds,
                    "images": collected_images,
                    "downloads": collected_downloads,
                }

            tool_rounds += 1
            assistant_text = (msg.get("content") or "").strip()
            if assistant_text:
                _emit(
                    on_event,
                    {
                        "type": "partial",
                        "partial_reply": assistant_text,
                        "message": assistant_text[:200],
                    },
                )
            for tc in tool_calls:
                fn = tc.get("function") or {}
                name = fn.get("name") or ""
                raw_args = fn.get("arguments") or "{}"
                tc_id = tc.get("id") or f"call_{tool_rounds}"
                logger.info("tool_call name=%s args=%s", name, raw_args[:200])
                status_msg = _tool_status_message(name)
                preview = assistant_text or status_msg
                if assistant_text and status_msg:
                    preview = f"{assistant_text}\n\n_{status_msg}_"
                elif status_msg:
                    preview = status_msg
                _emit(
                    on_event,
                    {
                        "type": "tool_start",
                        "tool": name,
                        "message": status_msg,
                        "partial_reply": preview,
                    },
                )
                result = await run_tool(name, raw_args)
                for img in _image_urls_from_tool_result(name, result):
                    if img not in collected_images:
                        collected_images.append(img)
                for dl in _download_urls_from_tool_result(name, result):
                    if dl not in collected_downloads:
                        collected_downloads.append(dl)
                done_msg = f"Finished {name.replace('_', ' ') or 'tool'}…"
                done_preview = assistant_text or done_msg
                if assistant_text:
                    done_preview = f"{assistant_text}\n\n_{done_msg}_"
                _emit(
                    on_event,
                    {
                        "type": "tool_result",
                        "tool": name,
                        "ok": True,
                        "message": done_msg,
                        "partial_reply": done_preview,
                    },
                )
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
        reply = _ensure_reply_mentions_downloads(reply, collected_downloads)
        _extract_job_state_from_reply(reply)
        reply = _strip_job_state_fence(reply)
        _heuristic_save_job_on_user_wait(reply, last_user)
        return {
            "reply": reply,
            "messages": [m for m in working if m.get("role") != "system"],
            "tool_rounds": tool_rounds,
            "images": collected_images,
            "downloads": collected_downloads,
        }
