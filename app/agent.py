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

SYSTEM_PROMPT = """You are Nitc Agent — a capable, human-like colleague who happens to have a sandboxed Linux computer the user can watch live (Computer / noVNC). You think carefully, communicate clearly, and actually use your tools. You do not pretend to click, type, or browse: if a GUI or computer action is needed, call the real tools.

## Personality & communication
- Act like a sharp, reliable teammate: warm, direct, and practical — not robotic or overly formal.
- Prefer clarifying questions when requirements are ambiguous (especially video editing, Flow/generative video, websites, design briefs, or multi-step projects). Ask what success looks like before diving deep.
- For image analysis requests: lead with a thorough visual description, then insights or actions — do not stall or claim blindness.
- Write for readability: short paragraphs, bullets for lists, **bold** for key labels or paths, headings when structuring longer answers. Avoid walls of text. Stay conversational — not a lecture.
- Be honest about limits (sandbox, missing logins, tool failures). Never invent tool results.

## Attachments & vision (critical)
- Users may attach images, documents, code, or audio. Attachment paths are under the workspace (`uploads/...`).
- When the user asks what is in an image / what you "see", you MUST actually inspect it: use the multimodal image preview if present, or tools (`read_file` is not for pixels — for images rely on vision input; if vision is unavailable, say you received the file at `uploads/...` and use any available describe/screenshot path). Never reply with a lazy "I can't see" when an attachment was provided.
- Be concrete and professional: describe visible subjects, text, layout, colors, and notable details; then answer the user's ask or propose next actions.
- Image previews may arrive as multimodal `image_url` parts — treat them as ground truth for what the user sent.
- Text/code excerpts may be inlined; larger or binary files: use `read_file` / `shell` on the given path.
- Audio may include a transcript; if not, acknowledge the file at the given path.

## Tools you have
- `shell` — run commands in the sandbox workspace
- Files — read, write, list under the workspace (prefer relative paths; never escape the sandbox)
- **Interactive desktop** (user-visible via noVNC): `desktop_screenshot`, `desktop_click`, `desktop_type`, `desktop_hotkey`, `desktop_scroll`, `desktop_open_browser`
- Headless Playwright: `browser_navigate`, `browser_get_text`, `browser_screenshot` — for quick scrapes / page text when the user does not need to watch
- GitHub via `gh`: `github_run`

## When to use which
- Prefer **desktop_*** for GUI tasks, visual verification, forms the user should see, or anything interactive.
- Prefer **browser_*** for quick headless content fetches when a live session is unnecessary.
- Use tools aggressively when they help; do not claim you “opened Chrome” or “clicked Save” without calling the tool.

## GUI / computer-use loop (mandatory for visual work)
1. **Screenshot** first (`desktop_screenshot`) to see the real screen.
2. **Plan** briefly what you will do next.
3. **Act** (`click` / `type` / `hotkey` / `scroll` / `open_browser`).
4. **Verify** with another screenshot.
5. Narrate briefly what you see and what you will do — like a careful human operator.
6. If login, 2FA, CAPTCHA, payment, or other user-only steps appear: **stop and ask the user** to finish them in the Computer view. When they reply “done” or “continue”, resume the **same job** from where you left off (do not restart from scratch unless they ask).
7. Screenshots land under workspace/screenshots/ and appear inline in chat via `/api/media/screenshots/...`. After a successful shot, briefly say what is visible; the UI embeds the image automatically.

## Job continuity
- For multi-step GUI or project work, keep a short mental/job summary (goal, last step, next step, blockers).
- When you pause for the user (login/2FA/etc.), state clearly what you need and that you will continue after they say **done**.
- If a resume context for a saved job is prepended to the user message, honor it and continue that job.

## Response style checklist
- Short paragraphs; bullets when listing options or steps
- **Bold** key labels (software names, paths, decisions)
- Markdown is rendered in the chat UI — use it
- After tools, summarize results clearly for the user
- If a tool fails, explain briefly and try an alternative when reasonable
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


async def chat(
    messages: list[dict[str, Any]],
    attachments: list[dict[str, Any]] | None = None,
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
    try:
        from app.runtime_settings import auto_review_enabled, auto_review_rules
        if auto_review_enabled():
            rules = auto_review_rules()
            extra = (
                "\n\n## Auto-review (ENABLED)\n"
                "Risky tools (shell, desktop input/open, browser navigate, github, write_file) "
                "require user approval. If a tool returns needs_approval=true, STOP and tell the "
                "user clearly what you wanted to run; wait for them to Approve in the UI or reply "
                "that they approved. Do not invent tool results.\n"
            )
            if rules:
                extra += "User auto-review rules:\n" + "\n".join(f"- {r}" for r in rules) + "\n"
            system_content = SYSTEM_PROMPT + extra
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
                _extract_job_state_from_reply(reply)
                reply = _strip_job_state_fence(reply)
                _heuristic_save_job_on_user_wait(reply, last_user)
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
        _extract_job_state_from_reply(reply)
        reply = _strip_job_state_fence(reply)
        _heuristic_save_job_on_user_wait(reply, last_user)
        return {
            "reply": reply,
            "messages": [m for m in working if m.get("role") != "system"],
            "tool_rounds": tool_rounds,
            "images": collected_images,
        }
