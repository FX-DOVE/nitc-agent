"""In-process approval gate for Auto-review: pause tool calls until Approve/Decline."""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import threading
import time
from pathlib import Path
from typing import Any, Callable

from app.config import get_settings

logger = logging.getLogger("nitc.approvals")
_lock = threading.RLock()
_approvals: dict[str, dict[str, Any]] = {}
_events: dict[str, asyncio.Event] = {}


def _store_path() -> Path:
    root = get_settings().workspace_path / "approvals"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _persist(rec: dict[str, Any]) -> None:
    try:
        path = _store_path() / f"{rec['id']}.json"
        path.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except OSError as exc:
        logger.warning("approval persist failed: %s", exc)


def create_approval(
    *,
    tool: str,
    args: dict[str, Any],
    job_id: str | None = None,
    risk: str = "medium",
    args_summary: str | None = None,
    rules: list[str] | None = None,
) -> dict[str, Any]:
    """Register a pending approval and return the public record."""
    approval_id = secrets.token_urlsafe(10)
    summary = args_summary or _summarize_args(tool, args)
    rec: dict[str, Any] = {
        "id": approval_id,
        "approval_id": approval_id,
        "type": "approval",
        "status": "pending",
        "tool": tool,
        "args": args,
        "args_summary": summary,
        "risk": risk,
        "rules": rules or [],
        "job_id": job_id,
        "decision": None,
        "created_at": time.time(),
        "resolved_at": None,
    }
    with _lock:
        _approvals[approval_id] = rec
        try:
            loop = asyncio.get_running_loop()
            _events[approval_id] = asyncio.Event()
        except RuntimeError:
            pass
        _persist(rec)
    return dict(rec)


def get_approval(approval_id: str) -> dict[str, Any] | None:
    with _lock:
        rec = _approvals.get(approval_id)
        if rec:
            return dict(rec)
    path = _store_path() / f"{approval_id}.json"
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                with _lock:
                    _approvals[approval_id] = data
                return dict(data)
        except (OSError, json.JSONDecodeError):
            return None
    return None


def resolve_approval(approval_id: str, decision: str) -> dict[str, Any] | None:
    """decision: approve | decline | allow_once"""
    decision = (decision or "").strip().lower()
    if decision not in ("approve", "decline", "allow_once"):
        raise ValueError("decision must be approve, decline, or allow_once")
    with _lock:
        rec = _approvals.get(approval_id)
        if not rec:
            path = _store_path() / f"{approval_id}.json"
            if path.is_file():
                try:
                    rec = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    return None
            if not rec:
                return None
            _approvals[approval_id] = rec
        if rec.get("status") != "pending":
            return dict(rec)
        rec["status"] = "approved" if decision in ("approve", "allow_once") else "declined"
        rec["decision"] = decision
        rec["resolved_at"] = time.time()
        _persist(rec)
        ev = _events.get(approval_id)
    if ev is not None:
        ev.set()
    # allow_once also grants runtime allow token for tool retry edge cases
    if decision in ("approve", "allow_once"):
        try:
            from app.runtime_settings import grant_allow_once

            grant_allow_once(str(rec.get("tool") or ""), rec.get("args") or {}, ttl_sec=180)
        except Exception:  # noqa: BLE001
            pass
    return dict(rec)


async def wait_for_decision(approval_id: str, timeout: float = 600.0) -> str:
    """
    Wait until Approve/Decline. Returns decision string.
    On timeout, auto-decline so the job does not hang forever.
    """
    with _lock:
        ev = _events.get(approval_id)
        if ev is None:
            ev = asyncio.Event()
            _events[approval_id] = ev
            rec = _approvals.get(approval_id)
            if rec and rec.get("status") != "pending":
                return str(rec.get("decision") or "decline")
    try:
        await asyncio.wait_for(ev.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        resolve_approval(approval_id, "decline")
        return "decline"
    rec = get_approval(approval_id) or {}
    return str(rec.get("decision") or "decline")


def public_approval(rec: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "approval",
        "approval_id": rec.get("id") or rec.get("approval_id"),
        "status": rec.get("status"),
        "tool": rec.get("tool"),
        "args_summary": rec.get("args_summary"),
        "risk": rec.get("risk"),
        "rules": rec.get("rules") or [],
        "job_id": rec.get("job_id"),
        "decision": rec.get("decision"),
        # include args for Approve to re-grant; UI may truncate display
        "args": rec.get("args") or {},
    }


def _summarize_args(tool: str, args: dict[str, Any]) -> str:
    if tool == "shell":
        cmd = str(args.get("command") or "")
        return cmd if len(cmd) <= 240 else cmd[:237] + "..."
    if tool == "github_run":
        a = str(args.get("args") or "")
        return a if len(a) <= 240 else a[:237] + "..."
    if tool in ("browser_navigate", "desktop_open_browser"):
        return str(args.get("url") or "")[:240]
    if tool == "write_file":
        return f"path={args.get('path')}"
    try:
        raw = json.dumps(args, ensure_ascii=False)
    except TypeError:
        raw = str(args)
    return raw if len(raw) <= 240 else raw[:237] + "..."


# Contextvar-style job binding for the current agent turn
_job_bind: dict[int, str] = {}
_event_sink: dict[int, Callable[[dict[str, Any]], None]] = {}


def bind_job(job_id: str | None, sink: Callable[[dict[str, Any]], None] | None = None) -> int:
    """Bind current asyncio task to a job for approval events. Returns token."""
    token = id(asyncio.current_task()) if asyncio.current_task() else secrets.randbelow(1_000_000_000)
    if job_id:
        _job_bind[token] = job_id
    if sink:
        _event_sink[token] = sink
    return token


def unbind(token: int) -> None:
    _job_bind.pop(token, None)
    _event_sink.pop(token, None)


def current_job_id() -> str | None:
    task = asyncio.current_task()
    if not task:
        return None
    # Prefer exact task id; also scan values (workers may nest)
    tid = id(task)
    if tid in _job_bind:
        return _job_bind[tid]
    # Fallback: single binding
    if len(_job_bind) == 1:
        return next(iter(_job_bind.values()))
    return None


def emit_approval_event(rec: dict[str, Any]) -> None:
    task = asyncio.current_task()
    tid = id(task) if task else None
    sink = _event_sink.get(tid) if tid else None
    if sink is None and len(_event_sink) == 1:
        sink = next(iter(_event_sink.values()))
    if sink:
        try:
            sink(public_approval(rec))
        except Exception:  # noqa: BLE001
            logger.exception("approval event sink failed")
