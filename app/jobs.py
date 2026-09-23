"""Durable async chat jobs: persist to disk, run agent in background independent of HTTP client."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.agent import chat as agent_chat
from app.attachments import resolve_upload
from app.config import get_settings
from app import store as durable_store

logger = logging.getLogger("nitc.jobs")

_SAFE_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_lock = threading.RLock()
_running: dict[str, asyncio.Task] = {}
_started = False
_workers_started = False
_job_queue: asyncio.Queue[str] | None = None
_queued_ids: set[str] = set()
_dispatcher_task: asyncio.Task | None = None


def jobs_root() -> Path:
    root = get_settings().workspace_path / "jobs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def sessions_root() -> Path:
    root = get_settings().workspace_path / "sessions"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_path(job_id: str) -> Path:
    if not _SAFE_ID.match(job_id):
        raise ValueError("Invalid job id")
    return jobs_root() / f"{job_id}.json"


def _session_path(session_id: str) -> Path:
    sid = (session_id or "").strip() or "default"
    if not _SAFE_ID.match(sid):
        # soften: sanitize
        sid = re.sub(r"[^A-Za-z0-9_.:-]+", "_", sid)[:128] or "default"
    return sessions_root() / f"{sid}.json"


def load_job(job_id: str) -> dict[str, Any] | None:
    try:
        path = _job_path(job_id)
    except ValueError:
        return None
    if not path.is_file():
        return None
    try:
        with _lock:
            data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def save_job(job: dict[str, Any]) -> None:
    job_id = job.get("id")
    if not job_id:
        raise ValueError("job missing id")
    path = _job_path(str(job_id))
    job = dict(job)
    job["updated_at"] = _now()
    tmp = path.with_suffix(".tmp")
    payload = json.dumps(job, ensure_ascii=False, indent=2)
    with _lock:
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(path)


def public_job(job: dict[str, Any]) -> dict[str, Any]:
    """Strip bulky internal fields for API responses."""
    pending = None
    for ev in reversed(list(job.get("events") or [])):
        if ev.get("type") == "approval" and ev.get("status") == "pending":
            pending = {
                "type": "approval",
                "approval_id": ev.get("approval_id"),
                "tool": ev.get("tool"),
                "args_summary": ev.get("args_summary"),
                "risk": ev.get("risk"),
                "args": ev.get("args") or {},
                "rules": ev.get("rules") or [],
            }
            break
        if ev.get("type") == "approval" and ev.get("status") in ("approved", "declined"):
            break
    return {
        "id": job.get("id"),
        "job_id": job.get("id"),
        "status": job.get("status"),
        "session_id": job.get("session_id"),
        "bot_id": job.get("bot_id") or job.get("session_id"),
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "error": job.get("error"),
        "reply": job.get("reply"),
        "tool_rounds": job.get("tool_rounds", 0),
        "images": job.get("images") or [],
        "downloads": job.get("downloads") or [],
        "events": job.get("events") or [],
        "user_message": job.get("user_message"),
        "attachments": job.get("attachments") or [],
        "partial_reply": job.get("partial_reply") or "",
        "pending_approval": pending,
        "instructions": job.get("instructions") or "",
    }


def append_event(job: dict[str, Any], kind: str, **extra: Any) -> None:
    events = list(job.get("events") or [])
    events.append({"ts": _now(), "type": kind, **extra})
    # Cap event log size
    if len(events) > 200:
        events = events[-200:]
    job["events"] = events


def create_job(
    *,
    messages: list[dict[str, Any]],
    attachments: list[dict[str, Any]] | None = None,
    session_id: str | None = None,
    bot_id: str | None = None,
    instructions: str | None = None,
) -> dict[str, Any]:
    job_id = uuid.uuid4().hex
    sid = (bot_id or session_id or "default").strip() or "default"
    # Validate attachments exist before queueing
    atts: list[dict[str, Any]] = []
    for a in attachments or []:
        if not isinstance(a, dict):
            continue
        if resolve_upload(a) is not None or a.get("path") or a.get("id"):
            atts.append({k: v for k, v in a.items() if v is not None})

    user_text = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content")
            user_text = c if isinstance(c, str) else str(c or "")
            break

    job: dict[str, Any] = {
        "id": job_id,
        "status": "queued",
        "session_id": sid,
        "bot_id": sid,
        "created_at": _now(),
        "updated_at": _now(),
        "started_at": None,
        "finished_at": None,
        "error": None,
        "reply": None,
        "partial_reply": "",
        "tool_rounds": 0,
        "images": [],
        "events": [],
        "user_message": user_text[:2000],
        "attachments": atts,
        "messages": messages,
        "result_messages": [],
        "instructions": (instructions or "").strip()[:4000],
        "downloads": [],
    }
    append_event(job, "queued", message="Job accepted")
    save_job(job)
    _mirror_session_user(sid, user_text, atts, job_id)
    return job


def _mirror_session_user(
    session_id: str,
    user_text: str,
    attachments: list[dict[str, Any]],
    job_id: str,
) -> None:
    path = _session_path(session_id)
    try:
        data: dict[str, Any]
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = {}
        else:
            data = {"session_id": session_id, "messages": [], "jobs": []}
        msgs = list(data.get("messages") or [])
        msgs.append(
            {
                "role": "user",
                "content": user_text,
                "attachments": attachments,
                "job_id": job_id,
                "ts": _now(),
            }
        )
        jobs = list(data.get("jobs") or [])
        if job_id not in jobs:
            jobs.append(job_id)
        data["messages"] = msgs[-200:]
        data["jobs"] = jobs[-100:]
        data["updated_at"] = _now()
        data["session_id"] = session_id
        tmp = path.with_suffix(".tmp")
        with _lock:
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(path)
        try:
            durable_store.append_message(
                session_id,
                role="user",
                content=user_text,
                attachments=attachments,
                job_id=job_id,
                created_at=data["messages"][-1].get("ts") if data.get("messages") else None,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("sqlite user mirror failed: %s", exc)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("session mirror failed: %s", exc)


def _mirror_session_assistant(
    session_id: str,
    reply: str,
    images: list[str],
    job_id: str,
    status: str,
) -> None:
    path = _session_path(session_id)
    try:
        data: dict[str, Any]
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = {}
        else:
            data = {"session_id": session_id, "messages": [], "jobs": []}
        msgs = list(data.get("messages") or [])
        msgs.append(
            {
                "role": "assistant",
                "content": reply,
                "images": images,
                "job_id": job_id,
                "status": status,
                "ts": _now(),
            }
        )
        data["messages"] = msgs[-200:]
        data["updated_at"] = _now()
        data["active_job_id"] = None if status in ("completed", "failed") else job_id
        tmp = path.with_suffix(".tmp")
        with _lock:
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(path)
        try:
            durable_store.append_message(
                session_id,
                role="assistant",
                content=reply,
                images=images,
                job_id=job_id,
                status=status,
                created_at=data["messages"][-1].get("ts") if data.get("messages") else None,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("sqlite assistant mirror failed: %s", exc)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("session mirror assistant failed: %s", exc)


def load_session(session_id: str) -> dict[str, Any] | None:
    # Prefer SQLite (durable); fall back to legacy JSON session files
    try:
        payload = durable_store.get_session_payload(session_id)
        if payload.get("messages"):
            return payload
    except Exception as exc:  # noqa: BLE001
        logger.warning("sqlite load_session failed: %s", exc)
        payload = None
    else:
        payload = payload  # may be empty messages
    path = _session_path(session_id)
    if not path.is_file():
        return payload
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else payload
    except (OSError, json.JSONDecodeError):
        return payload


def list_jobs(
    *,
    session_id: str | None = None,
    limit: int = 50,
    status: str | None = None,
) -> list[dict[str, Any]]:
    root = jobs_root()
    items: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        if path.name.endswith(".tmp"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        if session_id and (data.get("session_id") or data.get("bot_id")) != session_id:
            continue
        if status and data.get("status") != status:
            continue
        items.append(public_job(data))
        if len(items) >= limit:
            break
    return items


async def _run_job(job_id: str) -> None:
    job = load_job(job_id)
    if not job:
        return
    if job.get("status") in ("completed", "failed", "cancelled"):
        return

    # Track for pool_stats / UI; worker already owns exclusivity via queue
    try:
        task = asyncio.current_task()
        if task is not None:
            _running[job_id] = task
    except Exception:  # noqa: BLE001
        pass

    job["status"] = "running"
    job["started_at"] = _now()
    append_event(job, "running", message="Agent loop started")
    save_job(job)

    token = None
    try:
        from app import approvals as approval_store

        def _sink(payload: dict[str, Any]) -> None:
            cur = load_job(job_id)
            if not cur:
                return
            # Mark previous pending approvals resolved if a new one arrives with status
            append_event(
                cur,
                "approval",
                approval_id=payload.get("approval_id"),
                tool=payload.get("tool"),
                args_summary=payload.get("args_summary"),
                risk=payload.get("risk"),
                args=payload.get("args") or {},
                rules=payload.get("rules") or [],
                status=payload.get("status") or "pending",
                message="Waiting for Approve/Decline",
            )
            if (payload.get("status") or "pending") == "pending":
                cur["status"] = "awaiting_approval"
                cur["partial_reply"] = (
                    f"Auto-review: approval needed for `{payload.get('tool')}` — "
                    f"{payload.get('args_summary') or ''}"
                )[:500]
            save_job(cur)

        token = approval_store.bind_job(job_id, _sink)
        messages = list(job.get("messages") or [])
        attachments = list(job.get("attachments") or [])
        instructions = job.get("instructions") or ""

        def _on_event(ev: dict[str, Any]) -> None:
            try:
                cur = load_job(job_id)
                if not cur:
                    return
                et = str(ev.get("type") or "progress")
                if et == "tool_result" and cur.get("status") == "awaiting_approval":
                    cur["status"] = "running"
                msg = str(ev.get("message") or "")[:500]
                partial = ev.get("partial_reply")
                if partial is not None:
                    cur["partial_reply"] = str(partial)[:12000]
                elif msg:
                    prev = (cur.get("partial_reply") or "").strip()
                    if not prev or prev.startswith("Thinking") or prev.startswith("Working") or prev.startswith("Using ") or prev.startswith("Finished ") or prev.startswith("Auto-review"):
                        cur["partial_reply"] = msg
                    else:
                        base = prev.split("\n\n_")[0].rstrip()
                        cur["partial_reply"] = f"{base}\n\n_{msg}_"
                append_event(
                    cur,
                    et,
                    message=msg or None,
                    tool=ev.get("tool"),
                    ok=ev.get("ok"),
                )
                save_job(cur)
            except Exception:
                logger.exception("job %s on_event failed", job_id)

        result = await agent_chat(
            messages,
            attachments=attachments or None,
            instructions=instructions or None,
            on_event=_on_event,
        )
        reply = result.get("reply") or ""
        images = result.get("images") or []
        downloads = result.get("downloads") or []
        tool_rounds = int(result.get("tool_rounds") or 0)

        job = load_job(job_id) or job
        job["status"] = "completed"
        job["finished_at"] = _now()
        job["reply"] = reply
        job["partial_reply"] = reply
        job["images"] = images
        job["downloads"] = downloads
        job["tool_rounds"] = tool_rounds
        job["result_messages"] = result.get("messages") or []
        job["error"] = None
        append_event(job, "completed", message="Agent finished", tool_rounds=tool_rounds)
        save_job(job)
        _mirror_session_assistant(
            str(job.get("session_id") or "default"),
            reply,
            images,
            job_id,
            "completed",
        )
        logger.info("job %s completed rounds=%s", job_id, tool_rounds)
    except Exception as exc:  # noqa: BLE001
        logger.exception("job %s failed", job_id)
        job = load_job(job_id) or job
        job["status"] = "failed"
        job["finished_at"] = _now()
        job["error"] = str(exc)[:2000]
        job["reply"] = job.get("reply") or f"Job failed: {exc}"
        append_event(job, "failed", message=str(exc)[:500])
        save_job(job)
        _mirror_session_assistant(
            str(job.get("session_id") or "default"),
            job["reply"],
            job.get("images") or [],
            job_id,
            "failed",
        )
    finally:
        try:
            from app import approvals as approval_store
            if token is not None:
                approval_store.unbind(token)
        except Exception:
            pass
        _running.pop(job_id, None)


def job_concurrency() -> int:
    try:
        n = int(get_settings().nitc_job_concurrency)
    except Exception:  # noqa: BLE001
        n = 2
    return max(1, min(8, n))


async def _worker_loop(worker_id: int) -> None:
    assert _job_queue is not None
    logger.info("job worker %s started (concurrency pool)", worker_id)
    while True:
        job_id = await _job_queue.get()
        try:
            with _lock:
                _queued_ids.discard(job_id)
            await _run_job(job_id)
        except Exception:  # noqa: BLE001
            logger.exception("worker %s crashed on job %s", worker_id, job_id)
        finally:
            _job_queue.task_done()


async def ensure_worker_pool() -> None:
    """Start FIFO worker pool once per process."""
    global _workers_started, _job_queue
    if _workers_started and _job_queue is not None:
        return
    n = job_concurrency()
    _job_queue = asyncio.Queue()
    for i in range(n):
        asyncio.create_task(_worker_loop(i), name=f"nitc-worker-{i}")
    _workers_started = True
    logger.info("job worker pool ready: %s concurrent", n)


def enqueue_job(job_id: str) -> None:
    """Queue a job for the worker pool. Never cancels in-flight jobs."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.error("enqueue_job called with no running loop for %s", job_id)
        return

    async def _put() -> None:
        await ensure_worker_pool()
        assert _job_queue is not None
        with _lock:
            # Already running or already waiting in queue — do not duplicate
            existing = _running.get(job_id)
            if existing and not existing.done():
                return
            if job_id in _queued_ids:
                return
            _queued_ids.add(job_id)
        await _job_queue.put(job_id)

    loop.create_task(_put(), name=f"nitc-enqueue-{job_id}")


def pool_stats() -> dict[str, Any]:
    running = sum(1 for t in _running.values() if t and not t.done())
    queued = len(_queued_ids)
    return {
        "concurrency": job_concurrency(),
        "running": running,
        "queued": queued,
        "workers_started": _workers_started,
    }


async def recover_queued_jobs() -> None:
    """On startup, re-queue jobs left in queued/running state (server restart)."""
    global _started
    if _started:
        return
    _started = True
    await ensure_worker_pool()
    root = jobs_root()
    recovered = 0
    # FIFO by created_at / mtime ascending so oldest runs first
    paths = sorted(root.glob("*.json"), key=lambda p: p.stat().st_mtime)
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        status = data.get("status")
        job_id = data.get("id")
        if not job_id or status not in ("queued", "running", "awaiting_approval"):
            continue
        if status in ("running", "awaiting_approval"):
            data["status"] = "queued"
            data["error"] = "Recovered after server restart"
            append_event(data, "recovered", message="Re-queued after restart")
            save_job(data)
        enqueue_job(str(job_id))
        recovered += 1
    if recovered:
        logger.info("recovered %s job(s) after startup", recovered)


async def retry_job(job_id: str) -> dict[str, Any] | None:
    """Create a new job from a failed/completed job's inputs, or re-queue if still queued."""
    old = load_job(job_id)
    if not old:
        return None
    if old.get("status") in ("queued", "running"):
        # already in flight — just return it
        return old
    new = create_job(
        messages=list(old.get("messages") or []),
        attachments=list(old.get("attachments") or []),
        session_id=old.get("session_id"),
        bot_id=old.get("bot_id"),
        instructions=old.get("instructions"),
    )
    new["retried_from"] = job_id
    append_event(new, "retry", message=f"Retry of {job_id}")
    save_job(new)
    enqueue_job(new["id"])
    return new
