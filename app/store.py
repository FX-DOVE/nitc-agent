"""SQLite durable store for bots + messages (workspace/nitc.db).

Migrates from workspace/bots.json and workspace/sessions/*.json on first use.
JSON files remain as export/backup targets; SQLite is the source of truth.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import get_settings

logger = logging.getLogger("nitc.store")

_SAFE_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_lock = threading.RLock()
_migrated = False

_BOT_FIELDS = (
    "id",
    "name",
    "color",
    "shape",
    "kind",
    "pinned",
    "pinnedAt",
    "unread",
    "section",
    "hidden",
    "systemPrompt",
    "instructions",
    "snippet",
    "updatedAt",
    "updated_at",
    "last_snippet",
    "activeJobId",
    "activeJobIds",
    "pendingJobId",
    "lastJobId",
)


def db_path() -> Path:
    root = get_settings().workspace_path
    root.mkdir(parents=True, exist_ok=True)
    return root / "nitc.db"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _connect() -> sqlite3.Connection:
    path = db_path()
    conn = sqlite3.connect(str(path), timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS bots (
            id TEXT PRIMARY KEY,
            data_json TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            updated_ms INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bot_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL DEFAULT '',
            attachments_json TEXT NOT NULL DEFAULT '[]',
            images_json TEXT NOT NULL DEFAULT '[]',
            job_id TEXT,
            status TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (bot_id) REFERENCES bots(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_messages_bot_created
            ON messages(bot_id, created_at, id);
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )
    conn.commit()


def _rotate_backup(path: Path, keep: int = 3) -> None:
    if not path.is_file():
        return
    try:
        bak = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, bak)
        # rotate older
        for i in range(keep, 0, -1):
            older = path.with_suffix(f"{path.suffix}.bak.{i}")
            newer = path.with_suffix(f"{path.suffix}.bak.{i-1}") if i > 1 else bak
            if i == 1:
                continue
            if newer.is_file():
                shutil.copy2(newer, older)
    except OSError as exc:
        logger.warning("backup rotate failed for %s: %s", path, exc)


def _sanitize_bot(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    bid = str(raw.get("id") or "").strip()
    if not bid:
        return None
    if not _SAFE_ID.match(bid):
        bid = re.sub(r"[^A-Za-z0-9_.:-]+", "_", bid)[:128] or "bot"
    out: dict[str, Any] = {"id": bid}
    for key in _BOT_FIELDS:
        if key == "id":
            continue
        if key in raw and raw[key] is not None:
            out[key] = raw[key]
    updated = out.get("updatedAt") or out.get("updated_at")
    if isinstance(updated, str):
        try:
            updated = int(
                datetime.fromisoformat(updated.replace("Z", "+00:00")).timestamp() * 1000
            )
        except Exception:
            updated = _now_ms()
    if not isinstance(updated, (int, float)):
        updated = _now_ms()
    out["updatedAt"] = int(updated)
    out["updated_at"] = _now_iso()
    if "name" not in out or not str(out.get("name") or "").strip():
        out["name"] = "Bot"
    out["name"] = str(out["name"])[:80]
    snippet = out.get("last_snippet") or out.get("snippet") or ""
    out["snippet"] = str(snippet).replace("\n", " ")[:120]
    out["last_snippet"] = out["snippet"]
    out.pop("messages", None)
    return out


def _safe_sid(session_id: str) -> str:
    sid = (session_id or "").strip() or "default"
    if not _SAFE_ID.match(sid):
        sid = re.sub(r"[^A-Za-z0-9_.:-]+", "_", sid)[:128] or "default"
    return sid


def init_db() -> None:
    """Create schema and migrate JSON → SQLite once."""
    global _migrated
    with _lock:
        conn = _connect()
        try:
            _ensure_schema(conn)
            row = conn.execute(
                "SELECT value FROM meta WHERE key = 'migrated_json_v1'"
            ).fetchone()
            if row and row["value"] == "1":
                _migrated = True
                return
            _migrate_from_json(conn)
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES ('migrated_json_v1', '1')"
            )
            conn.commit()
            _migrated = True
            logger.info("SQLite store ready at %s", db_path())
        finally:
            conn.close()


def _migrate_from_json(conn: sqlite3.Connection) -> None:
    root = get_settings().workspace_path
    bots_file = root / "bots.json"
    bots: list[dict[str, Any]] = []
    if bots_file.is_file():
        try:
            data = json.loads(bots_file.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("bots"), list):
                bots = [b for b in (_sanitize_bot(x) for x in data["bots"]) if b]
            elif isinstance(data, list):
                bots = [b for b in (_sanitize_bot(x) for x in data) if b]
            logger.info("migrating %d bots from bots.json", len(bots))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("bots.json migrate failed: %s", exc)

    for bot in bots:
        conn.execute(
            "INSERT OR REPLACE INTO bots(id, data_json, updated_at, updated_ms) VALUES (?,?,?,?)",
            (
                bot["id"],
                json.dumps(bot, ensure_ascii=False),
                bot.get("updated_at") or _now_iso(),
                int(bot.get("updatedAt") or _now_ms()),
            ),
        )

    sessions_dir = root / "sessions"
    if sessions_dir.is_dir():
        for path in sorted(sessions_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict):
                continue
            sid = _safe_sid(str(data.get("session_id") or path.stem))
            # ensure bot row exists so FK works
            exists = conn.execute("SELECT 1 FROM bots WHERE id = ?", (sid,)).fetchone()
            if not exists:
                stub = _sanitize_bot({"id": sid, "name": sid})
                if stub:
                    conn.execute(
                        "INSERT OR IGNORE INTO bots(id, data_json, updated_at, updated_ms) VALUES (?,?,?,?)",
                        (
                            stub["id"],
                            json.dumps(stub, ensure_ascii=False),
                            stub["updated_at"],
                            stub["updatedAt"],
                        ),
                    )
            # skip if messages already present
            count = conn.execute(
                "SELECT COUNT(*) AS c FROM messages WHERE bot_id = ?", (sid,)
            ).fetchone()["c"]
            if count:
                continue
            msgs = data.get("messages") or []
            if not isinstance(msgs, list):
                continue
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                role = str(m.get("role") or "user")
                content = m.get("content")
                if not isinstance(content, str):
                    content = str(content or "")
                atts = m.get("attachments") or m.get("attachments_json") or []
                images = m.get("images") or []
                created = str(m.get("ts") or m.get("created_at") or _now_iso())
                conn.execute(
                    """INSERT INTO messages(
                        bot_id, role, content, attachments_json, images_json,
                        job_id, status, created_at
                    ) VALUES (?,?,?,?,?,?,?,?)""",
                    (
                        sid,
                        role,
                        content,
                        json.dumps(atts, ensure_ascii=False),
                        json.dumps(images, ensure_ascii=False),
                        m.get("job_id"),
                        m.get("status"),
                        created,
                    ),
                )
            logger.info("migrated session %s (%d msgs)", sid, len(msgs))


def load_bots() -> list[dict[str, Any]]:
    init_db()
    with _lock:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT data_json FROM bots ORDER BY updated_ms DESC"
            ).fetchall()
            out: list[dict[str, Any]] = []
            for row in rows:
                try:
                    bot = _sanitize_bot(json.loads(row["data_json"]))
                    if bot:
                        out.append(bot)
                except json.JSONDecodeError:
                    continue
            return out[:200]
        finally:
            conn.close()


def save_bots(bots: list[Any]) -> list[dict[str, Any]]:
    init_db()
    cleaned: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in bots or []:
        bot = _sanitize_bot(item)
        if not bot or bot["id"] in seen:
            continue
        seen.add(bot["id"])
        cleaned.append(bot)
    cleaned = cleaned[:200]
    with _lock:
        conn = _connect()
        try:
            existing = {
                r["id"] for r in conn.execute("SELECT id FROM bots").fetchall()
            }
            new_ids = {b["id"] for b in cleaned}
            # upsert
            for bot in cleaned:
                conn.execute(
                    "INSERT OR REPLACE INTO bots(id, data_json, updated_at, updated_ms) VALUES (?,?,?,?)",
                    (
                        bot["id"],
                        json.dumps(bot, ensure_ascii=False),
                        bot.get("updated_at") or _now_iso(),
                        int(bot.get("updatedAt") or _now_ms()),
                    ),
                )
            # delete removed (messages cascade)
            for old_id in existing - new_ids:
                conn.execute("DELETE FROM bots WHERE id = ?", (old_id,))
            conn.commit()
        finally:
            conn.close()
        # JSON mirror + .bak for ops
        _export_bots_json(cleaned)
    return cleaned


def _export_bots_json(bots: list[dict[str, Any]]) -> None:
    path = get_settings().workspace_path / "bots.json"
    try:
        _rotate_backup(path)
        payload = {"bots": bots, "updated_at": _now_iso()}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        logger.warning("bots.json export failed: %s", exc)


def public_bots_payload(bots: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    items = bots if bots is not None else load_bots()
    return {
        "bots": items,
        "updated_at": _now_iso(),
        "count": len(items),
        "storage": "sqlite",
    }


def ensure_bot(bot_id: str, name: str | None = None) -> None:
    init_db()
    sid = _safe_sid(bot_id)
    with _lock:
        conn = _connect()
        try:
            row = conn.execute("SELECT 1 FROM bots WHERE id = ?", (sid,)).fetchone()
            if row:
                return
            stub = _sanitize_bot({"id": sid, "name": name or sid})
            if not stub:
                return
            conn.execute(
                "INSERT OR IGNORE INTO bots(id, data_json, updated_at, updated_ms) VALUES (?,?,?,?)",
                (stub["id"], json.dumps(stub, ensure_ascii=False), stub["updated_at"], stub["updatedAt"]),
            )
            conn.commit()
        finally:
            conn.close()


def append_message(
    bot_id: str,
    *,
    role: str,
    content: str,
    attachments: list[Any] | None = None,
    images: list[Any] | None = None,
    job_id: str | None = None,
    status: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    init_db()
    sid = _safe_sid(bot_id)
    ensure_bot(sid)
    created = created_at or _now_iso()
    atts = attachments or []
    imgs = images or []
    with _lock:
        conn = _connect()
        try:
            if job_id:
                dup = conn.execute(
                    "SELECT id FROM messages WHERE bot_id = ? AND job_id = ? AND role = ? ORDER BY id DESC LIMIT 1",
                    (sid, job_id, str(role or "user")),
                ).fetchone()
                if dup:
                    return {
                        "id": dup["id"],
                        "bot_id": sid,
                        "role": role,
                        "content": content if isinstance(content, str) else str(content or ""),
                        "attachments": atts,
                        "images": imgs,
                        "job_id": job_id,
                        "status": status,
                        "created_at": created,
                        "ts": created,
                        "deduped": True,
                    }
            cur = conn.execute(
                """INSERT INTO messages(
                    bot_id, role, content, attachments_json, images_json,
                    job_id, status, created_at
                ) VALUES (?,?,?,?,?,?,?,?)""",
                (
                    sid,
                    str(role or "user"),
                    content if isinstance(content, str) else str(content or ""),
                    json.dumps(atts, ensure_ascii=False),
                    json.dumps(imgs, ensure_ascii=False),
                    job_id,
                    status,
                    created,
                ),
            )
            # trim to last 500 per bot
            conn.execute(
                """DELETE FROM messages WHERE bot_id = ? AND id NOT IN (
                    SELECT id FROM messages WHERE bot_id = ? ORDER BY id DESC LIMIT 500
                )""",
                (sid, sid),
            )
            # bump bot snippet for user/assistant
            if role in ("user", "assistant") and content:
                snip = str(content).replace("\n", " ")[:120]
                row = conn.execute("SELECT data_json FROM bots WHERE id = ?", (sid,)).fetchone()
                if row:
                    try:
                        bot = json.loads(row["data_json"])
                    except json.JSONDecodeError:
                        bot = {"id": sid}
                    bot["snippet"] = snip
                    bot["last_snippet"] = snip
                    bot["updatedAt"] = _now_ms()
                    bot["updated_at"] = _now_iso()
                    if job_id and role == "assistant":
                        bot["lastJobId"] = job_id
                    conn.execute(
                        "UPDATE bots SET data_json = ?, updated_at = ?, updated_ms = ? WHERE id = ?",
                        (
                            json.dumps(bot, ensure_ascii=False),
                            bot["updated_at"],
                            bot["updatedAt"],
                            sid,
                        ),
                    )
            conn.commit()
            msg_id = cur.lastrowid
        finally:
            conn.close()
    return {
        "id": msg_id,
        "bot_id": sid,
        "role": role,
        "content": content,
        "attachments": atts,
        "images": imgs,
        "job_id": job_id,
        "status": status,
        "created_at": created,
        "ts": created,
    }


def list_messages(bot_id: str, *, limit: int = 200) -> list[dict[str, Any]]:
    init_db()
    sid = _safe_sid(bot_id)
    limit = max(1, min(500, int(limit or 200)))
    with _lock:
        conn = _connect()
        try:
            rows = conn.execute(
                """SELECT id, role, content, attachments_json, images_json, job_id, status, created_at
                   FROM messages WHERE bot_id = ? ORDER BY id ASC""",
                (sid,),
            ).fetchall()
        finally:
            conn.close()
    # return last N
    rows = rows[-limit:]
    out: list[dict[str, Any]] = []
    for r in rows:
        try:
            atts = json.loads(r["attachments_json"] or "[]")
        except json.JSONDecodeError:
            atts = []
        try:
            images = json.loads(r["images_json"] or "[]")
        except json.JSONDecodeError:
            images = []
        out.append(
            {
                "id": r["id"],
                "role": r["role"],
                "content": r["content"],
                "attachments": atts,
                "images": images,
                "job_id": r["job_id"],
                "status": r["status"],
                "created_at": r["created_at"],
                "ts": r["created_at"],
            }
        )
    return out


def get_session_payload(session_id: str) -> dict[str, Any]:
    """Shape compatible with legacy /api/sessions/{id}."""
    sid = _safe_sid(session_id)
    msgs = list_messages(sid, limit=200)
    return {
        "session_id": sid,
        "bot_id": sid,
        "messages": msgs,
        "jobs": [],
        "updated_at": msgs[-1]["created_at"] if msgs else None,
        "storage": "sqlite",
    }
