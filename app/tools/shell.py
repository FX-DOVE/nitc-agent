"""Sandboxed shell tool — runs commands in the workspace directory."""

from __future__ import annotations

import asyncio
import os
import json
import re

from app.config import get_settings

# Patterns that look like host-escape / destructive attempts outside workspace.
# These are best-effort guards; the real boundary is the container + cwd.
_BLOCKED_PATTERNS = [
    re.compile(r"(^|[;&|]\s*)rm\s+(-[a-zA-Z]*\s+)*(/|~|/etc|/usr|/var|/home|/root|/boot)", re.I),
    re.compile(r"(^|[;&|]\s*)mkfs\b", re.I),
    re.compile(r"(^|[;&|]\s*)dd\s+.*\bof=/dev/", re.I),
    re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;", re.I),  # fork bomb
    re.compile(r"(^|[;&|]\s*)shutdown\b", re.I),
    re.compile(r"(^|[;&|]\s*)reboot\b", re.I),
    re.compile(r"(^|[;&|]\s*)mount\b", re.I),
    re.compile(r"(^|[;&|]\s*)umount\b", re.I),
    re.compile(r"(^|[;&|]\s*)chmod\s+(-R\s+)?777\s+/", re.I),
    re.compile(r"(^|[;&|]\s*)chown\s+.*\s+/", re.I),
]


def _is_blocked(command: str) -> str | None:
    for pat in _BLOCKED_PATTERNS:
        if pat.search(command):
            return f"Blocked potentially destructive command matching: {pat.pattern}"
    return None


async def run_shell(command: str, timeout: int | None = None) -> str:
    settings = get_settings()
    workspace = str(settings.workspace_path)
    timeout = timeout or settings.shell_timeout_seconds

    blocked = _is_blocked(command)
    if blocked:
        return json.dumps({"ok": False, "error": blocked, "exit_code": -1})

    # Run via bash -lc so pipes/redirects work, but force cwd to workspace.
    proc = await asyncio.create_subprocess_exec(
        "bash",
        "-lc",
        command,
        cwd=workspace,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={
            **dict(os.environ),
            "HOME": workspace,
            "PWD": workspace,
        },
    )

    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return json.dumps(
            {
                "ok": False,
                "error": f"Command timed out after {timeout}s",
                "exit_code": -1,
                "command": command,
            }
        )

    stdout = stdout_b.decode("utf-8", errors="replace")[-50_000:]
    stderr = stderr_b.decode("utf-8", errors="replace")[-20_000:]
    return json.dumps(
        {
            "ok": proc.returncode == 0,
            "exit_code": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "cwd": workspace,
            "command": command,
        }
    )
