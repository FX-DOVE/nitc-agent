"""GitHub tool — wraps the gh CLI with GITHUB_TOKEN."""

from __future__ import annotations

import asyncio
import json
import os
import shlex

from app.config import get_settings


async def github_run(args: str) -> str:
    """Run `gh <args>` with token from env. Returns stdout/stderr/exit code."""
    settings = get_settings()
    token = settings.github_token or os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or ""

    if not args or not args.strip():
        return json.dumps({"ok": False, "error": "args is required (e.g. 'api user')"})

    # Prevent shell injection: split into argv, never pass through bash -c
    try:
        argv = shlex.split(args)
    except ValueError as exc:
        return json.dumps({"ok": False, "error": f"Could not parse args: {exc}"})

    if argv and argv[0] == "gh":
        argv = argv[1:]

    env = dict(os.environ)
    if token:
        env["GH_TOKEN"] = token
        env["GITHUB_TOKEN"] = token
    env["GH_PROMPT_DISABLED"] = "1"
    env["GH_NO_UPDATE_NOTIFIER"] = "1"

    workspace = str(settings.workspace_path)
    proc = await asyncio.create_subprocess_exec(
        "gh",
        *argv,
        cwd=workspace,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=120)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return json.dumps({"ok": False, "error": "gh timed out after 120s", "exit_code": -1})

    stdout = stdout_b.decode("utf-8", errors="replace")[-50_000:]
    stderr = stderr_b.decode("utf-8", errors="replace")[-20_000:]

    hint = ""
    if not token and proc.returncode != 0:
        hint = " Set GITHUB_TOKEN in .env for authenticated gh commands."

    return json.dumps(
        {
            "ok": proc.returncode == 0,
            "exit_code": proc.returncode,
            "stdout": stdout,
            "stderr": stderr + hint,
            "args": argv,
        }
    )
