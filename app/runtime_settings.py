"""Persisted UI/runtime settings under workspace/settings.json (auto-review, etc.)."""

from __future__ import annotations

import json
import logging
import re
import secrets
import threading
import time
from pathlib import Path
from typing import Any

from app.config import get_settings

logger = logging.getLogger("nitc.settings")
_lock = threading.RLock()

# Everyday coding / inspection — never gated by default Auto-review.
FREE_TOOLS = frozenset({
    "read_file",
    "write_file",
    "list_dir",
    "zip_paths",
    "desktop_screenshot",
    "browser_screenshot",
    "browser_get_text",
})

# May be gated depending on args / patterns (not blanket).
CONDITIONAL_TOOLS = frozenset({
    "shell",
    "browser_navigate",
    "desktop_open_browser",
    "github_run",
    "desktop_click",
    "desktop_type",
    "desktop_hotkey",
    "desktop_scroll",
})

# Legacy alias — prefer should_require_approval()
RISKY_TOOLS = frozenset(CONDITIONAL_TOOLS | {"write_file"})  # write_file kept for old UI copy only

# Shell commands that are everyday under workspace — free when auto-review is on.
_SHELL_ALLOW_RE = re.compile(
    r"(?ix)^("
    r"ls|pwd|whoami|date|uname|echo|printf|cat|head|tail|wc|file|stat|"
    r"mkdir|touch|cp|mv|ln|chmod|chown|tee|find|grep|rg|sed|awk|sort|uniq|diff|"
    r"python3?|pip3?|node|npm|npx|yarn|pnpm|ruby|perl|php|"
    r"zip|unzip|tar|gzip|gunzip|xz|7z|"
    r"git|gh|"
    r"which|type|command|env|printenv|export|"
    r"test|\[|"
    r"cd"
    r")\b"
)

# Destructive / exfil / privilege — always gate when auto-review is on.
_SHELL_DENY_RE = [
    re.compile(r"(?i)\brm\s+(-[a-zA-Z]*\s+)*-?[rR].*\b(-[a-zA-Z]*f|/|~)"),
    re.compile(r"(?i)\brm\s+-rf\b"),
    re.compile(r"(?i)\bmkfs\b"),
    re.compile(r"(?i)\bdd\s+.*\bof=/dev/"),
    re.compile(r"(?i)\b(shutdown|reboot|halt|poweroff)\b"),
    re.compile(r"(?i)\b(mount|umount)\b"),
    re.compile(r"(?i)\bsudo\b"),
    re.compile(r"(?i)\b(curl|wget)\b.*\|\s*(ba)?sh\b"),
    re.compile(r"(?i)\b(ba)?sh\s*<\s*\(\s*(curl|wget)\b"),
    re.compile(r"(?i)\bssh\s+\S+@"),
    re.compile(r"(?i)\bscp\b"),
    re.compile(r"(?i)\brsync\b.*\s(-e|--rsh)"),
    re.compile(r"(?i)\b(chmod\s+(-R\s+)?777\s+/)"),
    re.compile(r"(?i)\b(nc|ncat|netcat)\b.*\s-e\b"),
    re.compile(r"(?i)(api[_-]?key|secret|token|password|passwd|authorization)\b.{0,40}\|\s*(curl|wget)"),
    re.compile(r"(?i)\bcurl\b.*\b(-d|--data|--upload-file|-T)\b.*(api[_-]?key|secret|token|password)"),
    re.compile(r"(?i)discord\.com/api/webhooks"),
    re.compile(r"(?i)\b(iptables|ufw|firewall-cmd)\b"),
    re.compile(r"(?i)\b(apk|apt-get|apt|dnf|yum)\s+(install|remove|purge)\b"),
    re.compile(r"(?i):\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;"),
]

_EXFIL_URL_RE = re.compile(
    r"(?i)(discord\.com/api/webhooks|hooks\.slack\.com|webhook\.site|"
    r"requestbin|pipedream\.net|ngrok\.|burpcollaborator|"
    r"pastebin\.com|transfer\.sh)"
)

DEFAULTS: dict[str, Any] = {
    "autoReview": False,
    "autoReviewRules": [],
    "tzAuto": True,
    "timeZone": "Africa/Lagos",
    "notifications": True,
    "appearance": "Black",
    "language": "System",
    "haptics": True,
    "plugins": {
        "browser": True,
        "desktop": True,
        "files": True,
        "shell": True,
        "github": True,
    },
}


def _path() -> Path:
    return get_settings().workspace_path / "settings.json"


def load_settings() -> dict[str, Any]:
    with _lock:
        path = _path()
        data = dict(DEFAULTS)
        if path.is_file():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    data.update(raw)
            except Exception:  # noqa: BLE001
                logger.exception("failed reading settings.json")
        if not isinstance(data.get("autoReviewRules"), list):
            data["autoReviewRules"] = []
        if not isinstance(data.get("plugins"), dict):
            data["plugins"] = dict(DEFAULTS["plugins"])
        data.setdefault("_allow_once", [])
        return data


def save_settings(patch: dict[str, Any]) -> dict[str, Any]:
    with _lock:
        cur = load_settings()
        for k, v in patch.items():
            if k.startswith("_"):
                continue
            cur[k] = v
        path = _path()
        to_write = {k: v for k, v in cur.items() if not k.startswith("_") or k == "_allow_once"}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(to_write, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return cur


def grant_allow_once(tool: str, args: dict[str, Any] | None = None, ttl_sec: int = 120) -> str:
    """Permit one matching risky tool call within ttl_sec. Returns token id."""
    token = secrets.token_urlsafe(12)
    with _lock:
        cur = load_settings()
        allows = list(cur.get("_allow_once") or [])
        allows.append({
            "id": token,
            "tool": tool,
            "args": args or {},
            "expires": time.time() + ttl_sec,
        })
        now = time.time()
        allows = [a for a in allows if float(a.get("expires") or 0) > now]
        cur["_allow_once"] = allows
        path = _path()
        to_write = {k: v for k, v in cur.items() if not k.startswith("_") or k == "_allow_once"}
        path.write_text(json.dumps(to_write, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return token


def consume_allow_once(tool: str, args: dict[str, Any]) -> bool:
    """Return True and consume if a matching allow-once grant exists."""
    with _lock:
        cur = load_settings()
        allows = list(cur.get("_allow_once") or [])
        now = time.time()
        kept: list[dict[str, Any]] = []
        matched = False
        for a in allows:
            if float(a.get("expires") or 0) <= now:
                continue
            if matched:
                kept.append(a)
                continue
            if a.get("tool") != tool:
                kept.append(a)
                continue
            matched = True
        if matched:
            cur["_allow_once"] = kept
            path = _path()
            to_write = {k: v for k, v in cur.items() if not k.startswith("_") or k == "_allow_once"}
            path.write_text(json.dumps(to_write, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return matched


def auto_review_enabled() -> bool:
    return bool(load_settings().get("autoReview"))


def auto_review_rules() -> list[str]:
    rules = load_settings().get("autoReviewRules") or []
    return [str(r) for r in rules if str(r).strip()]


def _rules_match(tool: str, args: dict[str, Any], rules: list[str]) -> str | None:
    blob = f"{tool} {json.dumps(args, ensure_ascii=False)}".lower()
    for rule in rules:
        r = rule.strip()
        if not r:
            continue
        # simple substring or glob-ish *
        if "*" in r:
            parts = [re.escape(p) for p in r.lower().split("*")]
            if re.search(".*".join(parts), blob):
                return rule
        elif r.lower() in blob:
            return rule
    return None


def _shell_is_risky(command: str) -> tuple[bool, str]:
    cmd = (command or "").strip()
    if not cmd:
        return False, ""
    for pat in _SHELL_DENY_RE:
        if pat.search(cmd):
            return True, f"destructive_or_exfil:{pat.pattern[:60]}"
    # pip/npm install in workspace — free; system package managers already denied
    # Multi-command: if any segment is deny-matched above we're done; otherwise allow
    # everyday prefixes. Unknown/complex shells (curl http without pipe) stay free
    # unless they look like exfil (covered above).
    # Soft mkdir/zip/python already allow-listed by not matching deny.
    # Gate only when deny matches OR command starts with clearly privileged ops not listed.
    return False, ""


def _url_is_risky(url: str) -> tuple[bool, str]:
    u = (url or "").strip()
    if not u:
        return False, ""
    if _EXFIL_URL_RE.search(u):
        return True, "suspicious_exfil_url"
    return False, ""


def should_require_approval(tool: str, args: dict[str, Any] | None = None) -> tuple[bool, str]:
    """
    Smart Auto-review: return (needs_approval, risk_reason).
    If Auto-review is OFF → never.
    Everyday file/zip/screenshot tools → never.
    Shell/network → only when destructive / exfil-like (or custom rules match).
    Benign desktop clicks/typing → free unless a user rule matches.
    """
    if not auto_review_enabled():
        return False, ""
    args = args or {}
    name = (tool or "").strip()
    rules = auto_review_rules()

    # User custom rules can force a gate even on free tools
    matched = _rules_match(name, args, rules)
    if matched and name not in FREE_TOOLS:
        return True, f"rule:{matched}"
    if matched and name in FREE_TOOLS:
        # Only force-gate free tools if rule explicitly mentions the tool name
        if name.lower() in matched.lower():
            return True, f"rule:{matched}"

    if name in FREE_TOOLS:
        return False, ""

    if name == "shell":
        risky, reason = _shell_is_risky(str(args.get("command") or ""))
        if risky:
            return True, reason
        # Custom rules may still match shell command text
        if matched:
            return True, f"rule:{matched}"
        return False, ""

    if name in ("browser_navigate", "desktop_open_browser"):
        risky, reason = _url_is_risky(str(args.get("url") or ""))
        if risky:
            return True, reason
        return False, ""

    if name == "github_run":
        # Mutating GitHub actions — gate when auto-review is on
        gh_args = str(args.get("args") or "").lower()
        mutating = any(
            k in gh_args
            for k in (
                " create", "delete", "repo create", "issue create", "pr create",
                "pr merge", "release create", "secret", "workflow run",
                "api -x post", "api -x put", "api -x patch", "api -x delete",
                "-x post", "-x put", "-x patch", "-x delete",
            )
        ) or gh_args.strip().startswith(("repo create", "issue create", "pr create"))
        if mutating or matched:
            return True, "github_mutating" if mutating else f"rule:{matched}"
        return False, ""

    if name in ("desktop_click", "desktop_type", "desktop_hotkey", "desktop_scroll"):
        # Benign desktop — free unless user rule matches
        if matched:
            return True, f"rule:{matched}"
        return False, ""

    # Unknown tools: free (prefer acting)
    return False, ""
