"""Pluggable tool registry for Nitc Agent."""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable

from app.tools import browser, computer, files, github_tool, shell

ToolHandler = Callable[..., Awaitable[str] | str]

# OpenAI-compatible tool definitions + handlers
TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": (
                "Run a shell command inside the agent workspace directory. "
                "Returns stdout, stderr, and exit code. Use for builds, git, "
                "scripts, and general computer work. Do not attempt host escapes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (optional, default from config)",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file relative to the workspace directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path under workspace",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write text content to a file under the workspace directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path under workspace",
                    },
                    "content": {
                        "type": "string",
                        "description": "File contents to write",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and directories under a workspace-relative path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative directory path (default '.')",
                        "default": ".",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_navigate",
            "description": "Open a URL in the headless Chromium browser and return page title + URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full URL to open (http/https)",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_get_text",
            "description": "Get visible text content from the current browser page (or navigate first if url given).",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Optional URL to navigate to first",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Max characters to return (default 15000)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_screenshot",
            "description": "Take a screenshot of the current page (or navigate first). Returns saved path under workspace/screenshots.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Optional URL to navigate to first",
                    },
                    "filename": {
                        "type": "string",
                        "description": "Optional filename (png)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_run",
            "description": (
                "Run a GitHub CLI (gh) command using GITHUB_TOKEN. "
                "Pass arguments without the leading 'gh'. "
                "Examples: 'repo list --limit 5', 'api user', "
                "'repo create OWNER/NAME --public --description \"…\"'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "args": {
                        "type": "string",
                        "description": "Arguments after 'gh' (e.g. 'repo view OWNER/NAME')",
                    },
                },
                "required": ["args"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "desktop_screenshot",
            "description": (
                "Capture a screenshot of the interactive Linux desktop (user-visible via noVNC). "
                "Saves under workspace/screenshots/. Prefer this for GUI tasks over headless browser screenshots."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Optional png filename under screenshots/",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "desktop_click",
            "description": "Click at (x, y) on the interactive desktop. Button 1=left, 2=middle, 3=right.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "X pixel coordinate"},
                    "y": {"type": "integer", "description": "Y pixel coordinate"},
                    "button": {
                        "type": "integer",
                        "description": "Mouse button (1 left, 2 middle, 3 right). Default 1",
                        "default": 1,
                    },
                    "clicks": {
                        "type": "integer",
                        "description": "Click count (1 or 2 for double-click). Default 1",
                        "default": 1,
                    },
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "desktop_type",
            "description": (
                "Type text into the focused desktop window. "
                "Optionally press a key afterwards (Return, Tab, Escape, BackSpace, …)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to type"},
                    "key": {
                        "type": "string",
                        "description": "Optional key to press after typing (e.g. Return, Tab)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "desktop_hotkey",
            "description": "Press a key combination on the desktop, e.g. keys=['ctrl','c'] or ['alt','F4'].",
            "parameters": {
                "type": "object",
                "properties": {
                    "keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Ordered keys in the combo",
                    },
                },
                "required": ["keys"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "desktop_scroll",
            "description": "Scroll at (x, y) on the desktop. direction: up/down/left/right.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down", "left", "right"],
                        "default": "down",
                    },
                    "amount": {
                        "type": "integer",
                        "description": "Number of scroll notches (default 3)",
                        "default": 3,
                    },
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "desktop_open_browser",
            "description": (
                "Open a URL in Chromium on the interactive desktop (visible in noVNC). "
                "Use for GUI browsing the user can watch; use browser_* for headless scrapes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full http(s) URL",
                    },
                },
                "required": ["url"],
            },
        },
    },
]

HANDLERS: dict[str, ToolHandler] = {
    "shell": shell.run_shell,
    "read_file": files.read_file,
    "write_file": files.write_file,
    "list_dir": files.list_dir,
    "browser_navigate": browser.browser_navigate,
    "browser_get_text": browser.browser_get_text,
    "browser_screenshot": browser.browser_screenshot,
    "github_run": github_tool.github_run,
    "desktop_screenshot": computer.desktop_screenshot,
    "desktop_click": computer.desktop_click,
    "desktop_type": computer.desktop_type,
    "desktop_hotkey": computer.desktop_hotkey,
    "desktop_scroll": computer.desktop_scroll,
    "desktop_open_browser": computer.desktop_open_browser,
}


def get_openai_tools() -> list[dict[str, Any]]:
    return TOOL_SPECS


async def run_tool(name: str, arguments: dict[str, Any] | str | None) -> str:
    """Dispatch a tool call by name. Always returns a string result."""
    if name not in HANDLERS:
        return json.dumps({"error": f"Unknown tool: {name}"})

    if arguments is None:
        args: dict[str, Any] = {}
    elif isinstance(arguments, str):
        try:
            args = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as exc:
            return json.dumps({"error": f"Invalid JSON arguments: {exc}"})
    else:
        args = arguments

    # Auto-review gate for risky shell/desktop/network tools
    try:
        from app.runtime_settings import (
            RISKY_TOOLS,
            auto_review_enabled,
            auto_review_rules,
            consume_allow_once,
        )
        if name in RISKY_TOOLS and auto_review_enabled():
            if not consume_allow_once(name, args):
                rules = auto_review_rules()
                return json.dumps({
                    "needs_approval": True,
                    "auto_review": True,
                    "tool": name,
                    "args": args,
                    "rules": rules,
                    "error": "auto_review_blocked",
                    "message": (
                        "Auto-review is ON. This risky action was paused for user approval. "
                        "Do not retry the same tool until the user Approves in the chat UI "
                        "(or says they approved). Summarize what you wanted to do and wait."
                    ),
                })
    except Exception:  # noqa: BLE001
        pass

    # Plugin disable gate
    try:
        from app.runtime_settings import load_settings
        plugs = (load_settings().get("plugins") or {})
        plugin_for = {
            "shell": "shell",
            "read_file": "files", "write_file": "files", "list_dir": "files",
            "browser_navigate": "browser", "browser_get_text": "browser", "browser_screenshot": "browser",
            "github_run": "github",
            "desktop_screenshot": "desktop", "desktop_click": "desktop", "desktop_type": "desktop",
            "desktop_hotkey": "desktop", "desktop_scroll": "desktop", "desktop_open_browser": "desktop",
        }
        pk = plugin_for.get(name)
        if pk and plugs.get(pk) is False:
            return json.dumps({"error": f"Plugin '{pk}' is disabled in Profile → Plugins"})
    except Exception:  # noqa: BLE001
        pass

    try:
        result = HANDLERS[name](**args)
        if hasattr(result, "__await__"):
            result = await result  # type: ignore[misc]
        return result if isinstance(result, str) else json.dumps(result)
    except TypeError as exc:
        return json.dumps({"error": f"Bad arguments for {name}: {exc}"})
    except Exception as exc:  # noqa: BLE001 — surface tool failures to the model
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


def register_tool(
    name: str,
    handler: ToolHandler,
    *,
    description: str,
    parameters: dict[str, Any],
) -> None:
    """Register an additional tool at runtime (for future connectors)."""
    HANDLERS[name] = handler
    TOOL_SPECS.append(
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        }
    )
