"""Pluggable tool registry for Nitc Agent."""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable

from app.tools import browser, files, github_tool, shell

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
