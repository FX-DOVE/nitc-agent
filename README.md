# Nitc Agent

**Phase 1** — self-hosted AI agent with a web chat UI, tool-calling loop, sandboxed shell/filesystem, headless browser (Playwright), and GitHub (`gh`) tooling.

Built from scratch as clean open-source software (MIT). Not affiliated with any proprietary agent product.

## Features (Phase 1)

| Capability | Tools |
|---|---|
| Chat UI | Simple SPA served by FastAPI at `/` |
| Agent brain | OpenAI-compatible tool-calling loop (max ~15 rounds) |
| Computer | `shell`, `read_file`, `write_file`, `list_dir` (workspace-scoped) |
| Browser | `browser_navigate`, `browser_get_text`, `browser_screenshot` (Playwright Chromium) |
| GitHub | `github_run` — wraps `gh` CLI with `GITHUB_TOKEN` |
| Extensibility | Pluggable registry in `app/tools/__init__.py` (`register_tool`) |

Shell and browser run **inside the Docker container**. The agent workspace is mounted at `./workspace`.

## Quick start (VPS / local)

### 1. Clone

```bash
git clone https://github.com/FX-DOVE/nitc-agent.git
cd nitc-agent
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — at minimum set OPENAI_API_KEY
```

Common providers:

```bash
# OpenRouter
OPENAI_API_KEY=sk-or-v1-...
OPENAI_BASE_URL=https://openrouter.ai/api/v1
MODEL=openai/gpt-4o-mini

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL=gpt-4o-mini

# Local (e.g. Ollama OpenAI-compatible endpoint)
OPENAI_API_KEY=ollama
OPENAI_BASE_URL=http://host.docker.internal:11434/v1
MODEL=llama3.2
```

Optional:

```bash
GITHUB_TOKEN=ghp_...   # for github_run / gh inside the container
```

### 3. Run with Docker Compose

```bash
docker compose up --build
```

Open **http://localhost:8080**

Health check: `GET /health`

### 4. Stop

```bash
docker compose down
```

## API

### `POST /api/chat`

```json
{
  "messages": [
    { "role": "user", "content": "List files in the workspace" }
  ]
}
```

Or shorthand:

```json
{ "message": "What is the title of https://example.com ?" }
```

Response:

```json
{
  "reply": "...",
  "tool_rounds": 1,
  "messages": [ ]
}
```

## Project layout

```
nitc-agent/
  app/
    main.py          # FastAPI: /health, /api/chat, static UI
    agent.py         # tool-calling agent loop
    config.py        # env settings
    tools/           # shell, files, browser, github + registry
    static/index.html
  workspace/         # sandbox (persisted via compose volume)
  Dockerfile
  docker-compose.yml
  .env.example
```

## Safety notes

- Never commit `.env` or tokens (see `.gitignore`).
- Paths for file tools are confined to `WORKSPACE_DIR`.
- Shell has a best-effort blocklist for destructive host patterns; the real boundary is the container.
- Browser is headless Chromium inside the container (`shm_size: 256mb` in compose).

## Roadmap

- **Phase 1** (this repo): chat UI, agent loop, shell/files/browser/GitHub, Docker
- **Phase 2**: streaming SSE, multi-user sessions, more connectors (Slack, Drive, …)
- **Phase 3**: persistent memory, scheduled jobs, multi-agent workflows

## License

MIT — see [LICENSE](LICENSE).
