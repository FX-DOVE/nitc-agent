# Nitc Agent

Self-hosted AI agent with a web chat UI, tool-calling loop, sandboxed shell/filesystem, **interactive Linux desktop (noVNC)**, headless browser (Playwright), and GitHub (`gh`) tooling.

**Phase 2** adds a persistent, viewable/clickable desktop the user can watch and take over — open-source from scratch (MIT). Not affiliated with any proprietary agent product.

## Features

| Capability | Tools / UI |
|---|---|
| Chat UI | Dark SPA at `/` — sidebar (New chat / Computer), message bubbles, fixed composer |
| Agent brain | OpenAI-compatible tool-calling loop (OpenRouter defaults) |
| Computer (CLI) | `shell`, `read_file`, `write_file`, `list_dir` (workspace-scoped) |
| **Interactive desktop** | `desktop_screenshot`, `desktop_click`, `desktop_type`, `desktop_hotkey`, `desktop_scroll`, `desktop_open_browser` |
| Browser (headless) | `browser_navigate`, `browser_get_text`, `browser_screenshot` (Playwright) |
| GitHub | `github_run` — wraps `gh` with `GITHUB_TOKEN` |
| Watch / take over | noVNC via **`/novnc/`** on port **8080** (Computer view; :6080 optional direct) |

## Architecture (Phase 2)

```
┌────────────────────┐     Docker network `nitc`     ┌──────────────────────────┐
│  agent (:8080)     │  HTTP computer tools           │  desktop                 │
│  FastAPI chat UI   │ ────────────────────────────► │  Xvfb + XFCE (lean)      │
│  + /novnc/ proxy   │    http://desktop:7090         │  Chromium, Thunar, term  │
│  + media screenshots│   WS/HTTP /novnc → :6080      │  desktop-api (:7090)     │
│                    │   shared volume ./workspace    │  x11vnc + noVNC (:6080)  │
└────────────────────┘ ◄────────────────────────────► └──────────────────────────┘
         ▲
         │ browser :8080  (chat + same-origin /novnc/ — mobile-friendly)
         └──────────────── user
```

- **Preferred pattern:** computer-use tools in the agent call a small **desktop-api** (FastAPI) *inside* the desktop container. That avoids fragile cross-container `DISPLAY` networking.
- Workspace files are shared via `./workspace` so screenshots and downloads persist for both services.
- Headless Playwright stays on the **agent** image for fast scrapes; GUI work uses the **desktop**.

## Quick start

### 1. Clone

```bash
git clone https://github.com/FX-DOVE/nitc-agent.git
cd nitc-agent
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — set OPENAI_API_KEY (and optionally GITHUB_TOKEN, VNC_PASSWORD)
```

Defaults target **OpenRouter** with a free chat model:

```bash
# OpenRouter (get a real key at https://openrouter.ai/keys — never invent keys)
OPENAI_API_KEY=sk-or-v1-...
OPENAI_BASE_URL=https://openrouter.ai/api/v1
MODEL=meta-llama/llama-3.3-70b-instruct:free

# Tryout default: no VNC password (INSECURE on the open internet)
VNC_PASSWORD=
VNC_NO_PASSWORD=1
NOVNC_PUBLIC_URL=http://localhost:6080
```

Free OpenRouter models and rate limits change over time. If the default is unavailable, pick another `:free` model from [openrouter.ai/models](https://openrouter.ai/models?q=free) and set `MODEL` accordingly.

On a VPS you mainly need port **8080** (chat + proxied noVNC at `/novnc/`). `NOVNC_PUBLIC_URL` is optional for a direct :6080 link; mobile carriers often block 6080.

### 3. Run chat + live desktop

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| Chat UI | http://localhost:8080 |
| Live desktop (same-origin) | http://localhost:8080/novnc/vnc.html |
| Live desktop (direct, optional) | http://localhost:6080/vnc.html |

Health: `GET /health` · UI config: `GET /api/config`

### 4. Watch / take over the desktop

1. Open the chat UI → **Computer** in the sidebar (or open noVNC fullscreen).
2. Click **Connect** — with the tryout defaults there is **no password** (`VNC_NO_PASSWORD=1`).
3. Ask the agent to open a site or click around — you see it live and can click/type yourself in the same session (`x11vnc -shared`). Screenshots also appear **inline in chat**.

The desktop image runs a **polished lean XFCE** session: soft wallpaper, Greybird + Papirus theme, bottom panel with Browser / Terminal / Files launchers, Chromium, `xfce4-terminal`, and Thunar — meant to feel like a real computer view, not a blank X root.

### 5. Stop

```bash
docker compose down
```

## Desktop image size / VPS memory

The XFCE desktop image is heavier than a bare openbox setup (often ~1–1.5 GB on disk after build). On a **4 GB VPS** leave headroom for the agent container: Compose uses `shm_size: 256mb` per service. If the host OOMs, lower resolution (`SCREEN_WIDTH` / `SCREEN_HEIGHT`) or add swap; do not raise Chromium flags that increase GPU/RAM use.

## Security (important)

- **Open no-password VNC on the internet is insecure.** The defaults (`VNC_PASSWORD=` empty + `VNC_NO_PASSWORD=1`) are for a quick tryout only. Anyone who can reach the desktop stream can control it.
- Prefer locking down for anything beyond a demo:
  - set a `VNC_PASSWORD` and `VNC_NO_PASSWORD=0`, and/or
  - firewall allowlist (e.g. `ufw allow from YOUR_IP to any port 8080`), and/or
  - reverse proxy with TLS + basic auth / SSO in front of the agent.
- Port **6080** is optional (direct noVNC). The UI embeds **same-origin `/novnc/`** on **8080** (HTTP + WebSocket reverse proxy) so phones work when carriers block 6080.
- Desktop-api port **7090** is **not** published to the host — only reachable on the Compose network.
- Never commit `.env` or tokens (see `.gitignore`).
- File tools stay confined to `WORKSPACE_DIR`; shell boundary is the container.

## When to use which browser

| Goal | Use |
|---|---|
| User should **see** the GUI / take over | `desktop_*` tools + Computer view |
| Quick scrape / extract text | Playwright `browser_*` tools |

## API

### `POST /api/chat`

```json
{ "messages": [ { "role": "user", "content": "Open example.com on the desktop and screenshot it" } ] }
```

### `GET /api/config`

Returns `novnc_public_url` and `novnc_embed_url` for the SPA (no secrets).

## Project layout

```
nitc-agent/
  app/
    main.py              # FastAPI: /health, /api/chat, /api/config, /novnc/ proxy, /api/media, static UI
    agent.py             # tool-calling loop + system prompt
    config.py
    tools/               # shell, files, browser, github, computer
    static/index.html    # Chat + Computer (noVNC) UI
  desktop/
    entrypoint.sh        # Xvfb, XFCE, x11vnc, websockify, desktop-api
    api/main.py          # desktop-api (screenshot/click/type/…)
  workspace/             # shared sandbox volume
  Dockerfile             # agent image
  Dockerfile.desktop     # XFCE desktop + noVNC + desktop-api
  docker-compose.yml
  .env.example
```

## Roadmap

- **Phase 1**: chat UI, agent loop, shell/files/browser/GitHub, Docker
- **Phase 2** (this): interactive desktop, noVNC, computer-use tools, Computer view
- **Phase 3**: streaming SSE, multi-user sessions, more connectors, persistent memory

## License

MIT — see [LICENSE](LICENSE).
