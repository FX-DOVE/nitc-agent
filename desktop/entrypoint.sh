#!/usr/bin/env bash
# Start Xvfb + lean XFCE desktop + x11vnc + noVNC + desktop-api
set -euo pipefail

export DISPLAY="${DISPLAY:-:99}"
export HOME="${HOME:-/home/desktop}"
export DESKTOP_WORKSPACE="${DESKTOP_WORKSPACE:-/home/desktop/workspace}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$HOME/.config}"
export XDG_DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-desktop}"
VNC_PASSWORD="${VNC_PASSWORD:-nitc}"
SCREEN_WIDTH="${SCREEN_WIDTH:-1280}"
SCREEN_HEIGHT="${SCREEN_HEIGHT:-800}"
SCREEN_DEPTH="${SCREEN_DEPTH:-24}"
VNC_PORT="${VNC_PORT:-5900}"
NOVNC_PORT="${NOVNC_PORT:-6080}"
DESKTOP_API_PORT="${DESKTOP_API_PORT:-7090}"

mkdir -p \
  "$HOME/.vnc" \
  "$HOME/Desktop" \
  "$DESKTOP_WORKSPACE/screenshots" \
  "$XDG_RUNTIME_DIR" \
  /tmp/.X11-unix
chmod 700 "$XDG_RUNTIME_DIR"
chmod 1777 /tmp/.X11-unix

for f in "$HOME/Desktop"/*.desktop; do
  [[ -f "$f" ]] && chmod +x "$f" || true
done

PASS="${VNC_PASSWORD:0:8}"
if ! x11vnc -storepasswd "$PASS" "$HOME/.vnc/passwd" >/tmp/vnc_store.log 2>&1; then
  echo "[entrypoint] storepasswd failed; will use -passwd" >&2
  USE_PASSWD_ARG=1
else
  chmod 600 "$HOME/.vnc/passwd"
  USE_PASSWD_ARG=0
fi

if [[ -z "${DBUS_SESSION_BUS_ADDRESS:-}" ]] && command -v dbus-launch >/dev/null 2>&1; then
  eval "$(dbus-launch --sh-syntax)"
  export DBUS_SESSION_BUS_ADDRESS
fi

echo "[entrypoint] Starting Xvfb on $DISPLAY (${SCREEN_WIDTH}x${SCREEN_HEIGHT}x${SCREEN_DEPTH})"
Xvfb "$DISPLAY" -screen 0 "${SCREEN_WIDTH}x${SCREEN_HEIGHT}x${SCREEN_DEPTH}" \
  -ac +extension GLX +render -noreset &

for i in $(seq 1 50); do
  if xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then
    break
  fi
  sleep 0.1
done
if ! xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then
  echo "[entrypoint] Xvfb failed to start" >&2
  exit 1
fi

# Wallpaper immediately so VNC is never a blank/code-only root window
if command -v hsetroot >/dev/null 2>&1; then
  if [[ -f /opt/nitc-wallpaper/wallpaper.png ]]; then
    hsetroot -fill /opt/nitc-wallpaper/wallpaper.png || hsetroot -solid "#121c30" || true
  else
    hsetroot -solid "#121c30" || true
  fi
elif command -v xsetroot >/dev/null 2>&1; then
  xsetroot -solid "#121c30" || true
fi

echo "[entrypoint] Starting XFCE components (wm + settings + panel + desktop)"
# Explicit components — more reliable in containers than a full xfce4-session
# (avoids session-save dialogs and duplicate panels).
xfwm4 --replace >/tmp/xfwm4.log 2>&1 &
xfsettingsd --replace >/tmp/xfsettingsd.log 2>&1 &
sleep 0.3
xfce4-panel --disable-wm-check >/tmp/xfce4-panel.log 2>&1 &
xfdesktop --disable-wm-check >/tmp/xfdesktop.log 2>&1 &
sleep 1

echo "[entrypoint] Starting x11vnc on :${VNC_PORT}"
if [[ "${USE_PASSWD_ARG:-0}" == "1" ]]; then
  x11vnc \
    -display "$DISPLAY" \
    -rfbport "$VNC_PORT" \
    -passwd "$PASS" \
    -forever \
    -shared \
    -noxdamage \
    -repeat \
    -o /tmp/x11vnc.log &
else
  x11vnc \
    -display "$DISPLAY" \
    -rfbport "$VNC_PORT" \
    -rfbauth "$HOME/.vnc/passwd" \
    -forever \
    -shared \
    -noxdamage \
    -repeat \
    -o /tmp/x11vnc.log &
fi

echo "[entrypoint] Starting noVNC/websockify on :${NOVNC_PORT}"
NOVNC_WEB="${NOVNC_WEB:-/usr/share/novnc}"
if [[ ! -d "$NOVNC_WEB" ]]; then
  NOVNC_WEB="/opt/novnc"
fi
websockify --web="$NOVNC_WEB" "${NOVNC_PORT}" "localhost:${VNC_PORT}" &

echo "[entrypoint] Starting desktop-api on :${DESKTOP_API_PORT}"
cd /opt/desktop-api
exec uvicorn main:app --host 0.0.0.0 --port "${DESKTOP_API_PORT}"
