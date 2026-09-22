#!/usr/bin/env bash
# Start Xvfb + lightweight DE + x11vnc + noVNC + desktop-api
set -euo pipefail

export DISPLAY="${DISPLAY:-:99}"
export HOME="${HOME:-/home/desktop}"
export DESKTOP_WORKSPACE="${DESKTOP_WORKSPACE:-/home/desktop/workspace}"
VNC_PASSWORD="${VNC_PASSWORD:-nitc}"
SCREEN_WIDTH="${SCREEN_WIDTH:-1280}"
SCREEN_HEIGHT="${SCREEN_HEIGHT:-800}"
SCREEN_DEPTH="${SCREEN_DEPTH:-24}"
VNC_PORT="${VNC_PORT:-5900}"
NOVNC_PORT="${NOVNC_PORT:-6080}"
DESKTOP_API_PORT="${DESKTOP_API_PORT:-7090}"

mkdir -p "$HOME/.vnc" "$DESKTOP_WORKSPACE/screenshots" /tmp/.X11-unix
chmod 1777 /tmp/.X11-unix

# VNC password (classic VNC truncates to 8 chars)
mkdir -p "$HOME/.vnc"
PASS="${VNC_PASSWORD:0:8}"
if ! x11vnc -storepasswd "$PASS" "$HOME/.vnc/passwd" >/tmp/vnc_store.log 2>&1; then
  # Fallback: passwd on cmdline for x11vnc below
  echo "[entrypoint] storepasswd failed; will use -passwd" >&2
  USE_PASSWD_ARG=1
else
  chmod 600 "$HOME/.vnc/passwd"
  USE_PASSWD_ARG=0
fi

echo "[entrypoint] Starting Xvfb on $DISPLAY (${SCREEN_WIDTH}x${SCREEN_HEIGHT}x${SCREEN_DEPTH})"
Xvfb "$DISPLAY" -screen 0 "${SCREEN_WIDTH}x${SCREEN_HEIGHT}x${SCREEN_DEPTH}" -ac +extension GLX +render -noreset &
XVFB_PID=$!

# Wait for X
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

echo "[entrypoint] Starting openbox + pcmanfm"
openbox &
# Desktop icons / wallpaper-ish background
pcmanfm --desktop --profile default >/tmp/pcmanfm.log 2>&1 &
# Simple panel optional — skip if tint2 missing
if command -v tint2 >/dev/null 2>&1; then
  tint2 >/tmp/tint2.log 2>&1 &
fi

echo "[entrypoint] Starting x11vnc on :${VNC_PORT}"
if [[ "${USE_PASSWD_ARG:-0}" == "1" ]]; then
  x11vnc     -display "$DISPLAY"     -rfbport "$VNC_PORT"     -passwd "$PASS"     -forever     -shared     -noxdamage     -repeat     -o /tmp/x11vnc.log     &
else
  x11vnc     -display "$DISPLAY"     -rfbport "$VNC_PORT"     -rfbauth "$HOME/.vnc/passwd"     -forever     -shared     -noxdamage     -repeat     -o /tmp/x11vnc.log     &
fi

echo "[entrypoint] Starting noVNC/websockify on :${NOVNC_PORT}"
# websockify ships with novnc package or standalone
NOVNC_WEB="${NOVNC_WEB:-/usr/share/novnc}"
if [[ ! -d "$NOVNC_WEB" ]]; then
  NOVNC_WEB="/opt/novnc"
fi
websockify --web="$NOVNC_WEB" "${NOVNC_PORT}" "localhost:${VNC_PORT}" &

echo "[entrypoint] Starting desktop-api on :${DESKTOP_API_PORT}"
cd /opt/desktop-api
exec uvicorn main:app --host 0.0.0.0 --port "${DESKTOP_API_PORT}"
