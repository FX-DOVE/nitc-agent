#!/usr/bin/env bash
# Start Xvfb + polished XFCE desktop + x11vnc + noVNC + desktop-api
set -euo pipefail

export DISPLAY="${DISPLAY:-:99}"
export HOME="${HOME:-/home/desktop}"
export DESKTOP_WORKSPACE="${DESKTOP_WORKSPACE:-/home/desktop/workspace}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$HOME/.config}"
export XDG_DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-desktop}"
export GTK_THEME="${GTK_THEME:-Greybird}"
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
  "$HOME/.config/xfce4/panel" \
  "$DESKTOP_WORKSPACE/screenshots" \
  "$XDG_RUNTIME_DIR" \
  /tmp/.X11-unix
chmod 700 "$XDG_RUNTIME_DIR"
chmod 1777 /tmp/.X11-unix

# Ensure desktop launchers exist + trusted/executable (volume must not wipe them)
for name in Browser Terminal Files; do
  src=""
  case "$name" in
    Browser) src=/home/desktop/.config/xfce4/panel/launcher-3/browser.desktop ;;
    Terminal) src=/home/desktop/.config/xfce4/panel/launcher-4/terminal.desktop ;;
    Files) src=/home/desktop/.config/xfce4/panel/launcher-5/files.desktop ;;
  esac
  if [[ ! -f "$HOME/Desktop/${name}.desktop" && -f "$src" ]]; then
    cp "$src" "$HOME/Desktop/${name}.desktop"
  fi
done
for f in "$HOME/Desktop"/*.desktop; do
  [[ -f "$f" ]] || continue
  chmod +x "$f"
  # Mark trusted so xfdesktop shows icons as launchable (no "Untrusted" dialog)
  if command -v gio >/dev/null 2>&1; then
    gio set -t string "$f" metadata::xfce-exe-checksum "$(sha256sum "$f" | awk '{print $1}')" 2>/dev/null || true
    gio set -t string "$f" "metadata::trusted" "true" 2>/dev/null || true
  fi
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

# Soft wallpaper immediately (never blank/black root)
WALL="/opt/nitc-wallpaper/wallpaper.png"
if command -v hsetroot >/dev/null 2>&1; then
  if [[ -f "$WALL" ]]; then
    hsetroot -fill "$WALL" || hsetroot -solid "#e8ecf5" || true
  else
    hsetroot -solid "#e8ecf5" || true
  fi
elif command -v xsetroot >/dev/null 2>&1; then
  xsetroot -solid "#e8ecf5" || true
fi

echo "[entrypoint] Starting XFCE (wm + settings + panel + desktop icons)"
xfwm4 --replace >/tmp/xfwm4.log 2>&1 &
xfsettingsd --replace >/tmp/xfsettingsd.log 2>&1 &
sleep 0.4
# Apply theme via xfconf if available
if command -v xfconf-query >/dev/null 2>&1; then
  xfconf-query -c xsettings -p /Net/ThemeName -s Greybird 2>/dev/null || true
  xfconf-query -c xsettings -p /Net/IconThemeName -s Papirus 2>/dev/null || true
  xfconf-query -c xfwm4 -p /general/theme -s Greybird 2>/dev/null || true
  xfconf-query -c xfce4-desktop -p /backdrop/screen0/monitor0/workspace0/last-image -s "$WALL" 2>/dev/null || true
fi
xfce4-panel --disable-wm-check >/tmp/xfce4-panel.log 2>&1 &
xfdesktop --disable-wm-check >/tmp/xfdesktop.log 2>&1 &
sleep 1.5
# Re-apply wallpaper through xfdesktop once it is up
if [[ -f "$WALL" ]] && command -v xfdesktop >/dev/null 2>&1; then
  xfdesktop --reload 2>/dev/null || true
fi

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
