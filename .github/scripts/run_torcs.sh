#!/usr/bin/env bash
set -e

RECORD="${RECORD:-0}"
FFMPEG_PID=""

cleanup() {
  if [ -n "$FFMPEG_PID" ] && kill -0 "$FFMPEG_PID" 2>/dev/null; then
    kill -INT "$FFMPEG_PID" 2>/dev/null || true
    wait "$FFMPEG_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

rm -f /etc/apt/sources.list.d/*
grep -v "packages.microsoft.com" /etc/apt/sources.list > /tmp/sources.list.tmp && mv /tmp/sources.list.tmp /etc/apt/sources.list || true

apt-get update -q
apt-get install -y -q xvfb python3-pip wmctrl xautomation libgl1-mesa-dri

if [ "$RECORD" = "1" ]; then
  apt-get install -y -q ffmpeg
fi

pip3 install numpy python3-xlib --quiet

Xvfb :1 -screen 0 1024x768x24 -ac &
sleep 2

export LIBGL_ALWAYS_SOFTWARE=1
export DISPLAY=:1

if [ "$RECORD" = "1" ]; then
  mkdir -p /workspace/results
  OUT="/workspace/results/torcs_run_$(date -u +%Y%m%d_%H%M%S).mp4"
  ffmpeg -y -nostdin -video_size 1024x768 -framerate 12 -f x11grab -draw_mouse 0 -i :1.0+0,0 \
    -c:v libx264 -preset veryfast -crf 26 -pix_fmt yuv420p "$OUT" &
  FFMPEG_PID=$!
  sleep 1
fi

cd /workspace/gym_torcs
python3 run.py --episodes 1 --steps 10000
