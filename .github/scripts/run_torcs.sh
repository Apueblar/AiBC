#!/usr/bin/env bash
set -e

rm -f /etc/apt/sources.list.d/*
grep -v "packages.microsoft.com" /etc/apt/sources.list > /tmp/sources.list.tmp && mv /tmp/sources.list.tmp /etc/apt/sources.list || true

apt-get update -q
apt-get install -y -q xvfb python3-pip wmctrl xautomation libgl1-mesa-dri

pip3 install numpy python3-xlib --quiet

Xvfb :1 -screen 0 1024x768x24 -ac &
sleep 2

export LIBGL_ALWAYS_SOFTWARE=1

cd /workspace/gym_torcs
python3 run.py --episodes 1 --steps 10000
