#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec docker run --rm --platform linux/amd64 \
  --user root \
  -e RECORD=1 \
  -v "${ROOT}:/workspace" \
  docker.io/johnsloe/torcs-competition:amd64 \
  bash /workspace/.github/scripts/run_torcs.sh
