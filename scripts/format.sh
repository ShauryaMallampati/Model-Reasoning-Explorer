#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

cd "$ROOT_DIR/backend"
python3 -m ruff format .
python3 -m black .

cd "$ROOT_DIR/frontend"
npm run format
