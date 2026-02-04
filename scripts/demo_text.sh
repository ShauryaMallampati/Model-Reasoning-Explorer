#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export MRE_CONFIG="$ROOT_DIR/backend/mre_config.toml"

python3 -m mre_backend.cli demo-text --prompt "The capital of France is" --max-tokens 12
