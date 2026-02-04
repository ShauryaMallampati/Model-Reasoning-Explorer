#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export MRE_CONFIG="$ROOT_DIR/backend/mre_config.toml"

python3 -m mre_backend.cli demo-vision --image "$ROOT_DIR/sample_data/images/sample.ppm"
