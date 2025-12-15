#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=.
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir backend --reload-dir config
