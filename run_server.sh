#!/bin/bash
cd "$(dirname "$0")"
exec uv run uvicorn inference_server:app --host 0.0.0.0 --port "${PORT:-18381}"
