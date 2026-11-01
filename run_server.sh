#!/bin/bash
# 事前に uv sync --extra gpu（または cpu）で環境を作っておく。
# NOGI_RUNTIME=gpu|cpu を指定したときだけ，その runtime に入れ替えてから起動する。
cd "$(dirname "$0")"
if [[ -n "${NOGI_RUNTIME:-}" ]]; then
  uv sync --extra "$NOGI_RUNTIME" || exit 1
fi
exec uv run uvicorn inference_server:app --host 0.0.0.0 --port "${PORT:-18381}"
