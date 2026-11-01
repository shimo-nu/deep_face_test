#!/bin/bash
# 事前に ./setup_env.sh gpu（または cpu）で環境を作っておく。
# NOGI_RUNTIME=gpu|cpu を指定したときだけ，その runtime に入れ替えてから起動する。
cd "$(dirname "$0")"
if [[ -n "${NOGI_RUNTIME:-}" ]]; then
  ./setup_env.sh "$NOGI_RUNTIME" || exit 1
fi
exec uv run uvicorn inference_server:app --host 0.0.0.0 --port "${PORT:-18381}"
