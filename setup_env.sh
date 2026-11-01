#!/bin/bash
# 使う onnxruntime を選んで環境を作る / 切り替える:  ./setup_env.sh gpu|gpu-cu12|cpu
#   gpu      : CUDA 13 版（NVIDIA ドライバー 580 以上）
#   gpu-cu12 : CUDA 12 版（nvidia-smi の CUDA Version が 12.x のマシン）
#
# onnxruntime と onnxruntime-gpu は同じ onnxruntime/ ディレクトリにファイルを置くため，
# 両方入った状態から片方を削除すると残った方のファイルまで消えて壊れる。
# --reinstall-package で選んだ方を入れ直し，壊れていても必ず元に戻す。
set -euo pipefail
cd "$(dirname "$0")"
case "${1:-}" in
  gpu|gpu-cu12) pkg=onnxruntime-gpu ;;
  cpu) pkg=onnxruntime ;;
  *) echo "usage: $0 gpu|gpu-cu12|cpu" >&2; exit 1 ;;
esac
uv sync --extra "$1" --reinstall-package "$pkg"
uv run python -c "import onnxruntime as o; print('onnxruntime', o.__version__, o.get_available_providers())"
