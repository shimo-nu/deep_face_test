"""推論サーバーを動かせる環境かを順番に確認する。

    uv sync --extra gpu   # 最初に 1 回（CPU のみなら --extra cpu）
    uv run python check_env.py
    uv run python check_env.py --image photo.jpg
"""
import argparse
import importlib.metadata as md
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.table import Table

console = Console(highlight=False)

OK, WARN, NG, SKIP = "ok", "warn", "ng", "skip"
STATUS_TEXT = {
    OK: "[green]使用可.[/green]",
    WARN: "[yellow]注意.[/yellow]",
    NG: "[red]使用不可.[/red]",
    SKIP: "[dim]スキップ.[/dim]",
}
LABEL_WIDTH = len("[onnxruntime]")
STATUS_WIDTH = 10


def _version(dist: str) -> str | None:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def _major_minor(v: str | None) -> tuple[int, int] | None:
    m = re.match(r"(\d+)\.(\d+)", v or "")
    return (int(m[1]), int(m[2])) if m else None


def report(name: str, status: str, info: dict | None = None, notes: list[str] | None = None) -> None:
    """[name] 状態  key : value ... を 1 行で出し，補足は下に字下げして出す。"""
    grid = Table.grid(padding=(0, 1))
    grid.add_column(width=LABEL_WIDTH, no_wrap=True)
    grid.add_column(width=STATUS_WIDTH, no_wrap=True)
    grid.add_column(overflow="fold")
    fields = "  ".join(f"[dim]{escape(k)} :[/dim] {escape(str(v))}" for k, v in (info or {}).items())
    grid.add_row(f"[bold cyan]{escape(f'[{name}]')}[/bold cyan]", STATUS_TEXT[status], fields)
    for n in notes or []:
        grid.add_row("", "", f"[dim]└[/dim] {escape(n)}")
    console.print(grid)


# ---------- 各チェック ----------
def check_python(ctx):
    v = sys.version_info
    status = OK if (3, 12) <= v[:2] < (3, 13) else NG
    notes = [] if status == OK else ["pyproject.toml は Python 3.12 を要求しています"]
    return status, {"version": sys.version.split()[0], "executable": sys.executable}, notes


def check_packages(ctx):
    names = ["insightface", "onnxruntime", "onnxruntime-gpu", "opencv-python-headless",
             "opencv-python", "numpy", "fastapi", "uvicorn"]
    vers = {n: _version(n) for n in names}
    info = {n: v for n, v in vers.items() if v}
    notes, status = [], OK
    missing = [n for n in ("insightface", "opencv-python-headless", "numpy", "fastapi", "uvicorn") if not vers[n]]
    if missing:
        status = NG
        notes.append(f"未インストール: {', '.join(missing)}  (uv sync --extra gpu|cpu)")
    if vers["onnxruntime"] and vers["onnxruntime-gpu"]:
        status = NG
        notes.append("onnxruntime と onnxruntime-gpu が両方入っています。"
                     "uv sync --extra gpu --reinstall-package onnxruntime-gpu で入れ直してください")
    elif not (vers["onnxruntime"] or vers["onnxruntime-gpu"]):
        status = NG
        notes.append("onnxruntime が入っていません。uv sync --extra gpu（または --extra cpu）を実行してください")
    if vers["opencv-python"]:
        status = max(status, WARN, key=[OK, WARN, NG].index)
        notes.append("opencv-python と opencv-python-headless が両方入っています")
    ctx["ort_gpu_pkg"] = bool(vers["onnxruntime-gpu"])
    return status, info, notes


def _libcuda_error() -> str | None:
    """NVIDIA ドライバーの libcuda.so.1 が読み込めるか（onnxruntime-gpu はこれが無いと CPU に落ちる）。"""
    import ctypes

    try:
        ctypes.CDLL("libcuda.so.1")
    except OSError as e:
        return str(e)
    return None


def check_gpu(ctx):
    ctx["libcuda_error"] = _libcuda_error()
    if not shutil.which("nvidia-smi"):
        ctx["driver_cuda"] = None
        if ctx.get("ort_gpu_pkg"):
            return NG, {"nvidia-smi": "見つかりません"}, ["NVIDIA ドライバーが見つかりません。GPU を使うにはドライバーが必要です"]
        return WARN, {"nvidia-smi": "見つかりません"}, ["NVIDIA GPU / ドライバーが見つからないため CPU で動作します"]
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout.strip().splitlines()
        header = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=20).stdout
    except (subprocess.SubprocessError, OSError) as e:
        ctx["driver_cuda"] = None
        return NG, {"nvidia-smi": "実行失敗"}, [str(e)]
    m = re.search(r"CUDA Version:\s*([\d.]+)", header)
    ctx["driver_cuda"] = m[1] if m else None
    name, driver, mem = (s.strip() for s in q[0].split(","))
    info = {"nvidia version": driver, "cuda version": ctx["driver_cuda"] or "不明", "gpu": name, "memory": mem}
    if len(q) > 1:
        info["gpus"] = len(q)
    if ctx["libcuda_error"]:
        return NG, info, [f"libcuda.so.1 を読み込めません: {ctx['libcuda_error']}"]
    return OK, info, []


def check_onnxruntime(ctx):
    try:
        import onnxruntime as ort
    except ImportError as e:
        return NG, {}, [str(e)]
    providers = ort.get_available_providers()
    info = {"version": ort.__version__, "device": ort.get_device(),
            "providers": ", ".join(p.replace("ExecutionProvider", "") for p in providers)}
    notes = []
    if not ctx.get("ort_gpu_pkg"):
        ctx["expect_cuda"] = False
        notes.append("CPU 版 onnxruntime を使用しています")
        return OK, info, notes

    ctx["expect_cuda"] = True
    rt_cuda = _version("nvidia-cuda-runtime") or _version("nvidia-cuda-runtime-cu12")
    cudnn = _version("nvidia-cudnn-cu13") or _version("nvidia-cudnn-cu12")
    info["cuda runtime (pip)"] = rt_cuda or "なし"
    info["cudnn (pip)"] = cudnn or "なし"
    if "CUDAExecutionProvider" not in providers:
        return NG, info, ["onnxruntime-gpu に CUDAExecutionProvider がありません"]
    status = OK
    if ctx.get("libcuda_error"):
        status = NG
        notes.append("NVIDIA ドライバー (libcuda.so.1) が無いため CUDA を使えません")
    need, have = _major_minor(rt_cuda), _major_minor(ctx.get("driver_cuda"))
    if need and have and have[0] < need[0]:
        status = NG
        notes.append(f"ドライバーが対応する CUDA {ctx['driver_cuda']} < 必要な CUDA {need[0]}.x 。NVIDIA ドライバーを更新してください")
    if not rt_cuda or not cudnn:
        status = WARN
        notes.append("pip の CUDA / cuDNN が見つかりません。システムの CUDA が使われます")
    try:
        ort.preload_dlls()
    except Exception as e:  # noqa: BLE001 - 何が起きても結果として表示する
        status = NG
        notes.append(f"CUDA / cuDNN の読み込みに失敗: {e}")
    return status, info, notes


def check_model(ctx):
    from nogi_face_check import MODEL_NAME, load_face_app

    model_dir = Path("~/.insightface/models").expanduser() / MODEL_NAME
    downloaded = model_dir.is_dir() and any(model_dir.glob("*.onnx"))
    if not downloaded:
        console.print(f"[dim]{MODEL_NAME} をダウンロードします (約 280MB)[/dim]")
    t = time.perf_counter()
    try:
        # onnxruntime が C++ 側から stderr に出すエラーログを退避し，失敗時の原因表示に使う
        with _capture_stderr() as captured:
            app = load_face_app()
    except Exception as e:  # noqa: BLE001
        return NG, {"model": MODEL_NAME}, [f"読み込み失敗: {e}"]
    elapsed = time.perf_counter() - t
    ctx["face_app"] = app
    used = app.det_model.session.get_providers()[0].replace("ExecutionProvider", "")
    ctx["used_provider"] = used
    info = {"model": MODEL_NAME, "dir": str(model_dir), "provider": used, "load": f"{elapsed:.1f}s"}
    if ctx.get("expect_cuda") and used != "CUDA":
        notes = ["onnxruntime-gpu なのに CPU にフォールバックしています（CUDA / cuDNN / ドライバーを確認）"]
        reasons = re.findall(r"with error: (.+)", captured.read())
        notes += sorted(set(reasons))[:3]
        return NG, info, notes
    return OK, info, []


class _capture_stderr:
    """fd 2 を一時ファイルに付け替え，C 拡張が出すログも含めて捕まえる。"""

    def __enter__(self):
        import tempfile

        sys.stderr.flush()
        self._file = tempfile.TemporaryFile(mode="w+")
        self._saved = os.dup(2)
        os.dup2(self._file.fileno(), 2)
        return self

    def __exit__(self, *exc):
        sys.stderr.flush()
        os.dup2(self._saved, 2)
        os.close(self._saved)
        return False

    def read(self) -> str:
        self._file.seek(0)
        return self._file.read()


def _sample_image(args):
    if args.image:
        return Path(args.image)
    known = Path(args.known_dir)
    if known.is_dir():
        from nogi_face_check import collect_known_images

        for paths in collect_known_images(known).values():
            return paths[0]
    import insightface

    return Path(insightface.__file__).parent / "data" / "images" / "t1.jpg"


def check_inference(ctx, args):
    if "face_app" not in ctx:
        return SKIP, {}, ["モデルが読み込めていません"]
    from nogi_face_check import read_image

    path = _sample_image(args)
    try:
        img = read_image(path)
    except (OSError, ValueError) as e:
        return NG, {"image": str(path)}, [f"画像を読めません: {e}"]
    app = ctx["face_app"]
    faces = app.get(img)  # ウォームアップ（GPU は初回が遅い）
    t = time.perf_counter()
    for _ in range(args.runs):
        faces = app.get(img)
    ms = (time.perf_counter() - t) / args.runs * 1000
    ctx["sample"] = (path, img)
    info = {"image": path.name, "size": f"{img.shape[1]}x{img.shape[0]}", "faces": len(faces),
            "time": f"{ms:.1f} ms/枚 ({ctx['used_provider']}, {args.runs} 回平均)"}
    if not faces:
        return WARN, info, ["顔が検出されませんでした（画像を --image で指定してください）"]
    if faces[0].normed_embedding is None:
        return NG, info, ["特徴量が計算されていません"]
    return OK, info, []


def check_gallery(ctx, args):
    from nogi_face_check import Gallery, collect_known_images

    known = Path(args.known_dir)
    people = collect_known_images(known) if known.is_dir() else {}
    n_imgs = sum(len(v) for v in people.values())
    path = Path(args.gallery)
    if not path.exists():
        status = WARN if people else NG
        notes = [f"{path} がありません。サーバー起動時に {known} から作成されます"
                 if people else f"{known} に登録画像がありません"]
        return status, {"known_dir": f"{len(people)} 人 / {n_imgs} 枚"}, notes
    try:
        gallery = Gallery.load(path)
    except Exception as e:  # noqa: BLE001
        return NG, {"gallery": str(path)}, [f"読み込み失敗: {e}  (python build_gallery.py で作り直してください)"]
    ctx["gallery"] = gallery
    info = {"gallery": f"{len(gallery.members)} 人 / {len(gallery)} 枚",
            "known_dir": f"{len(people)} 人 / {n_imgs} 枚"}
    notes = []
    newer = [p for ps in people.values() for p in ps if p.stat().st_mtime > path.stat().st_mtime]
    missing = sorted(set(people) - set(gallery.members))
    if missing:
        notes.append(f"gallery に未登録のメンバー: {', '.join(missing)}")
    if newer:
        notes.append(f"gallery より新しい登録画像が {len(newer)} 枚あります")
    if notes:
        notes.append("python build_gallery.py か POST /gallery/reload?rebuild=true で作り直してください")
    return (WARN if notes else OK), info, notes


def check_identify(ctx, args):
    if "gallery" not in ctx or "sample" not in ctx:
        return SKIP, {}, ["gallery または推論テストが未完了です"]
    from nogi_face_check import identify

    path, img = ctx["sample"]
    results = identify(ctx["face_app"], ctx["gallery"], img)
    if not results:
        return WARN, {"image": path.name}, ["顔が検出されませんでした"]
    r = results[0]
    return OK, {"image": path.name, "name": r["name"], "similarity": f"{r['similarity']:.3f}",
                "prob": f"{r['prob']:.3f}"}, []


def check_port(ctx, args):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("0.0.0.0", args.port))
        except OSError:
            return WARN, {"port": args.port}, ["使用中です（サーバーが既に起動している可能性があります）"]
    return OK, {"port": args.port}, []


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--known-dir", default=os.environ.get("NOGI_KNOWN_DIR", "./nogi_member_face"))
    parser.add_argument("--gallery", default=os.environ.get("NOGI_GALLERY", "./gallery.npz"))
    parser.add_argument("--image", help="推論テストに使う画像（既定: 登録画像の 1 枚目）")
    parser.add_argument("--runs", type=int, default=5, help="推論時間を測る回数")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 18381)))
    args = parser.parse_args()

    checks = [
        ("python", lambda c: check_python(c)),
        ("packages", lambda c: check_packages(c)),
        ("gpu", lambda c: check_gpu(c)),
        ("onnxruntime", lambda c: check_onnxruntime(c)),
        ("model", lambda c: check_model(c)),
        ("inference", lambda c: check_inference(c, args)),
        ("gallery", lambda c: check_gallery(c, args)),
        ("identify", lambda c: check_identify(c, args)),
        ("port", lambda c: check_port(c, args)),
    ]
    console.rule("[bold]nogi-face 環境チェック")
    ctx, results = {}, {}
    for name, fn in checks:
        with console.status(f"[cyan]{name}[/cyan] を確認中..."):
            try:
                status, info, notes = fn(ctx)
            except Exception as e:  # noqa: BLE001 - 想定外の失敗も 1 行で報告して続ける
                status, info, notes = NG, {}, [f"{type(e).__name__}: {e}"]
        results[name] = status
        report(name, status, info, notes)

    console.rule()
    ng = [n for n, s in results.items() if s == NG]
    warn = [n for n, s in results.items() if s == WARN]
    if ng:
        console.print(f"[bold red]使用不可[/bold red]: {', '.join(ng)} を確認してください")
    else:
        device = ctx.get("used_provider", "?")
        tail = f"（注意: {', '.join(warn)}）" if warn else ""
        console.print(f"[bold green]使用可[/bold green]: {device} で推論できます{tail}")
    sys.exit(1 if ng else 0)


if __name__ == "__main__":
    main()
