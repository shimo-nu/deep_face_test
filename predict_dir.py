"""フォルダ内の画像をまとめて判別し，BB を描いた画像を出力フォルダに保存する。

    uv run python predict_dir.py test_images -o out/test_images
    uv run python predict_dir.py test_images -o out/test_images -r --min-sim 0.45

- 入力フォルダの構成（サブフォルダ）は出力フォルダでも保たれる（-r 指定時）
- 出力フォルダには判別結果と推論時間をまとめた results.json も保存する
- 画像の親フォルダ名が登録メンバー名と一致する場合は，それを正解として正解率も出す
"""
import argparse
import json
import logging
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
from rich.console import Console
from rich.markup import escape
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from nogi_face_check import (
    DEFAULT_MIN_MARGIN,
    DEFAULT_MIN_SIM,
    DEFAULT_TEMPERATURE,
    MODEL_NAME,
    VALID_EXTS,
    Gallery,
    build_gallery,
    draw_results,
    identify,
    load_face_app,
    read_image,
)

console = Console(highlight=False)


def list_images(input_dir: Path, recursive: bool) -> list[Path]:
    it = input_dir.rglob("*") if recursive else input_dir.iterdir()
    return sorted(p for p in it
                  if p.is_file() and p.suffix.lower() in VALID_EXTS and not p.name.startswith("."))


def gpu_name() -> str | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10, check=True).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    return out.strip().splitlines()[0] if out.strip() else None


def print_device(app) -> str:
    """実際に使われている onnxruntime の provider（CUDA / CPU）を表示して返す。"""
    import onnxruntime as ort

    provider = app.det_model.session.get_providers()[0].replace("ExecutionProvider", "")
    if provider == "CUDA":
        name = gpu_name() or "不明な GPU"
        console.print(f"[bold cyan]\\[device][/bold cyan] [green]GPU (CUDA)[/green]  {escape(name)}")
    else:
        note = "  [yellow](GPU はあるが CUDA を使えず CPU にフォールバック。check_env.py で確認)[/yellow]" \
            if gpu_name() else "  [dim](GPU なし)[/dim]"
        console.print(f"[bold cyan]\\[device][/bold cyan] [yellow]CPU[/yellow]{note}")
    console.print(f"[bold cyan]\\[model][/bold cyan]  {MODEL_NAME}  onnxruntime {ort.__version__}")
    return provider


def load_gallery(app, gallery_path: Path, known_dir: Path) -> Gallery:
    if gallery_path.exists():
        return Gallery.load(gallery_path)
    console.print(f"[yellow]{escape(str(gallery_path))} が無いため {escape(str(known_dir))} から作成します[/yellow]")
    with console.status("ギャラリーを作成中..."):
        gallery = build_gallery(app, known_dir)
    gallery.save(gallery_path)
    return gallery


def save_image(path: Path, img) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() if path.suffix.lower() in VALID_EXTS else ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ValueError(f"{ext} へのエンコードに失敗しました")
    path.with_suffix(ext).write_bytes(buf.tobytes())


def ms_stats(values: list[float]) -> dict:
    if not values:
        return {}
    s = sorted(values)
    return {"mean": statistics.fmean(s), "median": statistics.median(s),
            "p95": s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))], "min": s[0], "max": s[-1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", type=Path, help="判別する画像が入ったフォルダ")
    parser.add_argument("-o", "--output-dir", type=Path, required=True, help="BB を描いた画像の出力先")
    parser.add_argument("-r", "--recursive", action="store_true", help="サブフォルダの画像も対象にする")
    parser.add_argument("--overwrite", action="store_true", help="空でない出力フォルダへの上書きを許可する")
    parser.add_argument("--gallery", type=Path, default=Path("./gallery.npz"))
    parser.add_argument("--known-dir", type=Path, default=Path("./nogi_member_face"))
    parser.add_argument("--min-sim", type=float, default=DEFAULT_MIN_SIM)
    parser.add_argument("--min-margin", type=float, default=DEFAULT_MIN_MARGIN)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("-q", "--quiet", action="store_true", help="画像ごとの結果を表示しない")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    input_dir, output_dir = args.input_dir.resolve(), args.output_dir.resolve()
    if not input_dir.is_dir():
        console.print(f"[red]入力フォルダがありません: {escape(str(input_dir))}[/red]")
        sys.exit(1)
    if output_dir == input_dir or (args.recursive and input_dir in output_dir.parents):
        console.print("[red]出力フォルダは入力フォルダ（とその中）以外を指定してください[/red]")
        sys.exit(1)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        console.print(f"[red]出力フォルダが空ではありません: {escape(str(output_dir))}  (--overwrite で上書き)[/red]")
        sys.exit(1)
    images = list_images(input_dir, args.recursive)
    if not images:
        console.print(f"[red]画像が見つかりません: {escape(str(input_dir))}[/red]")
        sys.exit(1)

    console.rule("[bold]フォルダ判別テスト")
    with console.status("モデルを読み込み中..."):
        t = time.perf_counter()
        app = load_face_app()
        load_s = time.perf_counter() - t
    provider = print_device(app)
    gallery = load_gallery(app, args.gallery, args.known_dir)
    console.print(f"[bold cyan]\\[gallery][/bold cyan] {len(gallery.members)} 人 / {len(gallery)} 枚  "
                  f"[dim]min_sim={args.min_sim} min_margin={args.min_margin}[/dim]")
    console.print(f"[bold cyan]\\[input][/bold cyan]  {escape(str(input_dir))}  ({len(images)} 枚)")
    console.print(f"[bold cyan]\\[output][/bold cyan] {escape(str(output_dir))}")

    # 初回推論は CUDA の初期化などで遅いので，計測前に 1 回流しておく
    with console.status("ウォームアップ中..."):
        try:
            app.get(read_image(images[0]))
        except ValueError:
            pass

    output_dir.mkdir(parents=True, exist_ok=True)
    records, read_ms, infer_ms, total_ms = [], [], [], []
    name_counts = Counter()
    labeled = correct = 0
    t_all = time.perf_counter()

    columns = (SpinnerColumn(), TextColumn("[cyan]判別中"), BarColumn(), MofNCompleteColumn(),
               TimeElapsedColumn(), TextColumn("残り"), TimeRemainingColumn(),
               TextColumn("[dim]{task.fields[current]}"))
    with Progress(*columns, console=console) as progress:
        task = progress.add_task("predict", total=len(images), current="")
        for path in images:
            rel = path.relative_to(input_dir)
            progress.update(task, current=escape(str(rel)))
            t0 = time.perf_counter()
            try:
                img = read_image(path)
            except ValueError:
                progress.console.print(f"[red]読み込み失敗[/red] {escape(str(rel))}")
                records.append({"file": str(rel), "error": "decode failed"})
                progress.advance(task)
                continue
            t1 = time.perf_counter()
            results = identify(app, gallery, img, min_sim=args.min_sim, min_margin=args.min_margin,
                               temperature=args.temperature, topk=args.topk)
            t2 = time.perf_counter()
            save_image(output_dir / rel, draw_results(img, results))
            t3 = time.perf_counter()

            read_ms.append((t1 - t0) * 1000)
            infer_ms.append((t2 - t1) * 1000)
            total_ms.append((t3 - t0) * 1000)
            names = [r["name"] for r in results]
            name_counts.update(names)

            # 親フォルダ名が登録メンバーなら正解ラベルとして使う
            truth = path.parent.name if path.parent.name in gallery.members else None
            hit = None
            if truth:
                labeled += 1
                hit = truth in names
                correct += hit
            records.append({"file": str(rel), "size": [img.shape[1], img.shape[0]], "truth": truth,
                            "correct": hit, "infer_ms": round(infer_ms[-1], 2), "faces": results})

            if not args.quiet:
                label = ", ".join(
                    f"[green]{escape(r['name'])}[/green] {r['similarity']:.2f}" if r["name"] != "Unknown"
                    else f"[yellow]Unknown[/yellow] [dim](? {escape(r['candidates'][0]['name'])} "
                         f"{r['candidates'][0]['similarity']:.2f})[/dim]"
                    for r in results) or "[dim]顔なし[/dim]"
                mark = "" if hit is None else (" [green]✓[/green]" if hit else " [red]✗[/red]")
                progress.console.print(f"{escape(str(rel))}  [dim]{infer_ms[-1]:6.1f} ms[/dim]  {label}{mark}")
            progress.advance(task)
    wall_s = time.perf_counter() - t_all

    # ---------- まとめ ----------
    console.rule("[bold]結果")
    n_ok = len(infer_ms)
    n_faces = sum(name_counts.values())
    summary = Table.grid(padding=(0, 2))
    summary.add_row("device", f"{'GPU (CUDA)' if provider == 'CUDA' else 'CPU'}")
    summary.add_row("画像", f"{n_ok} 枚 (失敗 {len(images) - n_ok})")
    summary.add_row("顔", f"{n_faces} 個 (判別 {n_faces - name_counts['Unknown']} / Unknown {name_counts['Unknown']})")
    if labeled:
        summary.add_row("正解率", f"{correct}/{labeled} = {correct / labeled:.1%}  [dim](親フォルダ名を正解として集計)[/dim]")
    summary.add_row("処理時間", f"{wall_s:.1f}s  ({n_ok / wall_s:.2f} 枚/s)  [dim]モデル読み込み {load_s:.1f}s は除く[/dim]")
    console.print(summary)

    speed = Table(title="1 枚あたりの時間 (ms)", title_justify="left")
    speed.add_column("")
    for col in ("mean", "median", "p95", "min", "max"):
        speed.add_column(col, justify="right")
    for label, vals in (("推論 (検出+特徴量+照合)", infer_ms), ("読み込み", read_ms), ("合計 (保存まで)", total_ms)):
        st = ms_stats(vals)
        if st:
            speed.add_row(label, *(f"{st[k]:.1f}" for k in ("mean", "median", "p95", "min", "max")))
    console.print(speed)

    if name_counts:
        people = Table(title="判別された人物", title_justify="left")
        people.add_column("name")
        people.add_column("faces", justify="right")
        for name, c in name_counts.most_common():
            people.add_row(f"[yellow]{name}[/yellow]" if name == "Unknown" else name, str(c))
        console.print(people)

    report = {
        "input_dir": str(input_dir), "output_dir": str(output_dir), "device": provider,
        "params": {"min_sim": args.min_sim, "min_margin": args.min_margin,
                   "temperature": args.temperature, "topk": args.topk},
        "summary": {"images": n_ok, "failed": len(images) - n_ok, "faces": n_faces,
                    "unknown": name_counts["Unknown"], "labeled": labeled, "correct": correct,
                    "wall_s": round(wall_s, 3), "model_load_s": round(load_s, 3),
                    "infer_ms": {k: round(v, 2) for k, v in ms_stats(infer_ms).items()},
                    "total_ms": {k: round(v, 2) for k, v in ms_stats(total_ms).items()}},
        "images": records,
    }
    (output_dir / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    console.print(f"BB 付き画像と results.json を保存しました: {escape(str(output_dir))}")


if __name__ == "__main__":
    main()
