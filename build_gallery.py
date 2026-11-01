"""登録画像から特徴量ファイル (gallery.npz) を作る。

    uv run python build_gallery.py
    uv run python build_gallery.py --eval   # 作成後に leave-one-out で精度を確認
"""
import argparse
import logging
import time

import numpy as np

from nogi_face_check import (
    DEFAULT_MIN_MARGIN,
    DEFAULT_MIN_SIM,
    Gallery,
    build_gallery,
    classify,
    load_face_app,
)


def evaluate(gallery: Gallery, min_sim: float, min_margin: float) -> None:
    """各登録画像を「自分以外のギャラリー」で判別し，正解率と類似度の分布を出す。"""
    correct = wrong = unknown = false_accept = 0
    genuine, impostor = [], []
    for i, (name, emb) in enumerate(zip(gallery.names, gallery.embeddings)):
        exclude = np.zeros(len(gallery), dtype=bool)
        exclude[i] = True
        scores = gallery.member_scores(emb, exclude=exclude)
        own = gallery.members.index(name)
        if np.isfinite(scores[own]):
            genuine.append(float(scores[own]))
        others = np.delete(scores, own)
        impostor.append(float(others.max()))
        r = classify(scores, gallery.members, min_sim, min_margin)
        if r["name"] == "Unknown":
            unknown += 1
        elif r["name"] == name:
            correct += 1
        else:
            wrong += 1
            print(f"  誤判別: {gallery.paths[i]} -> {r['name']} ({r['similarity']:.3f})")
        # 本人を丸ごと除いて「未登録の人物」として扱ったとき，誰かと判定してしまうか
        r_open = classify(others, [m for m in gallery.members if m != name], min_sim, min_margin)
        if r_open["name"] != "Unknown":
            false_accept += 1
    n = len(gallery)
    print(f"\nleave-one-out (min_sim={min_sim}, min_margin={min_margin}, {n} 枚)")
    print(f"  正解 {correct} ({correct / n:.1%}) / 誤判別 {wrong} ({wrong / n:.1%}) / Unknown {unknown} ({unknown / n:.1%})")
    print(f"  未登録扱いでの誤認 {false_accept} ({false_accept / n:.1%})")
    q = [5, 25, 50, 75, 95]
    print(f"  本人との類似度         p{q}: {np.round(np.percentile(genuine, q), 3).tolist()}")
    print(f"  他人の最大類似度       p{q}: {np.round(np.percentile(impostor, q), 3).tolist()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--known-dir", default="./nogi_member_face")
    parser.add_argument("--out", default="./gallery.npz")
    parser.add_argument("--eval", action="store_true", help="作成後に leave-one-out 評価を行う")
    parser.add_argument("--eval-only", action="store_true", help="既存の --out を読み込んで評価だけ行う")
    parser.add_argument("--min-sim", type=float, default=DEFAULT_MIN_SIM)
    parser.add_argument("--min-margin", type=float, default=DEFAULT_MIN_MARGIN)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.eval_only:
        gallery = Gallery.load(args.out)
    else:
        app = load_face_app()
        t = time.time()
        gallery = build_gallery(app, args.known_dir)
        gallery.save(args.out)
        print(f"{args.out}: {len(gallery.members)} 人 / {len(gallery)} 枚 ({time.time() - t:.1f}s)")
        for m, c in gallery.counts().items():
            print(f"  {m}: {c}")

    if args.eval or args.eval_only:
        evaluate(gallery, args.min_sim, args.min_margin)


if __name__ == "__main__":
    main()
