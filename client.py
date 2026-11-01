"""推論サーバーに画像を送り，判別結果の JSON と BB 付き画像を保存する。

    uv run python client.py -i photo.jpg
    uv run python client.py -i a.jpg b.jpg --url http://192.168.207.227:18381 --out out
"""
import argparse
import base64
import json
from datetime import datetime
from pathlib import Path

import requests

DEFAULT_URL = "http://192.168.207.227:18381"


def predict(url: str, image_path: Path, timeout: int = 60, **params) -> dict:
    with open(image_path, "rb") as f:
        files = {"file": (image_path.name, f, "application/octet-stream")}
        r = requests.post(f"{url.rstrip('/')}/predict",
                          params={"return_image": True, **params}, files=files, timeout=timeout)
    r.raise_for_status()
    return r.json()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--image_path", nargs="+", required=True, type=Path)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--out", default="out", type=Path)
    parser.add_argument("--min-sim", type=float)
    parser.add_argument("--min-margin", type=float)
    parser.add_argument("--topk", type=int)
    args = parser.parse_args()

    params = {k: v for k, v in {"min_sim": args.min_sim, "min_margin": args.min_margin,
                                "topk": args.topk}.items() if v is not None}
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    for path in args.image_path:
        res = predict(args.url, path, **params)
        image_b64 = res.pop("image_base64")
        stem = args.out / f"{stamp}_{path.stem}"
        stem.with_suffix(".jpg").write_bytes(base64.b64decode(image_b64))
        stem.with_suffix(".json").write_text(json.dumps(res, ensure_ascii=False, indent=2))

        print(f"{path}: {res['num_faces']} faces -> {stem}.jpg / .json")
        for f in res["faces"]:
            cands = ", ".join(f"{c['name']} {c['similarity']:.2f}" for c in f["candidates"])
            print(f"  {f['name']:<22} sim={f['similarity']:.3f} prob={f['prob']:.3f} bbox={f['bbox']}  [{cands}]")


if __name__ == "__main__":
    main()
