import requests
import json
from pathlib import Path
import numpy as np
import cv2
import argparse
from datetime import datetime
from typing import Any, Dict, List, Tuple
import numpy as np
import requests
import cv2
from pathlib import Path

def draw_results_on_image(
    rgb_img: np.ndarray,
    results: List[Dict[str, Any]],
    box_color: Tuple[int, int, int] = (0, 255, 0),
    text_scale: float = 0.7,
    text_thickness: int = 2
) -> np.ndarray:
    vis = rgb_img.copy()
    for r in results:
        top, right, bottom, left = r["box"]
        top, right, bottom, left = int(top), int(right), int(bottom), int(left)

        name = r.get("name", "Unknown")
        sim = r.get("similarity", 0.0)
        cands = r.get("candidates", [])

        cv2.rectangle(vis, (left, top), (right, bottom), box_color, 2)

        label = f"{name} ({sim * 100:.1f}%)"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)
        y1 = max(0, top - th - 6)
        cv2.rectangle(vis, (left, y1), (left + tw + 6, y1 + th + 6), (0, 0, 0), -1)
        cv2.putText(vis, label, (left + 3, y1 + th + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 0), text_thickness)

        if name == "Unknown" and cands:
            offset = y1 + th + 8
            for cand_name, score in cands[:3]:
                txt = f"  → {cand_name} ({score * 100:.1f}%)"
                cv2.putText(vis, txt, (left + 3, offset),
                            cv2.FONT_HERSHEY_SIMPLEX, text_scale * 0.7, (0, 200, 255), 1)
                offset += int(th * 0.9)
    return vis


def load_image_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_rgb(path: str, rgb_img: np.ndarray) -> None:
    bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, bgr)


def call_api_and_save(
    api_url: str,
    image_path: str,
    output_path: str,
    topk_candidates: int = 3,
    timeout: int = 60
) -> None:
    rgb = load_image_rgb(image_path)
    params = {"topk_candidates": topk_candidates, "num_jitters_detect": 1}

    with open(image_path, "rb") as f:
        files = {"file": (Path(image_path).name, f, "image/jpeg")}
        r = requests.post(api_url, params=params, files=files, timeout=timeout)

    r.raise_for_status()
    response = r.json()

    results: List[Dict[str, Any]] = response.get("results", [])
    for rr in results:
        if "box" in rr:
            rr["box"] = tuple(int(x) for x in rr["box"])

    rgb_img_for_viz = draw_results_on_image(rgb, results)
    save_rgb(output_path, rgb_img_for_viz)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_path')
    args = parser.parse_args()

    date_now = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    print(date_now)
    call_api_and_save(
        api_url="http://192.168.207.227:18381/predict",
        image_path=args.image_path,
        output_path=f"out/{date_now}.jpg",
    )

