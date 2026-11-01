"""InsightFace (SCRFD + ArcFace) による登録メンバーの顔判別。

- 検出: SCRFD (buffalo_l の det_10g)
- 特徴量: ArcFace (buffalo_l の w600k_r50, 512 次元)
- 照合: 登録画像 1 枚ごとの特徴量とのコサイン類似度を NumPy で計算し，
  メンバーごとの最大値をそのメンバーのスコアとする
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)
# insightface の顔位置合わせが毎回出す scikit-image の非推奨警告を抑える
warnings.filterwarnings("ignore", message="`estimate` is deprecated", category=FutureWarning)

MODEL_NAME = "buffalo_l"
VALID_EXTS = {".jpg", ".jpeg", ".png"}
# 登録メンバーとして扱わないディレクトリ
EXCLUDE_DIRS = {"preset_images_dir", "multipleFaces", "noFace"}

DEFAULT_MIN_SIM = 0.50
DEFAULT_MIN_MARGIN = 0.05
DEFAULT_TEMPERATURE = 0.05


# ---------- モデル ----------
def load_face_app(det_size=None, det_thresh: float = 0.5, providers=None):
    """検出と認識だけを読み込んだ FaceAnalysis を返す（初回はモデルを自動ダウンロード）。

    providers が None のときは CUDA → CPU の順で自動選択される。
    """
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name=MODEL_NAME,
        allowed_modules=["detection", "recognition"],
        providers=providers,
    )
    app.prepare(ctx_id=0, det_thresh=det_thresh, det_size=det_size)
    det_session = getattr(app.det_model, "session", None)
    if det_session is not None:
        logger.info("onnxruntime providers: %s", det_session.get_providers())
    return app


def decode_image(data: bytes) -> np.ndarray:
    """画像バイト列を BGR uint8 に変換する（EXIF の回転も反映される）。"""
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("画像としてデコードできません")
    return img


def read_image(path: str | Path) -> np.ndarray:
    return decode_image(Path(path).read_bytes())


def _largest_face(faces):
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


# ---------- ギャラリー ----------
@dataclass
class Gallery:
    names: np.ndarray  # (N,) 各特徴量のメンバー名
    embeddings: np.ndarray  # (N, 512) L2 正規化済み
    paths: np.ndarray  # (N,) 元画像のパス
    members: list[str] = field(init=False)
    _member_index: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.members = sorted(set(self.names.tolist()))
        lookup = {m: i for i, m in enumerate(self.members)}
        self._member_index = np.array([lookup[n] for n in self.names], dtype=np.int64)

    def __len__(self):
        return len(self.names)

    def counts(self) -> dict[str, int]:
        return {m: int((self.names == m).sum()) for m in self.members}

    def member_scores(self, query: np.ndarray, exclude: np.ndarray | None = None) -> np.ndarray:
        """query (D,) に対するメンバーごとの最大コサイン類似度 (M,) を返す。"""
        sims = self.embeddings @ query
        if exclude is not None:
            sims = np.where(exclude, -np.inf, sims)
        scores = np.full(len(self.members), -np.inf, dtype=np.float32)
        np.maximum.at(scores, self._member_index, sims)
        return scores

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            names=self.names,
            embeddings=self.embeddings,
            paths=self.paths,
            model=np.array(MODEL_NAME),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Gallery":
        with np.load(path, allow_pickle=False) as d:
            model = str(d["model"]) if "model" in d.files else None
            if model != MODEL_NAME:
                raise ValueError(f"{path} は {model} で作られています（現在のモデル: {MODEL_NAME}）")
            return cls(names=d["names"], embeddings=d["embeddings"].astype(np.float32), paths=d["paths"])


def collect_known_images(known_dir: str | Path) -> dict[str, list[Path]]:
    """known_dir/<メンバー名>/*.jpg を {メンバー名: [画像パス]} にまとめる。"""
    people = {}
    for person_dir in sorted(Path(known_dir).iterdir()):
        if not person_dir.is_dir() or person_dir.name.startswith(".") or person_dir.name in EXCLUDE_DIRS:
            continue
        imgs = sorted(
            p for p in person_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_EXTS and not p.name.startswith(".")
        )
        if imgs:
            people[person_dir.name] = imgs
    return people


def build_gallery(app, known_dir: str | Path) -> Gallery:
    """登録画像から特徴量を作る。1 枚に複数の顔があるときは一番大きい顔を使う。"""
    names, embs, paths = [], [], []
    for name, img_paths in collect_known_images(known_dir).items():
        for p in img_paths:
            try:
                faces = app.get(read_image(p))
            except ValueError:
                logger.warning("読み込めない画像をスキップ: %s", p)
                continue
            if not faces:
                logger.warning("顔が見つからない画像をスキップ: %s", p)
                continue
            if len(faces) > 1:
                logger.info("顔が %d 個あるため最大の顔を使用: %s", len(faces), p)
            names.append(name)
            embs.append(_largest_face(faces).normed_embedding)
            paths.append(str(p))
    if not embs:
        raise RuntimeError(f"{known_dir} から顔を 1 つも登録できませんでした")
    return Gallery(
        names=np.array(names),
        embeddings=np.vstack(embs).astype(np.float32),
        paths=np.array(paths),
    )


# ---------- 判別 ----------
def _softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    z = (x - x.max()) / temperature
    e = np.exp(z)
    return e / e.sum()


def classify(
    scores: np.ndarray,
    members: list[str],
    min_sim: float = DEFAULT_MIN_SIM,
    min_margin: float = DEFAULT_MIN_MARGIN,
    temperature: float = DEFAULT_TEMPERATURE,
    topk: int = 3,
) -> dict:
    """メンバーごとのスコアから名前・類似度・確率・候補を決める。

    prob は登録メンバー内での相対的な確からしさ（類似度の softmax）で，
    未登録の人物でも 1 に近くなり得るため，Unknown 判定には similarity を使う。
    """
    probs = _softmax(scores, temperature)
    order = np.argsort(-scores)
    top1 = float(scores[order[0]])
    top2 = float(scores[order[1]]) if len(order) > 1 else -1.0
    known = top1 >= min_sim and (top1 - top2) >= min_margin
    return {
        "name": members[order[0]] if known else "Unknown",
        "similarity": top1,
        "prob": float(probs[order[0]]),
        "candidates": [
            {"name": members[j], "similarity": float(scores[j]), "prob": float(probs[j])}
            for j in order[:topk]
        ],
    }


def identify(
    app,
    gallery: Gallery,
    img_bgr: np.ndarray,
    min_sim: float = DEFAULT_MIN_SIM,
    min_margin: float = DEFAULT_MIN_MARGIN,
    temperature: float = DEFAULT_TEMPERATURE,
    topk: int = 3,
) -> list[dict]:
    """画像中の全ての顔を検出し，誰かを判別する。結果は左から順に並べる。"""
    results = []
    for face in app.get(img_bgr):
        x1, y1, x2, y2 = (int(round(v)) for v in face.bbox)
        r = {"bbox": [x1, y1, x2, y2], "det_score": float(face.det_score)}
        r.update(classify(gallery.member_scores(face.normed_embedding), gallery.members,
                          min_sim, min_margin, temperature, topk))
        results.append(r)
    results.sort(key=lambda r: r["bbox"][0])
    return results


# ---------- 描画 ----------
KNOWN_COLOR = (0, 200, 0)  # BGR
UNKNOWN_COLOR = (0, 140, 255)


def draw_results(
    img_bgr: np.ndarray,
    results: list[dict],
    show_candidates: bool = True,
    candidate_min_sim: float = 0.30,
) -> np.ndarray:
    """BB と名前を描く。Unknown の顔には candidate_min_sim 以上の候補だけを添える。"""
    vis = img_bgr.copy()
    h, w = vis.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for r in results:
        x1, y1, x2, y2 = r["bbox"]
        # 文字の大きさは顔の幅に合わせる（集合写真でラベル同士が重なりにくくする）
        scale = float(np.clip((x2 - x1) / 220, 0.4, max(0.5, min(w, h) / 900)))
        thick = max(1, int(round(scale * 2)))
        known = r["name"] != "Unknown"
        color = KNOWN_COLOR if known else UNKNOWN_COLOR
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thick + 1)

        lines = [f"{r['name']} {r['similarity']:.2f}"]
        if not known and show_candidates:
            lines += [f"? {c['name']} {c['similarity']:.2f}"
                      for c in r["candidates"][:3] if c["similarity"] >= candidate_min_sim]
        sizes = [cv2.getTextSize(t, font, scale, thick)[0] for t in lines]
        line_h = max(th for _, th in sizes) + int(8 * scale)
        box_w = max(tw for tw, _ in sizes) + int(8 * scale)
        box_h = line_h * len(lines)
        # ラベルは枠の上に置き，はみ出す場合は枠の下に置く
        top = y1 - box_h if y1 - box_h >= 0 else min(y2, h - box_h)
        left = min(max(0, x1), max(0, w - box_w))
        cv2.rectangle(vis, (left, top), (left + box_w, top + box_h), color, -1)
        for i, text in enumerate(lines):
            y = top + line_h * (i + 1) - int(5 * scale)
            cv2.putText(vis, text, (left + int(4 * scale), y), font, scale, (0, 0, 0), thick, cv2.LINE_AA)
    return vis
