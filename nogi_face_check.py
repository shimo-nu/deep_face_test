# =======================
# face_recognition 単独版：検出関数（＋任意の照合）
# そのまま Jupyter で1セル実行OK
# =======================
from pathlib import Path
import os, hashlib
import numpy as np
import cv2
import face_recognition

# ---------- ユーティリティ ----------
VALID_EXTS = {".jpg", ".jpeg", ".png"}

def ensure_rgb_uint8(img):
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr

def l2_normalize(v, eps=1e-12):
    v = np.asarray(v)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, eps, None)

def cosine_sim(a, b):
    a = l2_normalize(a); b = l2_normalize(b)
    return np.dot(a, b.T)

# ---------- 顔エンコード ----------
def _safe_face_locations(img_rgb):
    locs = face_recognition.face_locations(img_rgb)
    return [(int(t), int(r), int(b), int(l)) for (t,r,b,l) in locs]

def _encode_rgb(img_rgb, face_locations=None, num_jitters=2, do_flip_tta=True):
    img_rgb = ensure_rgb_uint8(img_rgb)
    if face_locations is None:
        face_locations = _safe_face_locations(img_rgb)
    embeddings = face_recognition.face_encodings(img_rgb, face_locations, num_jitters=num_jitters)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if do_flip_tta and len(embeddings):
        img_flip = img_rgb[:, ::-1, :].copy()
        w = img_rgb.shape[1]
        locs_flip = [(t, w - l, b, w - r) for (t,r,b,l) in face_locations]
        locs_flip = [(int(t), int(r), int(b), int(l)) for (t,r,b,l) in locs_flip]
        emb_flip = face_recognition.face_encodings(img_flip, locs_flip, num_jitters=1)
        if len(emb_flip) == len(embeddings):
            embeddings = (embeddings + np.asarray(emb_flip, dtype=np.float32)) / 2.0
    embeddings = l2_normalize(embeddings)
    return embeddings, face_locations

def _auto_scale_encode(img_rgb, target_min_face_px=80, scales=(1.0,1.5,2.0), num_jitters=2):
    H, W = img_rgb.shape[:2]
    best = ([], [])
    for s in scales:
        img_s = img_rgb if s == 1.0 else cv2.resize(img_rgb, (int(W*s), int(H*s)), interpolation=cv2.INTER_LINEAR)
        encs, locs = _encode_rgb(img_s, num_jitters=num_jitters, do_flip_tta=True)
        if len(locs):
            widths = [r - l for (t,r,b,l) in locs]
            if max(widths) >= target_min_face_px:
                if s != 1.0:
                    locs = [(int(t/s), int(r/s), int(b/s), int(l/s)) for (t,r,b,l) in locs]
                return encs, locs
            best = (encs, [(int(t/s), int(r/s), int(b/s), int(l/s)) if s!=1.0 else (t,r,b,l) for (t,r,b,l) in locs])
    return best

# ---------- ギャラリー構築 ----------
_DLIB_CACHE = {"sig": None, "templates": None, "dir": None, "person_thr": None}

def _dir_signature(known_dir: Path):
    h = hashlib.sha1()
    for root, dirs, files in os.walk(known_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in sorted(files):
            if f.startswith("."): continue
            p = Path(root) / f
            if p.suffix.lower() not in VALID_EXTS: continue
            stat = p.stat()
            h.update(str(p).encode()); h.update(str(int(stat.st_mtime)).encode())
    return h.hexdigest()

def _collect_known_images(known_dir: Path):
    people = {}
    subdirs = [p for p in known_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if subdirs:
        for person_dir in sorted(subdirs):
            imgs = []
            for ext in VALID_EXTS:
                imgs += list(person_dir.glob(f"*{ext}"))
            imgs = [p for p in imgs if not p.name.startswith(".")]
            if imgs:
                people[person_dir.name] = imgs
    else:
        imgs = [p for p in known_dir.iterdir()
                if p.is_file() and p.suffix.lower() in VALID_EXTS and not p.name.startswith(".")]
        for p in imgs:
            people.setdefault(p.stem, []).append(p)
    return people

def _build_person_thresholds_robust(gallery: dict, percentile=0.98, floor=0.58, cap=0.84):
    names = list(gallery.keys())
    if not names:
        return {}
    G = np.vstack([gallery[n] for n in names])
    thr = {}
    for i, name in enumerate(names):
        sims = np.dot(G[i], G.T); sims = np.delete(sims, i)
        if sims.size == 0:
            t = floor
        else:
            q = float(np.quantile(sims, percentile))
            q25, q75 = np.percentile(sims, [25, 75])
            iqr = q75 - q25
            iqr_cap = q75 + 1.5 * iqr
            t = min(q, iqr_cap)
        thr[name] = float(np.clip(t, floor, cap))
    return thr

def build_gallery_face_recognition_cached(
    known_dir: str | Path,
    num_jitters: int = 1,
    force_rebuild: bool = False,
    person_thr_percentile: float = 0.98,
    person_thr_floor: float = 0.58,
    person_thr_cap: float = 0.84
):
    known_dir = Path(known_dir)
    sig = _dir_signature(known_dir)
    if (not force_rebuild) and _DLIB_CACHE["sig"] == sig and _DLIB_CACHE["dir"] == str(known_dir):
        return _DLIB_CACHE["templates"], _DLIB_CACHE["person_thr"]
    people = _collect_known_images(known_dir)
    templates = {}
    for name, paths in people.items():
        vecs = []
        for path in paths:
            img = face_recognition.load_image_file(str(path))
            img = ensure_rgb_uint8(img)
            encs, _ = _auto_scale_encode(img, target_min_face_px=80, scales=(1.0,1.5,2.0), num_jitters=num_jitters)
            if len(encs):
                vecs.append(np.mean(encs, axis=0))
        if vecs:
            templates[name] = l2_normalize(np.mean(np.vstack(vecs), axis=0))
    person_thr = _build_person_thresholds_robust(templates, percentile=person_thr_percentile,
                                                 floor=person_thr_floor, cap=person_thr_cap)
    _DLIB_CACHE.update({"sig": sig, "templates": templates, "dir": str(known_dir), "person_thr": person_thr})
    print(f"[face_recognition] gallery loaded: {list(templates.keys())}")
    return templates, person_thr

# ---------- メイン関数（TOP3候補付き） ----------
def detect_faces_face_recognition(
    image,
    known_dir: str | Path | None = None,
    dlib_min_sim: float = 0.66,
    dlib_min_margin: float = 0.03,
    high_conf_override: float = 0.90,
    
    num_jitters_detect: int = 2,
    topk_candidates: int = 3
):
    if isinstance(image, (str, Path)):
        img_rgb = face_recognition.load_image_file(str(image))
    else:
        img_rgb = ensure_rgb_uint8(image)

    encs, face_locations = _auto_scale_encode(img_rgb, target_min_face_px=80,
                                              scales=(1.0,1.5,2.0), num_jitters=num_jitters_detect)

    templates, person_thr = ({}, {})
    if known_dir is not None:
        templates, person_thr = build_gallery_face_recognition_cached(known_dir, num_jitters=1, force_rebuild=False)

    names = list(templates.keys())
    gal = np.vstack([templates[n] for n in names]) if names else None

    results = []
    for q, (t,r,b,l) in zip(encs, face_locations):
        label, sim_best, candidates = "Unknown", 0.0, []
        if gal is not None and len(gal) > 0:
            sims = cosine_sim(q[None, :], gal)[0]
            order = np.argsort(-sims)
            topk = [(names[j], float(sims[j])) for j in order[:topk_candidates]]
            j = int(order[0])
            top1 = float(sims[j])
            top2 = float(sims[order[1]]) if len(order) > 1 else -1.0

            # 高確度オーバーライド
            if top1 >= high_conf_override and (top1 - top2) >= dlib_min_margin * 0.5:
                label, sim_best = names[j], top1
            else:
                ok = (top1 >= dlib_min_sim) and ((top1 - top2) >= dlib_min_margin) and \
                     (top1 >= person_thr.get(names[j], dlib_min_sim))
                label, sim_best = (names[j], top1) if ok else ("Unknown", top1)
            candidates = topk
        results.append({
            'box': (int(t), int(r), int(b), int(l)),
            'name': label,
            'similarity': float(sim_best),
            'candidates': candidates
        })
    return results




# =======================
# visualize helpers for detect_faces_face_recognition
# =======================
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import face_recognition

def _to_rgb(img_or_path):
    """path でも ndarray でも受け取り、RGB ndarray にして返す"""
    if isinstance(img_or_path, (str, Path)):
        rgb = face_recognition.load_image_file(str(img_or_path))  # 既にRGB
        return rgb
    arr = np.asarray(img_or_path)
    if arr.ndim == 3 and arr.shape[2] == 3:
        # ユーザがBGRで渡してる可能性があるので、簡易判定でBGR→RGBを選択
        # 迷ったら明示的に RGB で渡すのが安心。ここでは「そのままRGBとして扱う」方針にします。
        return arr.astype(np.uint8, copy=False)
    elif arr.ndim == 2:
        return np.stack([arr]*3, axis=-1).astype(np.uint8, copy=False)
    else:
        raise ValueError("Unsupported image array format.")

# ---------- visualize ----------
def draw_results_on_image(rgb_img, results, box_color=(0,255,0), text_scale=0.7, text_thickness=2):
    vis = rgb_img.copy()
    for r in results:
        (t, rgt, btm, lft) = r['box']
        name = r.get('name', 'Unknown')
        sim  = r.get('similarity', 0.0)
        cands = r.get('candidates', [])
        # rectangle
        cv2.rectangle(vis, (lft, t), (rgt, btm), box_color, 2)
        # メインラベル
        label = f"{name} ({sim*100:.1f}%)"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)
        y1 = max(0, t - th - 6)
        cv2.rectangle(vis, (lft, y1), (lft + tw + 6, y1 + th + 6), (0, 0, 0), -1)
        cv2.putText(vis, label, (lft + 3, y1 + th + 3), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0,255,0), text_thickness)

        # # Unknown の場合、候補を下に
        # if name == "Unknown" and cands:
        #     offset = y1 + th + 8
        #     for (cand_name, score) in cands:
        #         txt = f"  → {cand_name} ({score*100:.1f}%)"
        #         cv2.putText(vis, txt, (lft + 3, offset), cv2.FONT_HERSHEY_SIMPLEX, text_scale*0.7, (0,200,255), 1)
        #         offset += int(th*0.9)
    return vis

def visualize_detection(image, res, title=None, save_path=None, dpi=120):
    """
    image: 画像パス or RGB ndarray
    res: detect_faces_face_recognition の返り値
    title: 図タイトル（任意）
    save_path: 保存先（任意）。指定したらPNGで保存
    """
    rgb = _to_rgb(image)
    vis = draw_results_on_image(rgb, res)
    plt.figure(figsize=(8, 8), dpi=dpi)
    plt.imshow(vis)
    if title: plt.title(title)
    plt.axis("off")
    if save_path:
        # matplotlib から保存すると色ずれなし（RGBのまま）
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.0)
        print(f"saved: {save_path}")
    plt.show()

def visualize_directory(input_dir, known_dir=None, glob_exts=("*.jpg","*.png"), cols=2, dpi=120,
                        dlib_min_sim=0.66, dlib_min_margin=0.03, high_conf_override=0.90):
    """
    ディレクトリ内の画像をグリッドで可視化
    """
    paths = []
    p = Path(input_dir)
    for pat in glob_exts:
        paths += list(p.glob(pat))
    paths = sorted(paths)
    if not paths:
        print("画像が見つかりません"); return

    # 行列数
    n = len(paths)
    rows = (n + cols - 1) // cols

    plt.figure(figsize=(8*cols, 6*rows), dpi=dpi)
    for i, fp in enumerate(paths, 1):
        res = detect_faces_face_recognition(
            str(fp),
            known_dir=known_dir,
            dlib_min_sim=dlib_min_sim,
            dlib_min_margin=dlib_min_margin,
            high_conf_override=high_conf_override
        )
        rgb = _to_rgb(str(fp))
        vis = draw_results_on_image(rgb, res)
        ax = plt.subplot(rows, cols, i)
        ax.imshow(vis); ax.set_title(fp.name); ax.axis("off")
    plt.tight_layout(); plt.show()
