from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import io

# main.py
from fastapi import FastAPI, File, UploadFile, Query
from typing import Optional
from pathlib import Path
from PIL import Image
import numpy as np
import io
from nogi_face_check import detect_faces_face_recognition
from datetime import datetime
from uuid import uuid4

import logging

app = FastAPI()
KNOWN_DIR="./nogi_member_face"

logger = logging.getLogger(f"uvicorn.{__name__}")

# --- 画像が np.uint8 RGB になってない時の安全化（detect内で使ってる想定に合わせて） ---
def ensure_rgb_uint8(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("image is None")
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    if img.dtype in (np.float32, np.float64):
        m = float(img.max() if img.size else 1.0)
        img = (img * (255.0 if m <= 1.0 else 1.0)).clip(0, 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)
    return np.ascontiguousarray(img)

# --- NumPy を JSON 化する軽量ヘルパ ---
def to_jsonable(obj):
    import numpy as _np
    if isinstance(obj, _np.generic):
        return obj.item()
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


SAVE_DIR = Path("uploads")

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    dlib_min_sim: float = Query(0.95),
    dlib_min_margin: float = Query(0.005),
    high_conf_override: float = Query(0.90),
    num_jitters_detect: int = Query(1, ge=0),
    topk_candidates: int = Query(3, ge=1),
):
    contents = await file.read()

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename).suffix.lower() or ".jpg"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = SAVE_DIR / f"{ts}_{uuid4().hex}{ext}"
    save_path.write_bytes(contents)

    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_array = np.array(image)
    img_array = ensure_rgb_uint8(img_array)

    logger.info(f"dlib_min_sim : {dlib_min_sim}, dlib_min_margin : {dlib_min_margin}, num_jitters : {num_jitters_detect}")
    results = detect_faces_face_recognition(
        image=img_array,
        known_dir=Path(KNOWN_DIR) if KNOWN_DIR else None,
        dlib_min_sim=dlib_min_sim,
        dlib_min_margin=dlib_min_margin,
        high_conf_override=high_conf_override,
        num_jitters_detect=num_jitters_detect,
        topk_candidates=topk_candidates,
    )

    payload = {
        "filename": file.filename,
        "saved_path": str(save_path),
        "results": to_jsonable(results),
        "params": {
            "dlib_min_sim": dlib_min_sim,
            "dlib_min_margin": dlib_min_margin,
            "high_conf_override": high_conf_override,
            "num_jitters_detect": num_jitters_detect,
            "topk_candidates": topk_candidates,
        },
    }
    
    return JSONResponse(content=payload)
