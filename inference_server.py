"""登録メンバーの顔判別 API。

    uv sync --extra gpu        # 最初に 1 回（CPU のみなら --extra cpu）
    ./run_server.sh

環境変数
    NOGI_KNOWN_DIR      登録画像のディレクトリ (既定: ./nogi_member_face)
    NOGI_GALLERY        特徴量ファイル (既定: ./gallery.npz, 無ければ起動時に作成)
    NOGI_SAVE_UPLOADS   1 にするとアップロード画像を ./uploads に保存
"""
import base64
import logging
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

from nogi_face_check import (
    DEFAULT_MIN_MARGIN,
    DEFAULT_MIN_SIM,
    DEFAULT_TEMPERATURE,
    Gallery,
    build_gallery,
    decode_image,
    draw_results,
    identify,
    load_face_app,
)

KNOWN_DIR = Path(os.environ.get("NOGI_KNOWN_DIR", "./nogi_member_face"))
GALLERY_PATH = Path(os.environ.get("NOGI_GALLERY", "./gallery.npz"))
SAVE_UPLOADS = os.environ.get("NOGI_SAVE_UPLOADS", "0") == "1"
UPLOAD_DIR = Path("uploads")

logger = logging.getLogger(f"uvicorn.{__name__}")
state = {}
_rebuild_lock = threading.Lock()


def _load_or_build_gallery(app, rebuild: bool = False) -> Gallery:
    if GALLERY_PATH.exists() and not rebuild:
        gallery = Gallery.load(GALLERY_PATH)
    else:
        logger.info("ギャラリーを %s から作成します", KNOWN_DIR)
        gallery = build_gallery(app, KNOWN_DIR)
        gallery.save(GALLERY_PATH)
    logger.info("ギャラリー: %d 人 / %d 枚", len(gallery.members), len(gallery))
    return gallery


@asynccontextmanager
async def lifespan(_: FastAPI):
    face_app = load_face_app()
    state["face_app"] = face_app
    state["gallery"] = _load_or_build_gallery(face_app)
    yield
    state.clear()


app = FastAPI(title="nogi-face", lifespan=lifespan)


async def _run(file: UploadFile, min_sim: float, min_margin: float, temperature: float, topk: int):
    contents = await file.read()
    try:
        img = decode_image(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    saved_path = None
    if SAVE_UPLOADS:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        ext = Path(file.filename or "").suffix.lower() or ".jpg"
        saved = UPLOAD_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex}{ext}"
        saved.write_bytes(contents)
        saved_path = str(saved)

    results = identify(state["face_app"], state["gallery"], img,
                       min_sim=min_sim, min_margin=min_margin, temperature=temperature, topk=topk)
    payload = {
        "filename": file.filename,
        "image_size": {"width": img.shape[1], "height": img.shape[0]},
        "num_faces": len(results),
        "people": sorted({r["name"] for r in results if r["name"] != "Unknown"}),
        "faces": results,
        "params": {"min_sim": min_sim, "min_margin": min_margin, "temperature": temperature, "topk": topk},
    }
    if saved_path:
        payload["saved_path"] = saved_path
    return img, results, payload


def _encode_jpeg(img) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise HTTPException(status_code=500, detail="JPEG エンコードに失敗しました")
    return buf.tobytes()


MinSim = Query(DEFAULT_MIN_SIM, ge=-1.0, le=1.0, description="これ未満の類似度は Unknown")
MinMargin = Query(DEFAULT_MIN_MARGIN, ge=0.0, description="1 位と 2 位の類似度差がこれ未満なら Unknown")
Temperature = Query(DEFAULT_TEMPERATURE, gt=0.0, description="prob を計算する softmax の温度")
TopK = Query(3, ge=1, description="candidates に含める人数")


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    min_sim: float = MinSim,
    min_margin: float = MinMargin,
    temperature: float = Temperature,
    topk: int = TopK,
    return_image: bool = Query(False, description="BB を描いた JPEG を image_base64 として含める"),
):
    """画像中の顔を検出し，誰が写っているかを JSON で返す。"""
    img, results, payload = await _run(file, min_sim, min_margin, temperature, topk)
    if return_image:
        payload["image_base64"] = base64.b64encode(_encode_jpeg(draw_results(img, results))).decode()
    return payload


@app.post("/predict/image", response_class=Response,
          responses={200: {"content": {"image/jpeg": {}}}})
async def predict_image(
    file: UploadFile = File(...),
    min_sim: float = MinSim,
    min_margin: float = MinMargin,
    temperature: float = Temperature,
    topk: int = TopK,
):
    """BB と名前を描いた JPEG を返す。"""
    img, results, _ = await _run(file, min_sim, min_margin, temperature, topk)
    return Response(content=_encode_jpeg(draw_results(img, results)), media_type="image/jpeg")


@app.get("/gallery")
def gallery_info():
    gallery = state["gallery"]
    return {"path": str(GALLERY_PATH), "num_members": len(gallery.members),
            "num_embeddings": len(gallery), "members": gallery.counts()}


@app.post("/gallery/reload")
def gallery_reload(rebuild: bool = Query(False, description="true なら登録画像から特徴量を作り直す")):
    """gallery.npz を読み直す。rebuild=true なら登録画像から作り直す（数十秒〜数分かかる）。"""
    if not _rebuild_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="別の再読み込みが実行中です")
    try:
        state["gallery"] = _load_or_build_gallery(state["face_app"], rebuild=rebuild)
    finally:
        _rebuild_lock.release()
    return gallery_info()


@app.get("/health")
def health():
    return {"status": "ok", "members": len(state["gallery"].members)}
