# nogi-face

InsightFace（buffalo_l：SCRFD で顔検出 + ArcFace で特徴量）を使い，画像に写っている登録メンバーを判別する推論サーバー。

## セットアップ（uv）

最初に 1 回，使う runtime を選んで環境を作る。

```bash
uv sync --extra gpu   # NVIDIA GPU（onnxruntime-gpu と CUDA / cuDNN を pip で導入。ホスト側は NVIDIA ドライバーのみ必要）
uv sync --extra cpu   # CPU のみ
```

以降は `uv run ...` を extras なしで使えば，選んだ runtime のまま動く（`uv run` はパッケージを追加するだけで削除はしないため）。

初回実行時に buffalo_l（約 280MB）が `~/.insightface/models/` にダウンロードされる。

> **extras なしの `uv sync` は使わない。** `uv sync` は環境を指定どおりに揃えるため，extras を付けないと onnxruntime が削除される。
> cpu / gpu を切り替えるときも `uv sync --extra cpu` のように `uv sync` で行う（`uv run --extra` だと前の onnxruntime が残り，混在する）。

### 環境チェック

```bash
uv run python check_env.py   # GPU / ドライバー / CUDA / モデル / 推論速度 / gallery を順に確認
```

> buffalo_l などの学習済みモデルは **非商用の研究目的に限る** ライセンス。業務で使う場合は事前に確認すること。

## 登録画像と特徴量

`nogi_member_face/<メンバー名>/*.jpg` に 1 人 1 顔の画像を置く。`preset_images_dir/`，`multipleFaces/`，`noFace/` は登録に使わない。

```bash
uv run python build_gallery.py --eval
```

`gallery.npz` が作られ，`--eval` を付けると leave-one-out で正解率と，未登録の人物を誤認する率を表示する。
`gallery.npz` が無い状態でサーバーを起動した場合は，起動時に自動で作成される。

## サーバー

```bash
./run_server.sh                    # uv sync で選んだ runtime のまま起動（port 18381）
NOGI_RUNTIME=cpu ./run_server.sh   # runtime を cpu に入れ替えてから起動（以降はそのまま cpu）
```

| エンドポイント | 内容 |
|---|---|
| `POST /predict` | JSON で結果を返す。`return_image=true` で BB 付き JPEG を `image_base64` に含める |
| `POST /predict/image` | BB と名前を描いた JPEG を返す |
| `GET /gallery` | 登録メンバーと枚数 |
| `POST /gallery/reload` | `gallery.npz` を読み直す。`rebuild=true` で登録画像から作り直す |
| `GET /health` | 死活確認 |

`/docs` で Swagger UI が使える。

### レスポンス例

```json
{
  "num_faces": 1,
  "people": ["EndoSakura"],
  "faces": [{
    "bbox": [86, 84, 248, 283],
    "det_score": 0.87,
    "name": "EndoSakura",
    "similarity": 0.71,
    "prob": 0.99,
    "candidates": [
      {"name": "EndoSakura", "similarity": 0.71, "prob": 0.99},
      {"name": "OzonoMomoko", "similarity": 0.48, "prob": 0.01}
    ]
  }]
}
```

- `bbox`: `[x1, y1, x2, y2]`（ピクセル）
- `similarity`: 一番近いメンバーとのコサイン類似度。`min_sim`（既定 0.50）未満，または 2 位との差が `min_margin`（既定 0.05）未満なら `name` は `Unknown`
- `prob`: 登録メンバー内での相対的な確からしさ（類似度の softmax，温度 `temperature`）。未登録の人物でも高くなり得るため，Unknown の判定には `similarity` を見る

## フォルダ単位のテスト

API を通さずに，フォルダ内の画像をまとめて判別し，BB を描いた画像を出力フォルダに保存する。

```bash
uv run python predict_dir.py test_images -o out/test_images        # -r でサブフォルダも対象
```

- 最初に使っている device（GPU (CUDA) / CPU）を表示し，進捗バーと画像ごとの結果・推論時間を出す
- 最後に枚数・顔の数・Unknown 数，1 枚あたりの時間（mean / median / p95）と人物ごとの集計を表示する
- 出力フォルダには入力と同じ構成で BB 付き画像を置き，全結果を `results.json` に保存する
- 画像の親フォルダ名が登録メンバー名（例: `test_images/EndoSakura/xxx.jpg`）なら正解として正解率も出す
- 空でない出力フォルダへは `--overwrite` を付けたときだけ書き込む

## クライアント

```bash
uv run python client.py -i photo.jpg --url http://<server>:18381
```

`out/` に BB 付き画像と JSON を保存する。
