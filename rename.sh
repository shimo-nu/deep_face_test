#!/usr/bin/env bash
set -euo pipefail

# リネーム対象の親ディレクトリ（デフォルト: ./a）
APPLY="false"
BASE_DIR="./nogi_member_face"

# 引数処理
for arg in "$@"; do
  case "$arg" in
    --apply) APPLY="true" ;;
    --dry-run) APPLY="false" ;;  # 明示ドライラン
    /*|.|./*|../*)
      BASE_DIR="$arg"
      ;;
    *)
      # 相対パス（例: a や people）
      if [[ -d "$arg" ]]; then
        BASE_DIR="$arg"
      fi
      ;;
  esac
done

if [[ ! -d "$BASE_DIR" ]]; then
  echo "親ディレクトリが見つかりません: $BASE_DIR" >&2
  exit 1
fi

# CamelCase ディレクトリ名 -> snake_case 接頭辞へ
camel_to_snake () {
  # 例: "AkimotoManatsu" -> "akimoto_manatsu"
  local s="$1"
  # 大文字の前に "_" を入れ、小文字化、先頭の "_" を除去、連続 "_" を1つに
  s="$(printf '%s' "$s" | sed -E 's/([A-Z])/_\1/g' | tr 'A-Z' 'a-z' | sed -E 's/^_+//; s/_+/_/g')"
  printf '%s' "$s"
}

# ある人物ディレクトリ内の既存最大インデックスを取得
# 例: akimoto_manatsu_1.jpg, _5.jpg があれば -> 5
max_index_in_dir () {
  local dir="$1"
  local prefix="$2"
  local max=0
  # 大文字拡張を無視して安全に列挙（nullglob 有効化）
  local oldopt
  oldopt=$(shopt -p nullglob || true)
  shopt -s nullglob
  for f in "$dir/${prefix}"_[0-9]*.jpg "$dir/${prefix}"_[0-9]*.JPG "$dir/${prefix}"_[0-9]*.jpeg "$dir/${prefix}"_[0-9]*.JPEG; do
    # ファイル名から数字部分だけ抽出
    local bn num
    bn="$(basename "$f")"
    # 末尾の .jpg/.jpeg を落としてから最後の "_" 以降
    num="${bn%.*}"
    num="${num##*_}"
    # 数字なら比較
    if [[ "$num" =~ ^[0-9]+$ ]]; then
      if (( num > max )); then max="$num"; fi
    fi
  done
  # 元に戻す
  eval "$oldopt" >/dev/null 2>&1 || true
  printf '%d' "$max"
}

# 実処理
echo "対象の親ディレクトリ: $BASE_DIR"
echo "モード: $([[ "$APPLY" == "true" ]] && echo APPLY || echo DRY-RUN)"
echo

# サブディレクトリを 1 階層だけ走査
find "$BASE_DIR" -mindepth 1 -maxdepth 1 -type d -print0 | while IFS= read -r -d '' person_dir; do
  dir_name="$(basename "$person_dir")"
  prefix="$(camel_to_snake "$dir_name")"

  # スキップ条件（空ディレクトリは続行でOK）
  # ここで unmatched JPG を列挙：prefix_[0-9]*.jpg / prefix_default.jpg を除外
  mapfile -d '' unmatched < <(
    find "$person_dir" -maxdepth 1 -type f \
      \( -iname '*.jpg' -o -iname '*.jpeg' \) \
      ! -iname "${prefix}_[0-9]*.jpg" \
      ! -iname "${prefix}_[0-9]*.jpeg" \
      ! -iname "${prefix}_default.jpg" \
      ! -iname "${prefix}_default.jpeg" -print0
  )

  # 何も無ければスキップ
  if (( ${#unmatched[@]} == 0 )); then
    echo "[$dir_name] 追加リネーム対象なし。"
    continue
  fi

  # 既存の最大インデックスを取得
  max_idx="$(max_index_in_dir "$person_dir" "$prefix")"
  next_idx=$(( max_idx + 1 ))

  echo "[$dir_name] prefix='${prefix}', 既存最大=${max_idx}, 変換対象=${#unmatched[@]} 件"

  # 変換を実施（投入順で連番付与）
  for src in "${unmatched[@]}"; do
    ext="$(printf '%s' "$src" | sed -E 's/.*\.([Jj][Pp][Ee]?[Gg])$/\1/')"
    # 拡張子は元と同じ大文字小文字を維持
    dst="$person_dir/${prefix}_${next_idx}.${ext}"
    if [[ -e "$dst" ]]; then
      echo "  衝突のためスキップ: $(basename "$src") -> $(basename "$dst") (既に存在)"
      continue
    fi
    if [[ "$APPLY" == "true" ]]; then
      mv -- "$src" "$dst"
      echo "  mv: $(basename "$src") -> $(basename "$dst")"
    else
      echo "  mv: $(basename "$src") -> $(basename "$dst")  (dry-run)"
    fi
    next_idx=$(( next_idx + 1 ))
  done

  echo
done

echo "完了。"
