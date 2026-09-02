#!/usr/bin/env bash
# 对 inspect 报告含 error 的译文重做一次：复用翻译缓存（命中块不调 API）、提高最小字号、放不下的块保留原文，然后重新检查。
# 用法: bash scripts/translate_retry.sh [base_name ...]   # 不传参则扫描 papers_zh 下所有含 error 的译文
# 环境变量 RETRY_FLAGS 可覆盖排版参数（默认 --min-font-size 7.5 --skip-overflow；设为空串则只重请求未译块）
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
ST=/Users/liyufeng/Code/super_translate
PY="$ST/.venv/bin/python"
ENV_FILE="${TRANSLATE_ENV_FILE:-/Users/liyufeng/Desktop/research/paper_china/.env}"
[ -f "$ENV_FILE" ] && { set -a; source "$ENV_FILE"; set +a; }
mkdir -p "$REPO/logs/translate"

has_error() { grep -q '"severity": *"error"' "$1" 2>/dev/null; }

targets=("$@")
if [ ${#targets[@]} -eq 0 ]; then
  for rep in "$REPO"/papers_zh/*.zh.inspect.json; do
    [ -f "$rep" ] || continue
    if has_error "$rep"; then
      b="$(basename "${rep%.zh.inspect.json}")"; targets+=("$b")
    fi
  done
fi
[ ${#targets[@]} -eq 0 ] && { echo "无需重试"; exit 0; }

for b in "${targets[@]}"; do
  src="$REPO/papers/$b.pdf"; out="$REPO/papers_zh/$b.zh.pdf"
  cache="${out}.translation-cache.jsonl"; rep="${out%.pdf}.inspect.json"
  log="$REPO/logs/translate/$b.retry.log"
  [ -f "$cache" ] || { echo "[skip] $b 无缓存"; continue; }
  echo "[retry] $b $(date '+%H:%M:%S')"
  tmp="${out%.pdf}.retry.pdf"
  ( cd "$ST" && "$PY" -m pdf_zh_translator translate "$src" "$tmp" \
      --api-mode deepseek --api-key-env PAPER_CHINA_DEEPSEEK_API_KEY \
      --cache-file "$cache" --preserve-graphics-text --timeout 180 \
      ${RETRY_FLAGS---min-font-size 7.5 --skip-overflow} ) > "$log" 2>&1
  if [ ! -f "$tmp" ]; then echo "[FAIL ] $b 重放未产出，见 $log"; continue; fi
  ( cd "$ST" && "$PY" -m pdf_zh_translator inspect "$src" "$tmp" --json-out "${tmp%.pdf}.inspect.json" ) >> "$log" 2>&1
  if has_error "${tmp%.pdf}.inspect.json"; then
    n_old=$(grep -c '"severity": *"error"' "$rep" 2>/dev/null || echo 99)
    n_new=$(grep -c '"severity": *"error"' "${tmp%.pdf}.inspect.json")
    if [ "$n_new" -lt "$n_old" ]; then
      mv "$tmp" "$out"; mv "${tmp%.pdf}.inspect.json" "$rep"
      echo "[better] $b errors $n_old -> ${n_new}（仍有 error，记入 QA 表）"
    else
      rm -f "$tmp" "${tmp%.pdf}.inspect.json"; echo "[keep ] $b 重放未改善（errors ${n_old}）"
    fi
  else
    mv "$tmp" "$out"; mv "${tmp%.pdf}.inspect.json" "$rep"; echo "[pass ] $b 重放后 QA 通过"
  fi
done
