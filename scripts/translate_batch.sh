#!/usr/bin/env bash
# 批量保版式翻译 papers/*.pdf -> papers_zh/*.zh.pdf（SuperTranslate 引擎），可断点续跑。
# 用法: bash scripts/translate_batch.sh [并行数=3] [论文目录=papers]
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PAR="${1:-3}"
SRC_DIR="${2:-papers}"
ENV_FILE="${TRANSLATE_ENV_FILE:-/Users/liyufeng/Desktop/research/paper_china/.env}"
SKILL_SH="/Users/liyufeng/Code/super_translate/skills/paper-translate/scripts/translate_one.sh"

if [ -f "$ENV_FILE" ]; then
  set -a; source "$ENV_FILE"; set +a
fi
mkdir -p "$REPO/papers_zh" "$REPO/logs/translate"

translate_one() {
  local pdf="$1"
  local base; base="$(basename "${pdf%.pdf}")"
  local out="$REPO/papers_zh/${base}.zh.pdf"
  local log="$REPO/logs/translate/${base}.log"
  local report="${out%.pdf}.inspect.json"
  # 已有译文与检查报告即跳过；QA 是否通过由 scripts/qa_table.py 汇总，不在此重做
  if [ -f "$out" ] && [ -f "$report" ]; then
    echo "[skip] $base 已完成"; return 0
  fi
  echo "[start] $base $(date '+%H:%M:%S')"
  if bash "$SKILL_SH" "$pdf" "$out" --timeout 180 > "$log" 2>&1; then
    echo "[done ] $base $(date '+%H:%M:%S')"
  else
    echo "[FAIL ] $base $(date '+%H:%M:%S') 见 $log"
  fi
}
export -f translate_one
export REPO SKILL_SH

# 小文件优先，早出结果
ls -Sr "$REPO/$SRC_DIR"/*.pdf | xargs -P "$PAR" -I{} bash -c 'translate_one "$@"' _ {}

echo "== 批次结束 $(date '+%H:%M:%S')"
ls "$REPO/papers_zh"/*.zh.pdf 2>/dev/null | wc -l | xargs echo "已产出中文 PDF:"
