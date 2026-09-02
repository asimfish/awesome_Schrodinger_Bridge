#!/usr/bin/env python3
"""汇总 papers_zh/*.zh.inspect.json 为 papers_zh/QA_REPORT.md（译本视觉 QA 结果表）。"""
import json, pathlib, csv, datetime, re
ROOT = pathlib.Path(__file__).resolve().parents[1]
try:
    import fitz
except ImportError:
    fitz = None

def pages(p):
    if fitz is None or not p.exists():
        return "?"
    d = fitz.open(p); n = d.page_count; d.close(); return n

rows = list(csv.DictReader(open(ROOT / "metadata/papers.tsv", encoding="utf-8"), delimiter="\t"))
ext = {r["key"]: r for r in csv.DictReader(open(ROOT / "metadata/extended.tsv", encoding="utf-8"), delimiter="\t")}
known = {r["id"] for r in rows}
for p in sorted((ROOT / "papers").glob("*.pdf")):  # 扩展条目的 PDF（非核心 25 篇）
    key = p.name.split("_")[0]
    if key in known: continue
    e = ext.get(key, {})
    rows.append(dict(id=key, title=e.get("title", p.stem), pdf_path=f"papers/{p.name}"))
L = ["# 中文译本视觉 QA 报告", "",
     f"生成时间：{datetime.date.today().isoformat()}。翻译引擎：[SuperTranslate](https://github.com/asimfish/super_translate)（DeepSeek 后端，保版式，`--preserve-graphics-text`）；"
     "QA 由引擎 `inspect` 子命令逐页比对原文/译文产出（页数一致性、图像/公式丢失、文字重叠、字号漂移、列表字号不一致等）。",
     "", "判定口径：**通过** = 0 个 error 级 issue；**通过（有备注）** = 全部 error 属于局部排版类（字号缩放/列表字号）或 ≤12 词的公式碎片保留英文（如 `is, T = T_aff[X, Y].`），译文内容完整、可读，问题位置已列出供读者知悉；**需人工复核** = 存在成段（>12 词）未译或表格结构错位；"
     "参考文献与图内文字按设计保留英文。", "",
     "| arXiv | 论文 | 页数（原/译） | issues | errors | 状态 | error 位置 |", "|---|---|---|---|---|---|---|"]
n_pass = n_note = n_fail = 0
for r in rows:
    stem = pathlib.Path(r["pdf_path"]).stem
    zh = ROOT / "papers_zh" / f"{stem}.zh.pdf"; rep = ROOT / "papers_zh" / f"{stem}.zh.inspect.json"
    if not zh.exists():
        L.append(f"| {r['id']} | {r['title']} | — | — | — | ⏳ 未生成 | |"); n_fail += 1; continue
    issues = []
    if rep.exists():
        d = json.load(open(rep, encoding="utf-8")); issues = d.get("issues", d if isinstance(d, list) else [])
    errs = [i for i in issues if i.get("severity") == "error"]
    def cosmetic(i):
        if i.get("code") in ("font_size_drift", "list_font_inconsistent", "font_ratio_drift"):
            return True
        if i.get("code") == "untranslated_block":
            m = re.search(r"\((\d+) words", i.get("message", ""))
            return bool(m) and int(m.group(1)) <= 12  # 公式碎片保留英文
        return False
    if not errs: status = "✅ 通过"; n_pass += 1
    elif all(cosmetic(i) for i in errs): status = "✅ 通过（有备注）"; n_note += 1
    else: status = "⚠️ 需人工复核"; n_fail += 1
    loc = "; ".join(f"p{i.get('page')} {i.get('code')}" for i in errs) or "—"
    L.append(f"| {r['id']} | [{r['title']}](./{zh.name}) | {pages(ROOT / r['pdf_path'])}/{pages(zh)} | {len(issues)} | {len(errs)} | {status} | {loc} |")
L += ["", f"**汇总**：通过 {n_pass} · 通过（有备注）{n_note} · 需复核 {n_fail} · 共 {len(rows)} 篇。", "",
      "复现：`bash scripts/translate_batch.sh 3` 生成译本并自动 QA；`bash scripts/translate_retry.sh` 对含 error 的译文用缓存重做；`python3 scripts/qa_table.py` 重建本表。"]
(ROOT / "papers_zh/QA_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print(f"QA 表已生成：通过 {n_pass}，通过（有备注）{n_note}，需复核 {n_fail}")
