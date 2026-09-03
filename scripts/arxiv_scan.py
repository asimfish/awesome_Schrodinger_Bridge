#!/usr/bin/env python3
"""近 N 个月 arXiv 扫描：SB / bridge matching / diffusion bridge 相关新论文 → survey/raw/S2_arxiv_scan_<date>.md + .tsv。
用法: python3 scripts/arxiv_scan.py [--months 12] [--max 400]
"""
import argparse, csv, datetime, json, pathlib, re, sys, time, urllib.parse, urllib.request, xml.etree.ElementTree as ET
ROOT = pathlib.Path(__file__).resolve().parents[1]
NS = {"a": "http://www.w3.org/2005/Atom"}; AX = "{http://arxiv.org/schemas/atom}"
ap = argparse.ArgumentParser(); ap.add_argument("--months", type=int, default=12); ap.add_argument("--max", type=int, default=400)
ap.add_argument("--query", default="", help="自定义 arXiv 检索式（URL 编码前的原文，不含日期）"); ap.add_argument("--tag", default="", help="输出文件后缀")
a = ap.parse_args()
end = datetime.date.today(); start = end - datetime.timedelta(days=30 * a.months)
base_q = a.query or '(all:"Schrödinger bridge" OR all:"Schrodinger bridge" OR all:"bridge matching" OR all:"diffusion bridge" OR all:"entropic optimal transport")'
q = urllib.parse.quote(base_q, safe='():') + f' AND submittedDate:[{start:%Y%m%d}0000 TO {end:%Y%m%d}2359]' 
known = set()
for f in ["metadata/papers.tsv", "metadata/extended.tsv"]:
    for r in csv.DictReader(open(ROOT / f, encoding="utf-8"), delimiter="\t"):
        known.add(r.get("id") or r.get("key"))
rows = []
for startidx in range(0, a.max, 100):
    url = ("https://export.arxiv.org/api/query?search_query=" + q.replace(" ", "+") + f"&start={startidx}&max_results=100&sortBy=submittedDate&sortOrder=descending")
    xml = None
    for t in range(4):
        try:
            xml = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "awesome-sb-scan/1.0"}), timeout=90).read(); break
        except Exception as e:
            print("  retry", t, str(e)[:60], file=sys.stderr); time.sleep(10 * (t + 1))
    if not xml: break
    entries = ET.fromstring(xml).findall("a:entry", NS)
    if not entries: break
    for e in entries:
        aid = e.find("a:id", NS).text.split("/abs/")[-1]; base = re.sub(r"v\d+$", "", aid)
        title = " ".join(e.find("a:title", NS).text.split())
        summ = " ".join((e.find("a:summary", NS).text or "").split())
        authors = [x.find("a:name", NS).text for x in e.findall("a:author", NS)]
        c = e.find(AX + "comment"); j = e.find(AX + "journal_ref")
        cats = [x.get("term") for x in e.findall("a:category", NS)]
        rows.append(dict(id=base, date=e.find("a:published", NS).text[:10], title=title, first_author=authors[0] if authors else "", n_authors=len(authors),
                         comment=(c.text.strip() if c is not None and c.text else ""), jref=(j.text.strip() if j is not None and j.text else ""),
                         cats=",".join(cats[:3]), known="yes" if base in known else "", abstract=summ))
    time.sleep(3.5)
# 相关性粗打分：标题命中 > 摘要命中
def score(r):
    t = r["title"].lower(); s = r["abstract"].lower()
    kw_t = ["schrödinger bridge", "schrodinger bridge", "bridge matching", "diffusion bridge", "entropic optimal transport", "adjoint", "sampler", "optimal control", "optimal transport", "boltzmann"]
    kw_s = ["schrödinger bridge", "schrodinger bridge", "bridge matching", "iterative markovian", "stochastic optimal control", "boltzmann", "sim-to-real", "imitation", "single-cell", "unpaired", "unnormalized", "energy", "reward", "fine-tun"]
    return 3 * sum(k in t for k in kw_t) + sum(k in s for k in kw_s)
for r in rows: r["score"] = score(r)
rows.sort(key=lambda r: (-r["score"], r["date"]), reverse=False)
tag = end.isoformat() + (f"_{a.tag}" if a.tag else "")
tsv = ROOT / f"survey/raw/S2_arxiv_scan_{tag}.tsv"
with open(tsv, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["id", "date", "score", "known", "title", "first_author", "n_authors", "comment", "jref", "cats"], delimiter="\t", lineterminator="\n", extrasaction="ignore")
    w.writeheader(); w.writerows(rows)
md = [f"# S2 · arXiv 扫描雷达（{start} → {end}）", "",
      f"检索式：SB / Schrodinger bridge / bridge matching / diffusion bridge / entropic optimal transport（标题+摘要，arXiv API），共 {len(rows)} 条；`known` = 已在 metadata 中。按标题/摘要关键词粗打分排序，仅作候选池，**未逐条核验**。", "",
      "| # | arXiv | 日期 | 分 | 已收录 | 标题 | 第一作者 | Comments（venue 线索） |", "|---|---|---|---|---|---|---|---|"]
for i, r in enumerate(rows, 1):
    md.append(f"| {i} | [{r['id']}](https://arxiv.org/abs/{r['id']}) | {r['date']} | {r['score']} | {'✓' if r['known'] else ''} | {r['title']} | {r['first_author']} | {r['comment'][:60]} |")
(ROOT / f"survey/raw/S2_arxiv_scan_{tag}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print(f"{len(rows)} 条，其中已收录 {sum(1 for r in rows if r['known'])}；高分(≥3) {sum(1 for r in rows if r['score'] >= 3)}；写入 {tsv.name}")
