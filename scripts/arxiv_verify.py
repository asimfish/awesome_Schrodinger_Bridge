#!/usr/bin/env python3
"""批量核验 arXiv ID ↔ 标题关键词，并缓存元数据到 metadata/.arxiv_meta_cache.json。
用法: python3 scripts/arxiv_verify.py ID=关键词 [ID=关键词 ...]   或   python3 scripts/arxiv_verify.py --file cands.tsv
"""
import sys, json, time, pathlib, urllib.request, xml.etree.ElementTree as ET
ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE = ROOT / "metadata/.arxiv_meta_cache.json"
NS = {"a": "http://www.w3.org/2005/Atom"}; AX = "{http://arxiv.org/schemas/atom}"

def fetch(ids, tries=5):
    url = "https://export.arxiv.org/api/query?id_list=" + ",".join(ids) + f"&max_results={len(ids)+5}"
    for k in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "awesome-sb-verify/1.0"}), timeout=60).read()
        except Exception as e:
            wait = 6 * (k + 1); print(f"  retry in {wait}s ({e})", file=sys.stderr); time.sleep(wait)
    raise SystemExit("arXiv API unavailable")

def parse(xml):
    out = {}
    for e in ET.fromstring(xml).findall("a:entry", NS):
        aid = e.find("a:id", NS).text.split("/abs/")[-1]; base = aid.rsplit("v", 1)[0] if "v" in aid[-3:] else aid
        c = e.find(AX + "comment"); j = e.find(AX + "journal_ref")
        out[base] = dict(title=" ".join(e.find("a:title", NS).text.split()),
                         authors=[a.find("a:name", NS).text for a in e.findall("a:author", NS)],
                         year=e.find("a:published", NS).text[:4],
                         comment=c.text.strip() if c is not None and c.text else "",
                         jref=j.text.strip() if j is not None and j.text else "")
    return out

def norm(s): return s.lower().replace("ö", "o").replace("$", "").replace("^", "")

def main(argv):
    cands = {}
    if argv and argv[0] == "--file":
        for line in pathlib.Path(argv[1]).read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                i, kw = line.split("\t", 1); cands[i.strip()] = kw.strip()
    else:
        for a in argv:
            i, kw = a.split("=", 1); cands[i] = kw
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    need = [i for i in cands if i not in cache]
    for k in range(0, len(need), 20):
        chunk = need[k:k + 20]
        cache.update(parse(fetch(chunk)))
        time.sleep(3.5)
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    ok, bad = [], []
    for i, kw in cands.items():
        g = cache.get(i); t = norm(g["title"]) if g else ""
        words = norm(kw).split()
        if g and all(w in t for w in words[:3]):
            ok.append(i)
        else:
            bad.append((i, kw, g["title"] if g else "NOT FOUND"))
    print(f"verified {len(ok)}/{len(cands)}")
    for b in bad: print("MISMATCH", *b, sep=" | ")
    return 0 if not bad else 1

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
