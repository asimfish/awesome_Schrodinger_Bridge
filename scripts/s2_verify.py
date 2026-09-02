#!/usr/bin/env python3
"""用 Semantic Scholar 批量核验 arXiv ID（title/year/venue/authors）并按标题搜索缺 ID 的论文；结果并入 metadata/.s2_meta_cache.json。
用法: python3 scripts/s2_verify.py --ids 2604.05673 2601.22408 ... [--search "标题一" "标题二" ...]
"""
import json, sys, time, pathlib, urllib.request, urllib.parse, argparse
ROOT = pathlib.Path(__file__).resolve().parents[1]; CACHE = ROOT / "metadata/.s2_meta_cache.json"
H = {"Content-Type": "application/json", "User-Agent": "awesome-sb/1.0"}; F = "fields=title,year,venue,authors,externalIds"

def get(url, data=None, tries=5):
    for k in range(tries):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(url, data=data, headers=H), timeout=60))
        except Exception as e:
            print("  retry", k, str(e)[:60], file=sys.stderr); time.sleep(5 * (k + 1))
    return None

def rec(p):
    return dict(title=p["title"], year=p.get("year"), venue=p.get("venue") or "", authors=[a["name"] for a in p.get("authors", [])])

ap = argparse.ArgumentParser(); ap.add_argument("--ids", nargs="*", default=[]); ap.add_argument("--search", nargs="*", default=[])
a = ap.parse_args()
cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
need = [i for i in a.ids if i not in cache]
for k in range(0, len(need), 40):
    chunk = need[k:k + 40]
    res = get("https://api.semanticscholar.org/graph/v1/paper/batch?" + F, json.dumps({"ids": ["ARXIV:" + i for i in chunk]}).encode()) or []
    for i, r in zip(chunk, res):
        if r: cache[i] = rec(r)
    time.sleep(2)
for i in a.ids:
    v = cache.get(i); print(i, "|", (v or {}).get("year"), "|", ((v or {}).get("venue") or "-")[:38], "|", (v or {}).get("title", "NOT FOUND")[:72])
for q in a.search:
    r = get("https://api.semanticscholar.org/graph/v1/paper/search?query=" + urllib.parse.quote(q) + "&limit=3&" + F)
    print("Q:", q[:70])
    for p in (r or {}).get("data", []):
        ax = (p.get("externalIds") or {}).get("ArXiv", "-")
        print("   ", ax, "|", p.get("year"), "|", (p.get("venue") or "-")[:30], "|", p["title"][:72])
        if ax != "-" and ax not in cache: cache[ax] = rec(p)
    time.sleep(1.5)
CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8"); print("cached:", len(cache))
