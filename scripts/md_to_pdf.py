#!/usr/bin/env python3
"""Markdown → 单文件 HTML（学术浅色风）→ Chrome headless PDF。用法: python3 scripts/md_to_pdf.py IN.md OUT.pdf [--title T]"""
import sys, re, pathlib, subprocess, markdown, argparse
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CSS = """
body{font-family:'PingFang SC','Noto Sans CJK SC','Hiragino Sans GB','Helvetica Neue',Arial,sans-serif;color:#222;max-width:960px;margin:0 auto;padding:36px 44px;line-height:1.65;font-size:14.5px}
h1{color:#003366;border-left:8px solid #CC0000;padding-left:14px;font-size:28px;margin-top:0}
h2{color:#003366;border-bottom:2px solid #0066CC;padding-bottom:4px;margin-top:34px;font-size:21px;page-break-after:avoid}
h3{color:#003366;font-size:17px;margin-top:22px;page-break-after:avoid}
blockquote{background:#EEF3F8;border-left:5px solid #0066CC;margin:12px 0;padding:8px 14px;color:#334}
table{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0}
tr{page-break-inside:avoid}
th{background:#003366;color:#fff;text-align:left;padding:6px 8px}td{padding:5px 8px;border-bottom:1px solid #D6DEE8;vertical-align:top}
tr:nth-child(even) td{background:#F7F9FC}
code{background:#F1F4F8;padding:1px 5px;border-radius:3px;font-size:12.5px;font-family:Menlo,Consolas,monospace}
img,svg{max-width:100%}
a{color:#0066CC;text-decoration:none}
hr{border:0;border-top:1px solid #D6DEE8;margin:26px 0}
ol li,ul li{margin:4px 0}
@page{size:A4;margin:16mm 14mm}
"""
ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("out"); ap.add_argument("--title", default="")
a = ap.parse_args(); src = pathlib.Path(a.src).resolve(); out = pathlib.Path(a.out).resolve()
md = src.read_text(encoding="utf-8")
# 内嵌本地 SVG 图（相对路径）：先占位，转换后回填，避免被 markdown 解析
svgs = []
def hold(m):
    p = (src.parent / m.group(1)).resolve()
    if p.exists() and p.suffix == ".svg":
        s = p.read_text(encoding="utf-8")
        s = re.sub(r'(<svg[^>]*?)\s+width="[^"]*"\s+height="[^"]*"', r'\1 style="width:100%;height:auto"', s, 1)
        svgs.append(s); return f"@@SVG{len(svgs) - 1}@@"
    return m.group(0)
md = re.sub(r"!\[[^\]]*\]\(([^)]+\.svg)\)", hold, md)
body = markdown.markdown(md, extensions=["tables", "fenced_code", "toc"])
for i, s in enumerate(svgs):
    body = body.replace(f"<p>@@SVG{i}@@</p>", f'<div style="margin:14px 0">{s}</div>').replace(f"@@SVG{i}@@", s)
title = a.title or re.search(r"^# (.+)$", md, re.M).group(1)
html = f"<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'><title>{title}</title><style>{CSS}</style></head><body>{body}</body></html>"
tmp = out.with_suffix(".html"); tmp.write_text(html, encoding="utf-8")
subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=3000",
                f"--print-to-pdf={out}", f"file://{tmp}"], check=False, capture_output=True)
print("wrote", out, out.stat().st_size // 1024, "KB;", "html:", tmp.name)
