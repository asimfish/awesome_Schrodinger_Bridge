#!/usr/bin/env python3
"""从 metadata/*.tsv 渲染 README.md（awesome-ml4co 风格）。

数据源（单点事实）：
- metadata/papers.tsv     核心 25 篇：有中文精读报告 + 保版式中文译本
- metadata/extended.tsv   扩展条目：key/title/authors/venue/year/section/paper_url/code_url/note/source
- metadata/resources.tsv  代码库/基准/教程/讲义/研讨会：kind/name/url/desc/meta
用法: python3 scripts/build_readme.py [--check]
"""
import csv, pathlib, re, sys, datetime

ROOT = pathlib.Path(__file__).resolve().parents[1]
SECTIONS = [
    ("1",   "Surveys, Tutorials & Foundations", "综述、教程与基础"),
    ("2",   "Diffusion Schrödinger Bridges & Bridge Matching", "扩散薛定谔桥与桥匹配（求解器）"),
    ("2.1", "IPF / DSB / IMF / DSBM Lineage", "IPF / DSB / IMF / DSBM 谱系"),
    ("2.2", "Paired Bridges (I²SB, DDBM, DBIM)", "成对数据桥"),
    ("2.3", "Generalized, Multi-marginal, Mean-field & Unbalanced SB", "广义 / 多边缘 / 平均场 / 非平衡 SB"),
    ("2.4", "Light, Latent & Few-step Bridges", "轻量 / 隐空间 / 少步桥"),
    ("2.5", "Discrete-state Bridges", "离散状态空间桥"),
    ("2.6", "Flow Matching, Stochastic Interpolants & SB Unification", "流匹配 / 随机插值 / SB 统一"),
    ("3",   "Sampling & Stochastic Optimal Control", "采样与随机最优控制"),
    ("3.1", "Adjoint / SOC Samplers (energy-only)", "Adjoint / SOC 采样器（仅能量）"),
    ("3.2", "SOC for Reward Fine-tuning & RL (Adjoint Matching lineage)", "SOC 奖励微调与 RL（Adjoint Matching 谱系）"),
    ("3.3", "Diffusion Samplers, Boltzmann Generators & Competitors", "扩散采样器 / Boltzmann 生成器 / 竞品"),
    ("4",   "Applications", "应用"),
    ("4.1", "Image Translation, Restoration & Editing", "图像翻译 / 修复 / 编辑"),
    ("4.2", "Video, 3D, Speech, Audio & Multimodal", "视频 / 3D / 语音 / 音频 / 多模态"),
    ("4.3", "Science: Single-cell, Molecules, Chemistry & Physics", "科学：单细胞 / 分子 / 化学 / 物理"),
    ("4.4", "Embodied AI: Sim2Real, Cross-domain Transfer & RL", "具身智能：sim2real / 跨域迁移 / RL"),
    ("4.5", "Optimal Transport for Imitation & Reward", "最优传输用于模仿学习与奖励"),
    ("5",   "Codebases & Benchmarks", "代码库与基准"),
    ("6",   "Chinese Deep-dive Reports & Topic Notes", "中文精读报告与专题笔记"),
    ("7",   "Trend Report & Slides", "趋势报告与汇报"),
    ("8",   "Contributing, Citation & License", "贡献、引用与许可"),
]
SEC_TITLE = {k: (en, zh) for k, en, zh in SECTIONS}

def read_tsv(p):
    p = ROOT / p
    if not p.exists():
        return []
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))

def anchor(k):
    en = SEC_TITLE[k][0]
    s = re.sub(r"[^a-z0-9 -]", "", en.lower().replace("/", " ")).strip().replace(" ", "-")
    s = re.sub(r"-+", "-", s)
    return f"{k.replace('.', '')}-{s}"

def link(label, url):
    return f"[{label}]({url})" if url else ""

def core_entries():
    out = []
    for r in read_tsv("metadata/papers.tsv"):
        pid = r["id"]
        report = ROOT / r["report_path"]
        zh = ROOT / "papers_zh" / (pathlib.Path(r["pdf_path"]).stem + ".zh.pdf")
        out.append(dict(
            key=pid, title=r["title"], authors=r.get("authors", ""), venue=r["venue"], year=r["year"],
            section=r["section"], paper_url=r["source_url"], code_url=r.get("code_url", ""),
            project_url=r.get("project_url", ""), report=r["report_path"] if report.exists() else "",
            zh_pdf=f"papers_zh/{zh.name}" if zh.exists() else "", en_pdf=r["pdf_path"], note="", core=True))
    return out

def ext_entries():
    out = []
    for r in read_tsv("metadata/extended.tsv"):
        en = sorted((ROOT / "papers").glob(f"{r['key']}_*.pdf")); zh = sorted((ROOT / "papers_zh").glob(f"{r['key']}_*.zh.pdf"))
        out.append(dict(key=r["key"], title=r["title"], authors=r.get("authors", ""), venue=r.get("venue", ""),
                        year=r.get("year", ""), section=r["section"], paper_url=r.get("paper_url", ""),
                        code_url=r.get("code_url", ""), project_url="", report=r.get("report", ""),
                        zh_pdf=f"papers_zh/{zh[0].name}" if zh else "", en_pdf=f"papers/{en[0].name}" if en else "",
                        note=r.get("note", ""), core=False))
    return out

def format_venue(venue, year):
    """ml4co 约定：Venue, Year (note)。'NeurIPS 2024 (Spotlight)' -> 'NeurIPS, 2024 (Spotlight)'；已含逗号/复杂期刊卷期的保持原样。"""
    m = re.match(r"^([A-Za-z][A-Za-z .&+\-]*?)\s+((?:19|20)\d{2})(\s*\(.*\))?$", venue.strip())
    if m:
        return f"{m.group(1)}, {m.group(2)}{m.group(3) or ''}"
    if re.search(r"\b(19|20)\d{2}\b", venue):
        return venue
    return f"{venue}, {year}".strip(", ")

def render_entry(i, e):
    links = [link("paper", e["paper_url"]), link("code", e["code_url"]), link("project", e["project_url"]),
             link("📄 PDF", e["en_pdf"]), link("📘 精读", e["report"]), link("🀄 译本", e["zh_pdf"])]
    links = ", ".join(x for x in links if x)
    vy = format_venue(e["venue"] or "arXiv", e["year"])
    star = " ⭐" if e["core"] else ""
    s = f"{i}. **{e['title'].rstrip('.')}.** {vy}. {links}{star}\n"
    if e["authors"]:
        s += f"\n *{e['authors']}*\n"
    if e["note"]:
        s += f"\n > {e['note']}\n"
    return s

def sort_key(e):
    return (e["year"] or "0", e["key"])

def build():
    core = core_entries()
    ext = ext_entries()
    by_sec = {}
    for e in core + ext:
        by_sec.setdefault(e["section"], []).append(e)
    res = read_tsv("metadata/resources.tsv")
    n_core, n_ext = len(core), len(ext)
    n_zh = sum(1 for e in core + ext if e["zh_pdf"])
    n_rep = sum(1 for e in core if e["report"])
    topics = sorted((ROOT / "topics").glob("E*.md"))
    today = datetime.date.today().isoformat()

    L = []
    L.append("# Awesome Schrödinger Bridge\n")
    L.append("[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) "
             f"![papers](https://img.shields.io/badge/papers-{n_core + n_ext}-blue) "
             f"![中文精读](https://img.shields.io/badge/%E4%B8%AD%E6%96%87%E7%B2%BE%E8%AF%BB-{n_rep}-orange) "
             f"![中文译本](https://img.shields.io/badge/%E4%B8%AD%E6%96%87%E8%AF%91%E6%9C%AC%20PDF-{n_zh}-red) "
             "![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen) "
             f"![last update](https://img.shields.io/badge/last%20update-{today}-lightgrey)\n")
    L.append("A curated list of papers, code, tutorials and **Chinese deep-dive reports** on the **Schrödinger Bridge (SB)** "
             "problem and its modern incarnations: diffusion Schrödinger bridges and bridge matching, generalized / "
             "multi-marginal / unbalanced SB, adjoint & stochastic-optimal-control samplers, and their applications in "
             "generative modeling, scientific data, and embodied AI (sim2real, cross-domain imitation).\n")
    L.append("本仓库系统整理薛定谔桥（Schrödinger Bridge）方向的论文与资源。**核心论文每篇配有中文精读报告（`reports/`）与保版式中文译本 PDF（`papers_zh/`，"
             "由 [SuperTranslate](https://github.com/asimfish/super_translate) 生成并经视觉 QA）**；20 份专题笔记（`topics/`）梳理方法谱系与基线；"
             "2025–2026 趋势调研与洞见见 `survey/`，汇报 PPT（HTML / PDF / Beamer）见 `slides/`。\n")
    L.append("*Maintained by [asimfish](https://github.com/asimfish). Entries marked ⭐ are core papers with full Chinese reports and translated PDFs. "
             "Venues are verified against arXiv comments / OpenReview / proceedings; preprints are labelled `arXiv`. "
             "Contributions are welcome — see [Contributing](#8-contributing-citation-license).*\n")
    L.append("**Legend**: `paper` arXiv/publisher page · `code` official implementation · `project` project page · "
             "`📄 PDF` English PDF in repo · `📘 精读` Chinese deep-dive report · `🀄 译本` layout-preserving Chinese PDF\n")
    L.append("## [Content](#content)\n")
    for k, en, zh in SECTIONS:
        indent = "" if "." not in k else "&emsp;"
        L.append(f"{indent}{k}. [{en}](#{anchor(k)}) — {zh}  ")
    L.append("")

    for k, en, zh in SECTIONS:
        head = "###" if "." in k else "##"
        L.append(f"<a name=\"{anchor(k)}\"></a>")
        L.append(f"{head} [{k}. {en}](#content)")
        L.append(f"*{zh}*\n")
        if k == "1":
            blk = render_resources(res, kinds=("survey", "tutorial", "lecture"))
            if "to be added" not in blk:
                L.append(blk)
            for i, e in enumerate(sorted(by_sec.get(k, []), key=sort_key), 1):
                L.append(render_entry(i, e))
        elif k == "5":
            L.append(render_resources(res, kinds=("code", "benchmark", "workshop")))
        elif k == "6":
            L.append(render_reports(core, topics))
        elif k == "7":
            L.append(render_trend())
        elif k == "8":
            L.append(render_contrib())
        elif k in ("2", "3", "4"):
            pass
        else:
            items = sorted(by_sec.get(k, []), key=sort_key)
            if not items:
                L.append("_(to be added)_\n")
            for i, e in enumerate(items, 1):
                L.append(render_entry(i, e))
    (ROOT / "README.md").write_text("\n".join(L).rstrip() + "\n", encoding="utf-8")
    return n_core, n_ext, n_zh, n_rep, len(res)

def render_resources(res, kinds):
    items = [r for r in res if r.get("kind") in kinds]
    if not items:
        return "_(to be added)_\n"
    names = {"survey": "Surveys", "tutorial": "Tutorials & Lecture Notes", "lecture": "Courses, Talks & Blogs",
             "code": "Codebases", "benchmark": "Benchmarks & Datasets", "workshop": "Workshops & Communities"}
    L = []
    for kind in kinds:
        sub = [r for r in items if r["kind"] == kind]
        if not sub:
            continue
        L.append(f"**{names[kind]}**\n")
        for i, r in enumerate(sub, 1):
            meta = f" — {r['meta']}" if r.get("meta") else ""
            L.append(f"{i}. [{r['name']}]({r['url']}){meta}  \n   {r['desc']}")
        L.append("")
    return "\n".join(L) + "\n"

def render_reports(core, topics):
    L = ["Every core paper has a Chinese deep-dive report (基本信息 / 一句话总结 / 方法核心 / 实验与结果 / 局限性 / 与相关方向的关系) "
         "and a layout-preserving Chinese translation. Start from [reports/INDEX.md](reports/INDEX.md) and the synthesis documents:\n",
         "- [综合文献地图：OT / SB 如何迁移具身跨域数据](reports/synthesis.md)",
         "- [Adjoint / Generalized / Structured Schrödinger Bridge 扩展文献综述](reports/sb_adjoint_extended_synthesis.md)",
         "- [SB × OT × Sim2Real：深度调研、前沿论文与学习资源导航](reports/deep_research_learning_resources.md)",
         "- [Guan-Horng Liu 研究工作专题：从最优控制到 SB、Adjoint Sampling 与 LLM Post-training](reports/guan_horng_liu_research_roadmap.md)\n",
         "| arXiv | Paper | Venue | 精读 | 英文 PDF | 中文译本 |", "|---|---|---|---|---|---|"]
    for e in sorted(core, key=sort_key):
        L.append(f"| {e['key']} | {e['title']} | {e['venue']} | "
                 f"{link('📘', e['report']) or '—'} | {link('📄', e['en_pdf'])} | {link('🀄', e['zh_pdf']) or '⏳'} |")
    L.append("\n**Topic notes (`topics/`, 20 份专题笔记)** — 方法谱系、基线协议与评测方案：\n")
    for p in topics:
        first = p.read_text(encoding="utf-8").splitlines()[0].lstrip("# ").strip()
        L.append(f"- [{p.stem.split('_')[0]}]({'topics/' + p.name}) {first}")
    return "\n".join(L) + "\n"

def render_trend():
    return ("- **Trend report (2025–2026)**: [survey/SB_TREND_REPORT_2026.md](survey/SB_TREND_REPORT_2026.md) · "
            "[PDF](survey/SB_TREND_REPORT_2026.pdf) — 五条主线的进展盘点、证据表与 insight。\n"
            "- **Raw survey notes**: [survey/raw/](survey/raw/) — WebSearch 证据记录（`S1_*`）与 arXiv 近 12 个月扫描雷达表（`S2_*`，277 篇候选，含未入选项）；复扫命令 `python3 scripts/arxiv_scan.py --months 12`。\n"
            "- **Slides**: [slides/awesome_sb_report.html](slides/awesome_sb_report.html)（HTML，←/→ 翻页，可打印）· "
            "[slides/awesome_sb_report.pdf](slides/awesome_sb_report.pdf) · Beamer 版 [slides/beamer/awesome_sb_beamer.pdf](slides/beamer/awesome_sb_beamer.pdf)\n")

def render_contrib():
    return ("**Contributing** — PRs are welcome. Add a row to `metadata/extended.tsv` (or `resources.tsv`) and run "
            "`python3 scripts/build_readme.py`; please verify the venue on arXiv/OpenReview and link the official code when it exists. "
            "Chinese reports follow the template in `reports/` (基本信息 → 一句话总结 → 方法核心 → 实验与结果 → 局限性 → 关系).\n\n"
            "**Citation**\n\n```bibtex\n@misc{awesome_schrodinger_bridge,\n  title  = {Awesome Schr\\\"odinger Bridge: Papers, Code, Chinese Deep-dive Reports and Translations},\n"
            "  author = {Li, Yufeng},\n  year   = {2026},\n  howpublished = {\\url{https://github.com/asimfish/awesome_Schrodinger_Bridge}}\n}\n```\n\n"
            "**License** — Curated text, reports and slides are released under [CC BY 4.0](LICENSE); scripts under MIT. "
            "Paper PDFs in `papers/` are the arXiv versions (see each paper's arXiv license); translated PDFs in `papers_zh/` are derivative "
            "works provided for non-commercial research use only — please cite the original papers.\n\n"
            "**Acknowledgements** — Format inspired by [awesome-ml4co](https://github.com/Thinklab-SJTU/awesome-ml4co). Translations by "
            "[SuperTranslate](https://github.com/asimfish/super_translate); writing polished with "
            "[shuorenhua](https://github.com/MrGeDiao/shuorenhua) and [anti-defensive-writing](https://github.com/Kiterlin/anti-defensive-writing); "
            "reports organised in the spirit of [PaperOrchestra](https://github.com/Ar9av/PaperOrchestra); slides built on "
            "[ppt-master](https://github.com/hugohe3/ppt-master) design tokens and [beamer-skill](https://github.com/Noi1r/beamer-skill).\n")

if __name__ == "__main__":
    n = build()
    print("README.md 已生成：core=%d ext=%d zh_pdf=%d reports=%d resources=%d" % n)
