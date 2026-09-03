#!/usr/bin/env python3
"""生成 slides/awesome_sb_report.html（18 页，academic_defense 视觉系统）。数字实时来自 metadata/*.tsv 与 papers_zh/QA_REPORT.md。"""
import csv, json, glob, pathlib, re, datetime
ROOT = pathlib.Path(__file__).resolve().parents[1]
core = list(csv.DictReader(open(ROOT / "metadata/papers.tsv", encoding="utf-8"), delimiter="\t"))
ext = list(csv.DictReader(open(ROOT / "metadata/extended.tsv", encoding="utf-8"), delimiter="\t"))
res = list(csv.DictReader(open(ROOT / "metadata/resources.tsv", encoding="utf-8"), delimiter="\t"))
n_core, n_ext, n_new = len(core), len(ext), sum(1 for r in ext if r["source"] == "survey2026")
n_zh = len(glob.glob(str(ROOT / "papers_zh/*.zh.pdf"))); n_topics = len(glob.glob(str(ROOT / "topics/E*.md")))
qa = (ROOT / "papers_zh/QA_REPORT.md").read_text(encoding="utf-8") if (ROOT / "papers_zh/QA_REPORT.md").exists() else ""
m = re.search(r"通过 (\d+) · 通过（有备注）(\d+) · 需复核 (\d+)", qa); q_pass, q_note, q_fail = (m.groups() if m else ("?", "?", "?"))
n_code = sum(1 for r in core if r["code_url"]) + sum(1 for r in ext if r["code_url"])
svg = (ROOT / "assets/sb_lineage.svg").read_text(encoding="utf-8")
svg = re.sub(r'viewBox="0 0 1280 600" width="1280" height="600"', 'viewBox="0 70 1280 500" width="1200" height="560" overflow="hidden" preserveAspectRatio="xMidYMid meet"', svg, 1)
svg = re.sub(r'<text x="40" y="38"[^<]*</text>\s*<text x="40" y="60"[^<]*</text>', '', svg, 1)
today = datetime.date.today().isoformat()

S = []
def content(title, key, body, sec, src=""):
    S.append(f'<section class="slide"><div class="hdr"><h1>{title}</h1><span class="sec">{sec}</span></div>'
             f'<div class="key">{key}</div><div class="body">{body}</div>'
             + (f'<div class="src">来源：{src}</div>' if src else "") +
             f'<div class="ftr"><span>awesome_Schrödinger_Bridge · 调研汇报 {today}</span><span>{sec}</span><span class="pn"></span></div></section>')
def chapter(num, title, desc):
    S.append(f'<section class="slide chapter"><div class="num">{num}</div><div class="vbar"></div><h1>{title}</h1><p>{desc}</p><div class="rule"></div></section>')

# 1 cover
S.append(f'''<section class="slide cover"><div class="bar"></div><div class="vbar"></div>
<h1>Awesome Schrödinger Bridge<br><span style="font-size:40px;color:#CC0000">从经典 SB 到 2026：求解器、控制、离散与具身应用</span></h1>
<h2>{n_core} 篇核心论文中文精读 + 保版式译本 · {n_topics} 份专题笔记 · {n_core + n_ext} 条主题清单 · 2025–2026 趋势与 insight</h2>
<div class="meta">汇报人：asimfish（Yufeng Li）<br>仓库：github.com/asimfish/awesome_Schrodinger_Bridge<br>日期：{today}</div>
<div class="foot">格式参照 awesome-ml4co · 翻译 SuperTranslate · 版式 ppt-master academic_defense · 写作规范 shuorenhua / anti-defensive-writing</div></section>''')
# 2 toc
content("目录", "本次汇报回答三个问题：交付了什么、SB 领域到 2026 年走到了哪、对具身 sim2real 意味着什么。",
 '''<div class="toc">
<div><b>1 · 仓库交付物</b><span>论文清单、精读、译本 QA、专题、脚本与复现</span></div>
<div><b>2 · 方法谱系全景</b><span>1932 → 2026：经典 / IPF / IMF / 分支 / 应用</span></div>
<div><b>3 · 五条主线的 2024H2–2026 进展</b><span>求解器 · 结构化 · 离散 · SOC 采样 · 应用面</span></div>
<div><b>4 · Insight</b><span>八条判断，每条附证据指针</span></div>
<div><b>5 · 对具身 sim2real 的启示</b><span>范式选择 · 基线 · 评测</span></div>
<div><b>6 · 观察清单与方法说明</b><span>未来 12 个月判据 · 核验口径 · 局限</span></div></div>''', "目录")
# 3 chapter 1
chapter("01", "仓库交付物", "一个可复现的 awesome 仓库：清单 + 精读 + 译本 + 专题 + 趋势报告 + 汇报，全部脚本可重建。")
# 4 overview
content("仓库总览：数字与结构", f"{n_core} 篇核心论文每篇有中文精读与保版式译本；{n_core + n_ext} 条主题清单按 5 大类 / 14 小类组织，venue 全部核验。",
 f'''<div class="kpi"><div><b>{n_core + n_ext}</b><span>论文条目（核心 {n_core} + 扩展 {n_ext}）</span></div><div><b>{n_core}</b><span>中文精读报告（reports/）</span></div><div><b>{n_zh}</b><span>保版式中文译本 PDF（papers_zh/）</span></div><div><b>{n_topics}</b><span>专题笔记（topics/ E01–E20）</span></div></div>
<div class="cols" style="margin-top:26px;height:auto"><div class="card"><h3>目录结构</h3><ul>
<li><code>README.md</code>：awesome-ml4co 风格清单，脚本生成</li><li><code>papers/</code> 英文 PDF · <code>papers_zh/</code> 中文译本 + QA 表</li>
<li><code>reports/</code> 逐篇精读 + 4 份综合 · <code>topics/</code> 20 专题</li><li><code>survey/</code> 趋势报告（md/pdf）+ 原始调研 · <code>slides/</code> 本汇报（html/pdf/beamer）</li>
<li><code>metadata/*.tsv</code> 单点事实源 · <code>scripts/</code> 生成与核验脚本</li></ul></div>
<div class="card red"><h3>纪律</h3><ul><li>venue 只写有证据的：arXiv Comments / OpenReview / proceedings 页；否则标 <code>arXiv</code></li>
<li>代码链接逐个 <code>gh api</code> 核验（{n_code} 条）</li><li>作者不确定即留空，不编</li><li>数字均有来源指针（R:/E:/arXiv:）</li></ul></div></div>''', "1 · 交付物", "metadata/papers.tsv、extended.tsv、resources.tsv；README.md")
# 5 translation QA
content("保版式中文译本与视觉 QA", f"{n_zh} 篇译本全部产出；QA：通过 {q_pass} · 通过（有备注）{q_note} · 需人工复核 {q_fail}。错误主要是公式碎片保留英文与个别字号缩放。",
 f'''<div class="cols"><div class="card"><h3>流程</h3><ul><li>引擎：SuperTranslate（DeepSeek 后端，<code>--preserve-graphics-text</code>）</li>
<li>公式 / 图表 / 参考文献 / 双栏原样保留，图内文字不翻</li><li>每篇自动 <code>inspect</code>：页数一致、图像/公式丢失、文字重叠、字号漂移</li>
<li>失败篇目：复用翻译缓存只重请求未译块，避免重复计费</li><li>3 路并行、tmux 托管、可断点续跑（<code>scripts/translate_batch.sh</code>）</li></ul></div>
<div class="card red"><h3>QA 口径（papers_zh/QA_REPORT.md）</h3><ul><li><b>通过</b>：0 个 error</li>
<li><b>通过（有备注）</b>：仅字号类 error 或 ≤12 词公式碎片保留英文（如 <code>is, T = T_aff[X, Y].</code>）</li>
<li><b>需人工复核</b>：存在 &gt;12 词成段未译或表格结构错位——已列出页码与位置</li>
<li>译本是研究辅助读物，引用请以英文原文为准</li></ul></div></div>''', "1 · 交付物", "papers_zh/QA_REPORT.md；logs/translate/")
# 6 chapter 2
chapter("02", "方法谱系全景", "SB 的三次范式切换：IPF → IMF → 一次训练；SOC 成为桥与采样的统一语言；离散与结构化是 2025–26 的增量。")
# 7 lineage
S.append(f'<section class="slide"><div class="hdr"><h1>Schrödinger Bridge 方法族谱系（1932 → 2026）</h1><span class="sec">2 · 谱系</span></div>'
         f'<div style="position:absolute;left:40px;top:84px;width:1200px;height:560px">{svg}</div>'
         f'<div class="ftr"><span>awesome_Schrödinger_Bridge · 调研汇报 {today}</span><span>assets/sb_lineage.svg · 红标 = 2025–2026 新增</span><span class="pn"></span></div></section>')
# 8 chapter 3
chapter("03", "五条主线的 2024H2–2026 进展", "求解器 · 结构化 SB · 离散状态空间 · SOC 采样与微调 · 应用面。每页一个 takeaway。")
# 9 solvers
content("主线 1 · 求解器：IPF → IMF → 一次训练与少步", "Takeaway：IMF 已是默认；竞争转到 ε 谱统一与免训练闭式采样。SB 与 FM 是一条连续谱的两端。",
 '''<div class="cols3"><div class="card"><h3>IPF → IMF</h3><ul><li>DSB（NeurIPS 21）：交替回归 score，误差累积</li><li>DSBM（NeurIPS 23）：Markov ↔ reciprocal 投影，两边缘始终保持</li><li>IDBM：第一次迭代已是合法 transport</li></ul></div>
<div class="card red"><h3>一次训练 / ε 谱</h3><ul><li>SB Flow（NeurIPS 24 Spotlight）：α-IMF 在线微调，α=1 退回 IMF</li><li><b>RSBM（2026）</b>：速度场在 ε∈(0,1] 上结构不变，降 ε 线性减方差；3 步 92% 成功率，3.8× 更少 NFE，无蒸馏</li></ul></div>
<div class="card"><h3>少步采样</h3><ul><li>训练侧：CDBM consistency、LBM 1-NFE latent</li><li>采样侧：DBIM 隐式；<b>UniDB++</b> 精确闭式反向解，5–10 步、最多 20×</li><li>理论收口：统一 bridge 框架（2025）、SB 基础指南（2026）</li></ul></div></div>
<p style="margin-top:22px;font-size:17px;color:#666">工程含义：先在大 ε 下借 SB 的边缘保持学粗耦合，再向小 ε 走换直轨迹与少步；缺的是自动选 ε 的准则。</p>''',
 "3 · 进展", "arXiv:2303.16852、R:2409.09347、arXiv:2604.05673、arXiv:2505.21528、arXiv:2503.21756、E01/E04")
# 10 structured
content("主线 2 · 结构化 SB：把领域知识写进路径", "Takeaway：多边缘、分叉、反馈、非平衡、函数空间——结构先验决定生物与科学应用能否落地。",
 '''<table><tr><th>结构</th><th>代表工作</th><th>关键点</th><th>证据</th></tr>
<tr><td>任务代价</td><td>GSBM（ICLR 24）</td><td>路径动能之外加可微状态代价：keypoint / depth / 逆动力学 / 安全约束</td><td>R:2310.02233</td></tr>
<tr><td>多边缘</td><td>3MSBM（NeurIPS 25）· <span class="tag red">MSBM 2025</span></td><td>相空间测度值样条 vs 逐区间局部 SB + 共享全局参数化；两种构造并存，未收敛</td><td>R:2506.10168 · arXiv:2510.16587</td></tr>
<tr><td>分叉</td><td><span class="tag red">BranchSBM（ICLR 26）</span></td><td>多条速度场 + 分支增长网络 = Unbalanced CondSOC 之和；单分支 SBM 在细胞命运分叉上模式塌缩；扩展到 150 PCs</td><td>ICLR 2026 项目页</td></tr>
<tr><td>反馈 / 非平衡</td><td>FSBM（ICLR 25 Oral）· UDSB · <span class="tag red">CytoBridge（NeurIPS 25）</span></td><td>闭环反馈控制求 SB；质量不守恒；平均场 + 非平衡 + 相互作用四网络建模</td><td>arXiv:2410.14055 · 2306.09099 · 2505.11197</td></tr>
<tr><td>约束域</td><td><span class="tag red">Reflected SBM（2026）</span></td><td>反射布朗运动参考过程 + IMF，样本保证在数据域内，开销可忽略</td><td>arXiv:2607.03626</td></tr>
<tr><td>函数空间</td><td>SOC in function spaces（NeurIPS 24）→ FAS（ICML 26）</td><td>把 adjoint 采样推到无限维</td><td>R:2511.06239</td></tr></table>
<div class="card red" style="margin-top:18px"><h3>判断</h3><p>结构先验（多边缘 / 分叉 / 非平衡）缺一，单细胞任务就会模式塌缩或质量守恒失真；3MSBM 与 MSBM 同年给出两种不同的多边缘构造，说明这个子问题尚未收敛。</p></div>''',
 "3 · 进展", "见表内证据列")
# 11 discrete
content("主线 3 · 离散状态空间：从翻译到采样与微调", "Takeaway：2025–26 增长最快的分支。驱动力是 dLLM 的兴起与 ground truth 基准的出现。",
 '''<div class="cols"><div class="card"><h3>理论与基准</h3><ul><li>DDSBM（ICLR 25）：连续时间 CTMC 图变换</li><li>CSBM（ICML 25）：离散时间 IMF 收敛性，覆盖 VQ 码本 / token / 原子类别</li><li><span class="tag red">catsbench（ICLR 26）</span>：有解析解的离散分布对——离散 SB 第一次能被严格评测；副产品 DLightSB / DLightSB-M / α-CSBM</li></ul></div>
<div class="card red"><h3>应用 / 微调 / 采样</h3><ul><li><span class="tag red">MadSBM</span>：肽序列 = 编辑图上的受控 CTMC，参考过程来自 ESM-2 logits；首次离散 classifier guidance</li><li><span class="tag red">DAM（ICLR 26）</span>：离散 adjoint 估计量，微调 LLaDA-8B 做数学推理</li><li><span class="tag red">DASBS（ICML 26）</span>：AM 机制与状态空间无关；循环群结构是必要条件</li><li><span class="tag red">MDNS</span>：CTMC 随机控制训练 masked 扩散采样器（Ising / Potts）</li></ul></div></div>
<p style="margin-top:20px;font-size:17px;color:#666">下一步竞争在参考过程的选择：MadSBM 用语言模型 logits 当参考，是把领域先验塞进 SB 的范例。</p>''',
 "3 · 进展", "ICML 2025 PMLR 267；ICLR 2026 poster 页；arXiv:2601.22408、2508.10684；R:2602.07132、R:2602.08243")
# 12 SOC
content("主线 4 · SOC 采样与微调：Adjoint 谱系的三年", "Takeaway：Adjoint Matching 把 SOC 变成回归，衍生出整条采样 / 微调谱系；2026 年 SMP 给它补上严格地基。",
 '''<div class="cols"><div class="card"><h3>谱系</h3><ul><li>源头：PIS / DDS / CMCD——全轨迹反传、on-policy 耦合、先验受限三大瓶颈（E14）</li>
<li>转折：Adjoint Matching（ICLR 25 Spotlight）memoryless 调度 + lean adjoint 回归</li>
<li>AS（ICML 25）：梯度更新数 ≫ 能量评估数，SPICE 构象 amortized 采样</li><li>ASBS（NeurIPS 25 Oral）：任意 source 分布 · <span class="tag red">NAAS</span>（NeurIPS 25）退火参考动力学 · <span class="tag red">WT-ASBS</span>（ICLR 26）metadynamics 偏置进 ASBS</li><li>FAS：函数空间 · DAM / DASBS：离散</li><li><span class="tag red">2026H1 横向扩张</span>QAM → TRQAM → ME-AM → MaxEnt-AM → Reinforce AM（RL）· CAM（组合优化，ICML 26）· MFC</li></ul></div>
<div class="card red"><h3>竞品、评测与理论</h3><ul><li>竞品 iDEM / NETS / Sendera："无偏 + 全模态覆盖"仍是 adjoint 线短板；评测需补 EUBO、前向指标、mode-coverage（E15）</li>
<li><span class="tag red">AM via SMP（2026）</span>：一般 Hamiltonian adjoint matching 目标，其期望一阶变分与 SOC 目标一致；lean adjoint = 状态无关扩散特例；AM = 连续时间逐次逼近法</li></ul></div></div>''',
 "3 · 进展", "R:2504.11713、R:2506.22565、R:2511.06239、R:2602.07132、R:2602.08243；arXiv:2604.08580；E06/E14/E15")
# 13 apps image
content("主线 5a · 应用：图像修复与翻译", "Takeaway：Doob h-transform 类桥是 SOC 终端罚 →∞ 的特例（UniDB）——这解释了过度平滑，也给出修法。",
 '''<div class="cols3"><div class="card"><h3>成对桥的演进</h3><ul><li>I²SB（ICML 23）：2–10 NFE vs Palette ~100</li><li>DDBM（ICLR 24）→ DBIM（ICLR 25）隐式采样</li><li><span class="tag red">UniDB（ICML 25）</span>：闭式最优控制器 + 可调终端罚，细节保真更好</li><li><span class="tag red">UniDB++</span>：免训练闭式加速</li><li><span class="tag red">RDBM / Bi-Bridge（CVPR 26）</span>：残差调制只扰动退化区（+1.55 dB）；双向一致性</li></ul></div>
<div class="card"><h3>unpaired 与少步</h3><ul><li>UNSB（ICLR 24）神经 SB + 对抗正则</li><li>ASBM（NeurIPS 24）对抗式 D-IMF</li><li>LBM（ICCV 25）SD latent 上 1 NFE</li></ul></div>
<div class="card red"><h3>失败模式（E17）</h3><ul><li>DDIB 两段桥拼接 ≠ 跨域 OT</li><li>精确 cycle consistency ≠ 对齐</li><li>语义漂移是机器人数据上最危险的失败</li></ul></div></div>''',
 "3 · 进展", "R:2302.05872；ICML 2025 PMLR 267 zhu25o；arXiv:2505.21528；E16/E17")
# 14 apps embodied
content("主线 5b · 应用：具身智能", "Takeaway：SB 的角色从\u201c数据翻译器\u201d转向\u201c策略本身\u201d；三种范式不可平替，耦合方式必须显式定义。",
 '''<table><tr><th>范式</th><th>代表</th><th>做什么</th><th>产出</th></tr>
<tr><td>表示对齐</td><td>EgoBridge（NeurIPS 25 + CoRL 25 Oral）· Guided OT co-training（NeurIPS 25）</td><td>在 joint feature-action 分布上 OT / UOT 对齐</td><td>只加对齐损失，不产数据</td></tr>
<tr><td>生成式 transport</td><td>SB Flow · BDGxRL（DSB 对齐源/目标域转移动力学 + 奖励调制）</td><td>翻译观测或 (s,a,s′)</td><td>产出可训练数据</td></tr>
<tr><td><span class="tag red">策略即桥</span></td><td>BridgePolicy（ICML 26）· RSBM（2026）</td><td>观测嵌入 SDE、从观测先验采样动作；ε 谱少步导航</td><td>52 仿真 + 5 真机任务优于生成式策略；3 步 92%</td></tr></table>
<div class="card red" style="margin-top:18px"><h3>缺口</h3><p>R10 审查：全库无真机评估统计协议（试次数 / 种子 / 置信区间）；E11 基于 SimplerEnv 给出四层指标 + 两档协议，功效约 170 rollouts/臂。2026 新论文里缺口依旧——判断。</p></div>''',
 "3 · 进展", "R:2509.19626、R:2509.18631、R:2602.23737；arXiv:2512.07212、2604.05673；synthesis §4.4；E11")
# 15 apps science
content("主线 5c · 应用：科学计算", "Takeaway：单细胞轨迹推断是结构化 SB 的主战场；分子构象是 Adjoint 线的主战场；物理 unfolding 证明少量真实数据下的稳定性。",
 '''<div class="cols3"><div class="card"><h3>单细胞 / 生物</h3><ul><li>TrajectoryNet（ICML 20）→ DMSB → 3MSBM → MSBM → <span class="tag red">BranchSBM</span></li><li>Aligned DSB（UAI 23）、UDSB</li><li>MadSBM 肽设计</li></ul></div>
<div class="card"><h3>分子 / 化学</h3><ul><li>React-OT（Nat. Mach. Intell. 25）：确定性 OT 生成过渡态</li><li>AS / ASBS：SPICE 构象 amortized 采样</li><li>iDEM / NETS：Boltzmann 采样基准 DW4 / LJ13 / LJ55</li></ul></div>
<div class="card"><h3>物理 / 语音</h3><ul><li>SBUnfold（PRD 24）：60 万仿真 + 1000 伪数据仍稳定</li><li>Bridge-TTS、SB 语音增强（Interspeech 24）</li></ul></div></div>''',
 "3 · 进展", "R:2404.13430、R:2308.12351、R:2506.10168；arXiv:2510.16587、2601.22408；ICLR 2026 项目页")
# 16 chapter 4
chapter("04", "Insight", "八条判断。每条注明证据指针；标\u201c判断\u201d处为本报告的推断，不是文献结论。")
# 17 I1-I4
content("Insight I1–I4", "ε 是主旋钮 · SOC 是统一语言 · 离散靠 dLLM 与基准驱动 · 结构先验决定生物应用成败",
 '''<div class="cols"><div class="card"><h3>I1 · ε（σ）已成为方法选择的主旋钮</h3><p>SB Flow 的 α、[SF]²M 的 σ、RSBM 的 ε、E04 的 ε=2σ² 说的是同一件事：SB 与 FM 是一条谱的两端，同一网络覆盖全谱。缺的是自动选 ε 的准则——判断。</p></div>
<div class="card"><h3>I2 · SOC 合并了\u201c桥\u201d与\u201c控制\u201d两套语言</h3><p>UniDB：h-transform = 终端罚→∞；AM 把 SOC 求解变回归；SMP 补地基。\u201c参考过程 + 终端罚 + 回归目标\u201d成为统一设计三元组。</p></div>
<div class="card red"><h3>I3 · 离散 SB 的爆发有两个驱动</h3><p>dLLM 兴起（DAM / DASBS / MDNS）与 ground truth 出现（catsbench）。下一步竞争在参考过程选择（MadSBM 用 ESM-2 logits）。</p></div>
<div class="card red"><h3>I4 · 结构先验决定生物应用成败，而非求解器精度</h3><p>多边缘、分叉、非平衡缺一即模式塌缩或质量失真；3MSBM 与 MSBM 同年给出两种多边缘构造，问题未收敛。</p></div></div>''',
 "4 · Insight", "arXiv:2604.05673、E04；ICML 2025；arXiv:2604.08580；ICLR 2026；R:2506.10168、arXiv:2510.16587")
# 18 I5-I8
content("Insight I5–I8", "具身：策略头而非数据管道 · 评测基础设施是短板 · 轻量求解器是探针与教师 · 开源三簇",
 '''<div class="cols"><div class="card red"><h3>I5 · 具身 SB 进入策略头，评测没跟上</h3><p>BridgePolicy / RSBM 的收益来自更好的先验（观测）而非更好的翻译。真机统计协议缺口在 2026 年依旧——判断。</p></div>
<div class="card red"><h3>I6 · 连续高维 SB 仍无 ground truth</h3><p>离散有 catsbench，能量采样有 DW4 / LJ 系与 SPICE，unpaired 翻译只有 FID + NFE。coupling 质量必须用独立于视觉质量的指标（E19）。</p></div>
<div class="card"><h3>I7 · 轻量闭式求解器 = 探针 + 教师</h3><p>LightSB(-M) 分钟级 ε 扫描（E03）；DLightSB 搬到离散；LBM / CDBM 说明重模型最终靠轻模型部署（E16）。</p></div>
<div class="card red"><h3>I9 · AM 成为生成式策略 RL 的默认优化器候选</h3><p>2026 上半年五条 RL 路线 + 组合优化 + 平均场控制都用「回归 adjoint 目标」替代「反传多步去噪」；diffusion/flow 策略的在线微调不再必须走 DDPO 式 policy gradient——判断，待真机协议验证。</p></div>
<div class="card"><h3>I8 · 开源生态三簇</h3><p>Meta/FAIR（Adjoint 系、flow_matching、GSBM）· Guan-Horng Liu 及合作者（SB-FBSDE → DASBS）· Korotin 组（LightSB 系、ASBM、CSBM、catsbench）。分别押注 SOC 与采样、控制视角的 SB、闭式与基准。</p></div></div>''',
 "4 · Insight", "ICML 2026；E15/E19；E03/E16；metadata/resources.tsv")
# 19 chapter 5 + advice
chapter("05", "对具身 sim2real 的启示", "面向 SB-Render-Lite 一类项目：先定范式、再选基线、评测先于方法。")
content("六条可执行建议", "先定范式再选方法；unpaired 用 DSBM-IMF 起步、用 ε 谱少步化；paired 双基线；约束进代价；\u201c策略即桥\u201d值得一试；评测先于方法。",
 '''<table><tr><th>#</th><th>建议</th><th>依据</th></tr>
<tr><td>1</td><td>表示对齐（EgoBridge / Guided OT）与生成式 transport（SB Flow / GSBM / BDGxRL）不可平替；叠加时按归因口径分别消融</td><td>synthesis §4.4</td></tr>
<tr><td>2</td><td>unpaired 主线以 DSBM-IMF 为边缘保持基线；同一网络从 ε=1 走到小 ε 换 3 步部署</td><td>E01；arXiv:2604.05673</td></tr>
<tr><td>3</td><td>paired 对照双基线 I²SB + DDBM；用 UniDB 可调终端罚检查是否过平滑</td><td>E02；ICML 2025</td></tr>
<tr><td>4</td><td>任务约束写进 GSBM 状态代价（keypoint / depth / 逆动力学），不做后处理</td><td>R:2310.02233</td></tr>
<tr><td>5</td><td>真机数据极少时试\u201c策略即桥\u201d（观测先验采样）——判断，需在 SimplerEnv 协议下验证</td><td>ICML 2026；E11</td></tr>
<tr><td>6</td><td>评测先于方法：四层指标 + 功效计算；coupling 用独立指标；视觉指标服从下游 policy success</td><td>E11、E19、synthesis §5</td></tr></table>''',
 "5 · 启示", "见表内依据列")
# 20 chapter 6 + watchlist + method
chapter("06", "观察清单与方法说明", "未来 12 个月的判据与触发动作；本报告的核验口径与局限。")
content("未来 12 个月观察清单", "五个观察点、各自的判据与触发动作；放榜后复核列入 metadata 维护。",
 '''<table><tr><th>观察点</th><th>判据</th><th>触发动作</th></tr>
<tr><td>ε 自动选择</td><td>出现按任务 / 数据自适应 ε 的准则并在 unpaired 翻译上验证</td><td>更新 I1；README §2.6</td></tr>
<tr><td>连续高维 SB 基准</td><td>catsbench 式解析解基准扩展到连续高维</td><td>更新 I6；README §5</td></tr>
<tr><td>dLLM 的 SB/SOC 微调</td><td>DAM / DASBS 之外出现第二条独立路线并在 ≥7B 验证</td><td>更新 §2.3</td></tr>
<tr><td>真机统计协议</td><td>具身 SB 论文报告试次数 / 种子 / 置信区间</td><td>更新 I5</td></tr>
<tr><td>放榜后复核</td><td>2105.11739、2602.23737 两条 preprint；FAS / DASBS 的 PMLR 页码；MSBM、MadSBM、RSBM、MDNS、AM-SMP 的 venue</td><td>更新 <code>metadata/*.tsv</code>，重建 README</td></tr></table>''',
 "6 · 观察", "survey/SB_TREND_REPORT_2026.md §5")
content("方法与证据、局限", "未核验即不写；数字取自摘要或论文页可见内容，未复现；2025H2–2026 覆盖以 WebSearch 命中为主，不保证完备。",
 f'''<div class="cols"><div class="card"><h3>来源层次</h3><ul><li>核心 {n_core} 篇：逐篇精读，经 10 路审查 + 修复</li><li>专题 {n_topics} 份：方法谱系、基线协议、评测方案</li>
<li>扩展 {n_ext} 条：{n_ext - n_new} 条基础论文（Semantic Scholar 批量核验 47/49，1 条记忆错误剔除）+ {n_new} 条 2025–26 新检索（21 条 WebSearch 直读会议页 + 72 条来自四条 arXiv API 近 12 月扫描共 448 篇候选的人工筛选）</li></ul></div>
<div class="card red"><h3>局限与复现</h3><ul><li>WebSearch 阶段 arXiv API 与 Semantic Scholar 限流；扫描阶段 API 恢复，277 篇候选中未入选者保留在 survey/raw 雷达表</li>
<li>作者不确定留空；venue 无会议页证据写 <code>arXiv</code></li><li>复现：<code>build_readme.py</code> · <code>s2_verify.py</code> · <code>translate_batch.sh</code> · <code>qa_table.py</code> · <code>build_slides.py</code></li></ul></div></div>''',
 "6 · 方法", "survey/SB_TREND_REPORT_2026.md §6")
# ending
S.append(f'''<section class="slide cover"><div class="bar"></div>
<h1 style="top:200px;left:0;width:1280px;text-align:center;font-size:48px">谢谢 · 欢迎 PR</h1>
<h2 style="top:290px;left:0;width:1280px;text-align:center">github.com/asimfish/awesome_Schrodinger_Bridge</h2>
<div class="meta" style="left:0;width:1280px;text-align:center;top:380px">加条目：改 <code>metadata/extended.tsv</code> → <code>python3 scripts/build_readme.py</code><br>读精读：<code>reports/INDEX.md</code> · 看趋势：<code>survey/SB_TREND_REPORT_2026.md</code> · 译本 QA：<code>papers_zh/QA_REPORT.md</code></div>
<div class="foot" style="text-align:center;padding-left:0">© 2026 Yufeng Li · CC BY 4.0（文本）/ MIT（脚本）· 译本仅供研究学习，引用请以英文原文为准</div></section>''')

html = (ROOT / "slides/_deck_template.html").read_text(encoding="utf-8").replace("{{SLIDES}}", "\n".join(S))
# 页码
secs = html.split('<section class="slide'); out = secs[0]
for k, s in enumerate(secs[1:], 1):
    out += '<section class="slide' + s.replace('<span class="pn"></span>', f'<span class="pn">{k} / {len(secs) - 1}</span>', 1)
(ROOT / "slides/awesome_sb_report.html").write_text(out, encoding="utf-8")
print("slides:", len(S), "pages ->", "slides/awesome_sb_report.html", f"{len(out)//1024} KB")
