# Awesome Schrödinger Bridge

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) ![papers](https://img.shields.io/badge/papers-166-blue) ![中文精读](https://img.shields.io/badge/%E4%B8%AD%E6%96%87%E7%B2%BE%E8%AF%BB-25-orange) ![中文译本](https://img.shields.io/badge/%E4%B8%AD%E6%96%87%E8%AF%91%E6%9C%AC%20PDF-29-red) ![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen) ![last update](https://img.shields.io/badge/last%20update-2026-09-04-lightgrey)

A curated list of papers, code, tutorials and **Chinese deep-dive reports** on the **Schrödinger Bridge (SB)** problem and its modern incarnations: diffusion Schrödinger bridges and bridge matching, generalized / multi-marginal / unbalanced SB, adjoint & stochastic-optimal-control samplers, and their applications in generative modeling, scientific data, and embodied AI (sim2real, cross-domain imitation).

本仓库系统整理薛定谔桥（Schrödinger Bridge）方向的论文与资源。**核心论文每篇配有中文精读报告（`reports/`）与保版式中文译本 PDF（`papers_zh/`，由 [SuperTranslate](https://github.com/asimfish/super_translate) 生成并经视觉 QA）**；20 份专题笔记（`topics/`）梳理方法谱系与基线；2025–2026 趋势调研与洞见见 `survey/`，汇报 PPT（HTML / PDF / Beamer）见 `slides/`。

*Maintained by [asimfish](https://github.com/asimfish). Entries marked ⭐ are core papers with full Chinese reports and translated PDFs. Venues are verified against arXiv comments / OpenReview / proceedings; preprints are labelled `arXiv`. Contributions are welcome — see [Contributing](#8-contributing-citation-license).*

**Legend**: `paper` arXiv/publisher page · `code` official implementation · `project` project page · `📄 PDF` English PDF in repo · `📘 精读` Chinese deep-dive report · `🀄 译本` layout-preserving Chinese PDF

## [Content](#content)

1. [Surveys, Tutorials & Foundations](#1-surveys-tutorials-foundations) — 综述、教程与基础  
2. [Diffusion Schrödinger Bridges & Bridge Matching](#2-diffusion-schrdinger-bridges-bridge-matching) — 扩散薛定谔桥与桥匹配（求解器）  
&emsp;2.1. [IPF / DSB / IMF / DSBM Lineage](#21-ipf-dsb-imf-dsbm-lineage) — IPF / DSB / IMF / DSBM 谱系  
&emsp;2.2. [Paired Bridges (I²SB, DDBM, DBIM)](#22-paired-bridges-isb-ddbm-dbim) — 成对数据桥  
&emsp;2.3. [Generalized, Multi-marginal, Mean-field & Unbalanced SB](#23-generalized-multi-marginal-mean-field-unbalanced-sb) — 广义 / 多边缘 / 平均场 / 非平衡 SB  
&emsp;2.4. [Light, Latent & Few-step Bridges](#24-light-latent-few-step-bridges) — 轻量 / 隐空间 / 少步桥  
&emsp;2.5. [Discrete-state Bridges](#25-discrete-state-bridges) — 离散状态空间桥  
&emsp;2.6. [Flow Matching, Stochastic Interpolants & SB Unification](#26-flow-matching-stochastic-interpolants-sb-unification) — 流匹配 / 随机插值 / SB 统一  
3. [Sampling & Stochastic Optimal Control](#3-sampling-stochastic-optimal-control) — 采样与随机最优控制  
&emsp;3.1. [Adjoint / SOC Samplers (energy-only)](#31-adjoint-soc-samplers-energy-only) — Adjoint / SOC 采样器（仅能量）  
&emsp;3.2. [SOC for Reward Fine-tuning & RL (Adjoint Matching lineage)](#32-soc-for-reward-fine-tuning-rl-adjoint-matching-lineage) — SOC 奖励微调与 RL（Adjoint Matching 谱系）  
&emsp;3.3. [Diffusion Samplers, Boltzmann Generators & Competitors](#33-diffusion-samplers-boltzmann-generators-competitors) — 扩散采样器 / Boltzmann 生成器 / 竞品  
4. [Applications](#4-applications) — 应用  
&emsp;4.1. [Image Translation, Restoration & Editing](#41-image-translation-restoration-editing) — 图像翻译 / 修复 / 编辑  
&emsp;4.2. [Video, 3D, Speech, Audio & Multimodal](#42-video-3d-speech-audio-multimodal) — 视频 / 3D / 语音 / 音频 / 多模态  
&emsp;4.3. [Science: Single-cell, Molecules, Chemistry & Physics](#43-science-single-cell-molecules-chemistry-physics) — 科学：单细胞 / 分子 / 化学 / 物理  
&emsp;4.4. [Embodied AI: Sim2Real, Cross-domain Transfer & RL](#44-embodied-ai-sim2real-cross-domain-transfer-rl) — 具身智能：sim2real / 跨域迁移 / RL  
&emsp;4.5. [Optimal Transport for Imitation & Reward](#45-optimal-transport-for-imitation-reward) — 最优传输用于模仿学习与奖励  
5. [Codebases & Benchmarks](#5-codebases-benchmarks) — 代码库与基准  
6. [Chinese Deep-dive Reports & Topic Notes](#6-chinese-deep-dive-reports-topic-notes) — 中文精读报告与专题笔记  
7. [Trend Report & Slides](#7-trend-report-slides) — 趋势报告与汇报  
8. [Contributing, Citation & License](#8-contributing-citation-license) — 贡献、引用与许可  

<a name="1-surveys-tutorials-foundations"></a>
## [1. Surveys, Tutorials & Foundations](#content)
*综述、教程与基础*

1. **Sinkhorn Distances: Lightspeed Computation of Optimal Transport.** NeurIPS, 2013. [paper](https://arxiv.org/abs/1306.0895)

 *Marco Cuturi*

 > 熵正则 OT 的 Sinkhorn 算法：SB 离散化后每一步 IPF 都是它。

2. **A survey of the Schrödinger problem and some of its connections with optimal transport.** Discrete Contin. Dyn. Syst., 2014. [paper](https://arxiv.org/abs/1308.0215)

 *Christian Léonard*

 > Schrödinger 问题的经典综述：大偏差起源、与熵正则 OT 的联系、动态/静态表述。

3. **Computational Optimal Transport.** Found. Trends Mach. Learn., 2019. [paper](https://arxiv.org/abs/1803.00567)

 *Gabriel Peyré, Marco Cuturi*

 > 计算 OT 教科书；熵正则、Sinkhorn、动态形式的标准参考。

4. **Stochastic Control Liaisons: Richard Sinkhorn Meets Gaspard Monge on a Schrödinger Bridge.** SIAM Review, 2021. [paper](https://arxiv.org/abs/2005.10963)

 *Yongxin Chen, Tryphon T. Georgiou, Michele Pavon*

 > 把 SB 讲成随机控制问题的权威综述：Sinkhorn 迭代 ↔ IPF ↔ 最优控制三位一体。

5. **Flow Matching Guide and Code.** arXiv, 2024. [paper](https://arxiv.org/abs/2412.06264), [code](https://github.com/facebookresearch/flow_matching)

 *Yaron Lipman, Marton Havasi, Peter Holderrieth, Neta Shaul, Matt Le et al.*

 > Flow Matching 官方指南与代码；含离散 FM 与与 SB/随机插值的关系。

6. **Foundations of Schrödinger Bridges for Generative Modeling.** arXiv, 2026. [paper](https://arxiv.org/abs/2603.18992)

 *Sophia Tang*

 > 从 OT、随机控制与路径空间优化出发系统建立 SB 数学基础的教程式指南，聚焦动态表述与现代生成建模的联系（含 DSBM 专节）。

7. **Notes on generative modeling: flow matching, diffusion, optimal transport and Schrödinger bridge.** arXiv, 2026. [paper](https://arxiv.org/abs/2606.30053)

 *Titouan Vayer*

 > Peyresq 2026 暑期学校讲义：用 OT 把 flow matching、扩散与 SB 串成一条线，强调 IMF 收敛到熵最优耦合而 rectified flow 不收敛到 OT 耦合。

<a name="2-diffusion-schrdinger-bridges-bridge-matching"></a>
## [2. Diffusion Schrödinger Bridges & Bridge Matching](#content)
*扩散薛定谔桥与桥匹配（求解器）*

<a name="21-ipf-dsb-imf-dsbm-lineage"></a>
### [2.1. IPF / DSB / IMF / DSBM Lineage](#content)
*IPF / DSB / IMF / DSBM 谱系*

1. **Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling.** NeurIPS, 2021. [paper](https://arxiv.org/abs/2106.01357), [code](https://github.com/JTT94/diffusion_schrodinger_bridge)

 *Valentin De Bortoli, James Thornton, Jeremy Heng, Arnaud Doucet*

 > DSB：用 IPF 交替回归 score 网络求解 SB，把 score-based 生成推广到任意两分布之间。

2. **Solving Schrödinger Bridges via Maximum Likelihood.** Entropy, 2021. [paper](https://arxiv.org/abs/2106.02081)

 *Francisco Vargas, Pierre Thodoroff, Neil D. Lawrence, Austen Lamacraft*

 > 用最大似然（GP 回归 drift）求解 SB 的早期方法。

3. **Deep Generative Learning via Schrödinger Bridge.** ICML, 2021. [paper](https://arxiv.org/abs/2106.10410)

 *Gefei Wang, Yuling Jiao, Qiang Xu, Yang Wang, Can Yang*

 > 早期用 SB 做生成建模的工作：以数据分布为终端的 SB + 分数估计。

4. **Likelihood Training of Schrödinger Bridge using Forward-Backward SDEs Theory.** ICLR, 2022. [paper](https://arxiv.org/abs/2110.11291), [code](https://github.com/ghliu/SB-FBSDE)

 *Tianrong Chen, Guan-Horng Liu, Evangelos A. Theodorou*

 > SB-FBSDE：用前向-后向 SDE 理论给 SB 一个似然训练目标，连通 SB 与 score-based 模型。

5. **Diffusion Schrödinger Bridge Matching.** NeurIPS, 2023. [paper](https://arxiv.org/abs/2303.16852), [code](https://github.com/yuyang-shi/dsbm-pytorch), [📄 PDF](papers/2303.16852_dsbm_diffusion_schrodinger_bridge_matching.pdf), [🀄 译本](papers_zh/2303.16852_dsbm_diffusion_schrodinger_bridge_matching.zh.pdf)

 *Yuyang Shi, Valentin De Bortoli, Andrew Campbell, Arnaud Doucet*

 > DSBM：Iterative Markovian Fitting（IMF）交替投影到 Markov 类与 reciprocal 类，避免 IPF 的误差累积；unpaired 翻译主力基线。

6. **Diffusion Bridge Mixture Transports, Schrödinger Bridge Problems and Generative Modeling.** JMLR, 2023. [paper](https://arxiv.org/abs/2304.00917)

 *Stefano Peluchetti*

 > IDBM：给 bridge matching 的理论地基——桥混合 → Markov 化 → 迭代收敛到 SB（第一次迭代即合法 transport）。

7. **Schrödinger Bridge Flow for Unpaired Data Translation.** NeurIPS, 2024 (Spotlight). [paper](https://arxiv.org/abs/2409.09347), [📄 PDF](papers/2409.09347_schrodinger_bridge_flow_unpaired_translation.pdf), [📘 精读](reports/2409.09347_schrodinger_bridge_flow_unpaired_translation.md), [🀄 译本](papers_zh/2409.09347_schrodinger_bridge_flow_unpaired_translation.zh.pdf) ⭐

 *Valentin De Bortoli, Iryna Korshunova, Andriy Mnih, Arnaud Doucet*

8. **BM2: Coupled Schrödinger Bridge Matching.** TMLR, 2024. [paper](https://arxiv.org/abs/2409.09376)

 *Stefano Peluchetti*

 > BM²：耦合的双向 bridge matching，单轮同时拟合两个方向。

9. **Schrödinger bridge for generative AI: Soft-constrained formulation and convergence analysis.** arXiv, 2025. [paper](https://arxiv.org/abs/2510.11829)

 *Jin Ma, Ying Tan, Renyuan Xu*

 > 软约束 SB（SCSBP）：把硬终端约束换成一般罚函数，得到 McKean–Vlasov 型控制问题；证明各罚强度下解存在并给出收敛分析——与 UniDB 的可调终端罚同一思路的理论版。

10. **Fractional Diffusion Bridge Models.** NeurIPS, 2025. [paper](https://arxiv.org/abs/2511.01795)

 *Gabriel Nobis, Maximilian Springenberg, Arina Belova, Rembert Daems, Christoph Knochenhauer et al.*

 > FDBM：以分数布朗运动的 Markov 近似（MA-fBM）驱动 diffusion bridge，保留长程记忆/粗糙性等非 Markov 特性；证明保耦合桥存在，并推广到 SB 给出 unpaired 翻译的损失。

11. **Efficient Generative Modeling beyond Memoryless Diffusion via Adjoint Schrödinger Bridge Matching.** arXiv, 2026. [paper](https://arxiv.org/abs/2602.15396)

 *Jeongwoo Shin, Jinhwan Sul, Joonseok Lee, Jaewong Choi, Jaemoo Choi*

 > Adjoint SBM（注意与 2024 年对抗式 ASBM 同名）：两阶段——先以 data-to-energy 采样视角学 SB 前向耦合（把数据运到能量定义的先验），再用简单匹配损失学反向生成；脱离 memoryless 前向过程后轨迹更直、高维更稳。

<a name="22-paired-bridges-isb-ddbm-dbim"></a>
### [2.2. Paired Bridges (I²SB, DDBM, DBIM)](#content)
*成对数据桥*

1. **I²SB: Image-to-Image Schrödinger Bridge.** ICML, 2023. [paper](https://arxiv.org/abs/2302.05872), [code](https://github.com/NVlabs/I2SB), [📄 PDF](papers/2302.05872_i2sb.pdf), [📘 精读](reports/2302.05872_i2sb.md), [🀄 译本](papers_zh/2302.05872_i2sb.zh.pdf) ⭐

 *Guan-Horng Liu, Arash Vahdat, De-An Huang, Evangelos A. Theodorou, Weili Nie, Anima Anandkumar*

2. **Aligned Diffusion Schrödinger Bridges.** UAI, 2023. [paper](https://arxiv.org/abs/2302.11419)

 *Vignesh Ram Somnath, Matteo Pariset, Ya-Ping Hsieh, María Rodríguez Martínez, Andreas Krause, Charlotte Bunne*

 > Aligned DSB：利用已配对数据的 SB（单细胞对齐场景）。

3. **Denoising Diffusion Bridge Models.** ICLR, 2024. [paper](https://arxiv.org/abs/2309.16948), [code](https://github.com/alexzhou907/DDBM), [📄 PDF](papers/2309.16948_ddbm_denoising_diffusion_bridge_models.pdf), [🀄 译本](papers_zh/2309.16948_ddbm_denoising_diffusion_bridge_models.zh.pdf)

 *Linqi Zhou, Aaron Lou, Samar Khanna, Stefano Ermon*

 > DDBM：在 paired 端点之间定义 Doob h-transform 桥，统一 I²SB 类方法的训练/采样接口。

4. **Diffusion Bridge Implicit Models.** ICLR, 2025. [paper](https://arxiv.org/abs/2405.15885), [code](https://github.com/thu-ml/DiffusionBridge)

 *Kaiwen Zheng, Guande He, Jianfei Chen, Fan Bao, Jun Zhu*

 > DBIM：给 diffusion bridge 的 DDIM 式隐式采样器，大幅减少 NFE。

5. **Control Consistency Losses for Diffusion Bridges.** arXiv, 2025. [paper](https://arxiv.org/abs/2512.05070)

 *Samuel Howard, Nikolas Nüsken, Jakiw Pidstrigach*

 > 控制一致性损失：基于最优控制的自洽性质在线迭代学习给定首末状态的条件扩散（diffusion bridge），无需对模拟轨迹求导，对稀有事件尤其有效。

6. **UniDB: A Unified Diffusion Bridge Framework via Stochastic Optimal Control.** ICML, 2025. [paper](https://proceedings.mlr.press/v267/zhu25o.html), [code](https://github.com/UniDB-SOC/UniDB)

 > UniDB：把 diffusion bridge 写成 SOC 问题并给出闭式最优控制器；Doob h-transform 类桥（DDBM 等）是终端罚系数 →∞ 的特例，可调终端罚显著改善修复细节。

7. **Structured Diffusion Bridges: Inductive Bias for Denoising Diffusion Bridges.** ICML, 2026. [paper](https://arxiv.org/abs/2605.02973)

 *Eitan Kosman, Gabriele Serussi, Chaim Baskin*

 > Structured Diffusion Bridges：刻画模态翻译可行解空间并用对齐约束收窄，成对监督降为可选启发；unpaired / 半配对 / 配对三种监督下表现一致，接近全配对质量。

8. **Resolving Endpoint Underfitting in Diffusion Bridges via Noise Alignment.** CVPR, 2026. [paper](https://arxiv.org/abs/2605.28962)

 *Yurong Gao, Zicheng Zhang, Congying Han, Tiande Guo, Xinmin Qiu*

 > NADB：发现按 score-matching 套路学桥会在 t→0 靶端欠拟合（输入与回归目标噪声水平差过大）；用均值网络给更干净的条件目标 + 噪声对齐映射修复。

9. **PRISM: Principled Reference Identification for Schrodinger Bridge Model.** arXiv, 2026. [paper](https://arxiv.org/abs/2608.06893)

 *Forouzan Fallah, Yezhou Yang*

 > PRISM：桥参考过程设计理论——刻画逐模态调度下仍可精确处理的时变高斯参考（瞬时协方差可交换），证明「不可见性原理」（精确 drift + 无限步下任何参考都恢复真后验），并在有限步预算下给出最优噪声谱 ∝ 传感器摧毁的信息谱。

<a name="23-generalized-multi-marginal-mean-field-unbalanced-sb"></a>
### [2.3. Generalized, Multi-marginal, Mean-field & Unbalanced SB](#content)
*广义 / 多边缘 / 平均场 / 非平衡 SB*

1. **Deep Generalized Schrödinger Bridge.** NeurIPS, 2022 (Oral). [paper](https://arxiv.org/abs/2209.09893), [code](https://github.com/ghliu/DeepGSB), [📄 PDF](papers/2209.09893_deep_generalized_schrodinger_bridge.pdf), [📘 精读](reports/2209.09893_deep_generalized_schrodinger_bridge.md), [🀄 译本](papers_zh/2209.09893_deep_generalized_schrodinger_bridge.zh.pdf) ⭐

 *Guan-Horng Liu, Tianrong Chen, Oswin So, Evangelos A. Theodorou*

2. **Deep Momentum Multi-Marginal Schrödinger Bridge.** NeurIPS, 2023. [paper](https://arxiv.org/abs/2303.01751), [code](https://github.com/TianrongChen/DMSB)

 *Tianrong Chen, Guan-Horng Liu, Molei Tao, Evangelos A. Theodorou*

 > DMSB：带动量的多边缘 SB，用于单细胞等多时间点轨迹推断（3MSBM 的前身）。

3. **Unbalanced Diffusion Schrödinger Bridge.** arXiv, 2023. [paper](https://arxiv.org/abs/2306.09099)

 *Matteo Pariset, Ya-Ping Hsieh, Charlotte Bunne, Andreas Krause, Valentin De Bortoli*

 > Unbalanced DSB：允许质量不守恒（细胞增殖/死亡）的 SB 求解。

4. **Generalized Schrödinger Bridge Matching.** ICLR, 2024 (Poster). [paper](https://arxiv.org/abs/2310.02233), [code](https://github.com/facebookresearch/generalized-schrodinger-bridge-matching), [📄 PDF](papers/2310.02233_generalized_schrodinger_bridge_matching.pdf), [📘 精读](reports/2310.02233_generalized_schrodinger_bridge_matching.md), [🀄 译本](papers_zh/2310.02233_generalized_schrodinger_bridge_matching.zh.pdf) ⭐

 *Guan-Horng Liu, Yaron Lipman, Maximilian Nickel, Brian Karrer, Evangelos A. Theodorou, Ricky T. Q. Chen*

5. **Stochastic Optimal Control for Diffusion Bridges in Function Spaces.** NeurIPS, 2024. [paper](https://arxiv.org/abs/2405.20630)

 *Byoungwoo Park, Jungwon Choi, Sungbin Lim, Juho Lee*

 > 函数空间上的 diffusion bridge SOC（FAS 的函数空间前身）。

6. **Feedback Schrödinger Bridge Matching.** ICLR, 2025 (Oral). [paper](https://arxiv.org/abs/2410.14055)

 *Panagiotis Theodoropoulos, Nikolaos Komianos, Vincent Pacelli, Guan-Horng Liu, Evangelos A. Theodorou*

 > Feedback SBM：把 SB 求解写成闭环反馈控制，提升多边缘/受约束场景的稳定性。

7. **Modeling Cell Dynamics and Interactions with Unbalanced Mean Field Schrödinger Bridge (CytoBridge).** NeurIPS, 2025. [paper](https://arxiv.org/abs/2505.11197), [code](https://github.com/zhenyiizhang/CytoBridge-NeurIPS)

 *Zhenyi Zhang, Zihan Wang, Yuhao Sun, Tiejun Li, Peijie Zhou*

 > UMFSB / CytoBridge：把平均场 SB 推广到非归一化分布，用四个网络（转移速度 v、增长率 g、log 密度 score s、相互作用势 Φ）显式建模细胞转移、增殖与相互作用，能量 + 重建 + Fokker–Planck 约束联合训练；在合成 GRN 与真实 scRNA-seq 上消除虚假转移。

8. **Momentum Multi-Marginal Schrödinger Bridge Matching.** NeurIPS, 2025. [paper](https://arxiv.org/abs/2506.10168), [code](https://github.com/panostheo98/3MSBM), [📄 PDF](papers/2506.10168_momentum_multi_marginal_sbm.pdf), [📘 精读](reports/2506.10168_momentum_multi_marginal_sbm.md), [🀄 译本](papers_zh/2506.10168_momentum_multi_marginal_sbm.zh.pdf) ⭐

 *Panagiotis Theodoropoulos, Augustinos D. Saravanos, Evangelos A. Theodorou, Guan-Horng Liu*

9. **Multi-Marginal Schrödinger Bridge Matching.** arXiv, 2025. [paper](https://arxiv.org/abs/2510.16587), [code](https://github.com/bw-park/MSBM)

 *Byoungwoo Park, Juho Lee*

 > MSBM：把 IMF 推广到多边缘约束——逐区间构造局部 SB 并共享全局控制参数化，保证中间边缘全部满足且轨迹连续；scRNA-seq 上与 3MSBM 同赛道。

10. **Entangled Schrödinger Bridge Matching.** arXiv, 2025. [paper](https://arxiv.org/abs/2511.07406)

 *Sophia Tang, Yinuo Zhang, Pranam Chatterjee*

 > EntangledSBM：学习相互作用多粒子系统的一阶与二阶随机动力学，每个粒子路径的方向与幅度依赖其他粒子路径；面向 MD 与异质细胞群。

11. **Multi-marginal temporal Schrödinger Bridge Matching from unpaired data.** ICML, 2026. [paper](https://arxiv.org/abs/2510.01894), [code](https://github.com/tgravier/MMDSBM-pytorch)

 *Thomas Gravier, Thomas Boyer, Auguste Genovesio*

 > MMtSBM：把 IMF 以因子化方式推到多边缘、直接从 unpaired 快照学时间演化，保留 DSBM 的理论保证；与 3MSBM / MSBM 构成多边缘三路并行。

12. **Contact Wasserstein Geodesics for Non-Conservative Schrödinger Bridges.** ICLR, 2026. [paper](https://arxiv.org/abs/2511.06856)

 *Andrea Testa, Søren Hauberg, Tamim Asfour, Leonel Rozo*

 > NCGSB / 接触 Wasserstein 测地线：用接触哈密顿力学放开能量守恒假设，允许能量随时间变化的广义 SB；参数化 Wasserstein 流形后化为有限维测地线计算，ResNet 实现、免迭代。

13. **Schrödinger Bridge Over A Compact Connected Lie Group.** arXiv, 2026. [paper](https://arxiv.org/abs/2603.14049)

 *Hamza Mahmood, Abhishek Halder, Adeel Akhtar*

 > 紧致连通李群上的 SB：坐标无关的 SOC 表述，证明 Schrödinger 系统解的存在唯一并构造几何控制器；SO(2)/SO(3) 数值例。

14. **A Generalized Sinkhorn Algorithm for Mean-Field Schrödinger Bridge.** arXiv, 2026. [paper](https://arxiv.org/abs/2604.06531)

 *Asmaa Eldesoukey, Yongxin Chen, Abhishek Halder*

 > 平均场 SB 的广义 Sinkhorn：推广 Hopf–Cole 变换，设计 Sinkhorn 型递归解相应积分-PDE 系统，给出收敛保证（排斥/吸引相互作用数值例）。

15. **Nonlocal Mean Field Schrödinger Bridge with Learned Interactions.** arXiv, 2026. [paper](https://arxiv.org/abs/2606.04265)

 *Daisuke Inoue, Dante Kalise, Mathieu Laurière*

 > 非局部平均场 SB：用神经代理替代二次复杂度的非局部相互作用项，四阶段交替更新前/后向势与代理，给出 Grönwall 型稳定性界。

16. **Reflected Schrödinger Bridge Matching.** arXiv, 2026. [paper](https://arxiv.org/abs/2607.03626)

 *Viktor Nilsson, Marcus Häggbom, Pierre Nyquist, Joakim Andén*

 > 反射 SBM：以单位超立方体上的反射布朗运动为参考过程，把 (α-)IMF 的部分 simulation-free 训练搬到反射 SB，保证样本始终落在数据域内；相对非反射基线额外开销可忽略，生成质量持平或略升。

17. **Twisted Schrödinger Bridge Matching.** arXiv, 2026. [paper](https://arxiv.org/abs/2607.16987)

 *Maxence Noble, Marie Scheid, Yazid Janati, Eric Moulines, Alain Durmus*

 > Twisted SBM：参考过程换成带时变势的 twisted 布朗运动（Feynman–Kac 变换），在 IMF/DSBM 框架内求广义 SB；DSBM 是零势特例。

18. **Branched Schrödinger Bridge Matching.** ICLR, 2026. [paper](https://iclr.cc/virtual/2026/poster/10008461), [code](https://github.com/sophtang/BranchSBM)

 *Sophia Tang, Yinuo Zhang, Alexander Tong, Pranam Chatterjee*

 > BranchSBM：学多条分叉的速度场 + 各分支的质量增长网络，把 Unbalanced CondSOC 之和作为目标，刻画单细胞命运分叉与药物扰动的多模态终态；单分支 SBM 会模式塌缩。

<a name="24-light-latent-few-step-bridges"></a>
### [2.4. Light, Latent & Few-step Bridges](#content)
*轻量 / 隐空间 / 少步桥*

1. **Light Schrödinger Bridge.** ICLR, 2024. [paper](https://arxiv.org/abs/2310.01174), [code](https://github.com/ngushchin/LightSB), [📄 PDF](papers/2310.01174_light_schrodinger_bridge.pdf), [🀄 译本](papers_zh/2310.01174_light_schrodinger_bridge.zh.pdf)

 *Alexander Korotin, Nikita Gushchin, Evgeny Burnaev*

 > LightSB：Gaussian-mixture 参数化的闭式 SB 求解器，分钟级训练，适合低维/latent。

2. **Light and Optimal Schrödinger Bridge Matching.** ICML, 2024. [paper](https://arxiv.org/abs/2402.03207), [code](https://github.com/SKholkin/LightSB-Matching)

 *Nikita Gushchin, Sergei Kholkin, Evgeny Burnaev, Alexander Korotin*

 > LightSB-M：把 LightSB 的闭式参数化与 bridge matching 结合，单步优化到 SB。

3. **Adversarial Schrödinger Bridge Matching.** NeurIPS, 2024. [paper](https://arxiv.org/abs/2405.14449), [code](https://github.com/Daniil-Selikhanovych/ASBM)

 *Nikita Gushchin, Daniil Selikhanovych, Sergei Kholkin, Evgeny Burnaev, Alexander Korotin*

 > ASBM：对抗式 D-IMF，用 GAN 判别器替代大量 NFE，像素空间少步 SB。

4. **Consistency Diffusion Bridge Models.** NeurIPS, 2024. [paper](https://arxiv.org/abs/2410.22637), [code](https://github.com/thu-ml/DiffusionBridge)

 *Guande He, Kaiwen Zheng, Jianfei Chen, Fan Bao, Jun Zhu*

 > CDBM：把 consistency 训练搬到 diffusion bridge，'先训 bridge 再压缩'到少步。

5. **LBM: Latent Bridge Matching for Fast Image-to-Image Translation.** ICCV, 2025. [paper](https://arxiv.org/abs/2503.07535), [code](https://github.com/gojasper/LBM)

 *Clément Chadebec, Onur Tasar, Sanjeev Sreetharan, Benjamin Aubin*

 > LBM：在 SD latent 上做 bridge matching，1 NFE 完成图像翻译/重光照/深度等任务。

6. **A Unified and Fast-Sampling Diffusion Bridge Framework via Stochastic Optimal Control (UniDB++).** arXiv, 2025. [paper](https://arxiv.org/abs/2505.21528), [code](https://github.com/2769433owo/UniDB-plusplus)

 *Mokai Pan, Kaizhen Zhu, Yuexin Ma, Yanwei Fu, Jingyi Yu et al.*

 > UniDB++：为 UniDB 反向 SDE 推出精确闭式解 + 数据预测 + SDE-Corrector，免训练把采样步数减到 5–10（最多 20×），并在特定条件下退化为 DBIM。

7. **LADB: Latent Aligned Diffusion Bridges for Semi-Supervised Domain Translation.** arXiv, 2025. [paper](https://arxiv.org/abs/2509.08628)

 *Xuqin Wang, Tao Wu, Yanfeng Zhang, Lu Liu, Dong Wang et al.*

 > LADB：半监督的 latent 对齐 diffusion bridge——在共享 latent 空间把预训练源域扩散模型与部分配对训练的目标域模型接起来，兼顾 unpaired 的灵活与 paired 的可控。

8. **A Closed-Form Framework for Schrödinger Bridges Between Arbitrary Densities.** arXiv, 2025. [paper](https://arxiv.org/abs/2511.07786)

 *Hanwen Huang*

 > 任意密度间 SB 的统一闭式框架：Schrödinger–Föllmer 过程与高斯 SB 都是特例，绕开迭代随机模拟。

9. **DBMSolver: A Training-free Diffusion Bridge Sampler for High-Quality Image-to-Image Translation.** CVPR, 2026. [paper](https://arxiv.org/abs/2605.05889), [code](https://github.com/snumprlab/dbmsolver)

 *Sankarshana Venugopal, Mohammad Mostafavi, Jonghyun Choi*

 > DBMSolver：利用 DBM 的 SDE/ODE 半线性结构做指数积分器，免训练 1/2 阶求解器；NFE 最多减 5×，DIODE 20 NFE 下 FID 较二阶基线降 53%。

10. **QDSB: Quantized Diffusion Schrödinger Bridges.** arXiv, 2026. [paper](https://arxiv.org/abs/2605.11983)

 *Tobias Fuchs, Florian Kalinke, Nadja Klein*

 > QDSB：simulation-free SB 需要 minibatch 熵 OT 耦合、代价高且扭曲全局几何——量化后在码本上求全局耦合，替代逐 minibatch 求解。

<a name="25-discrete-state-bridges"></a>
### [2.5. Discrete-state Bridges](#content)
*离散状态空间桥*

1. **Discrete Diffusion Schrödinger Bridge Matching for Graph Transformation.** ICLR, 2025. [paper](https://arxiv.org/abs/2410.01500)

 *Jun Hyeong Kim, Seonghwan Kim, Seokhyun Moon, Hyeongwoo Kim, Jeheon Woo, Woo Youn Kim*

 > 离散 DSBM 做图变换（分子/图编辑）。

2. **Categorical Schrödinger Bridge Matching.** ICML, 2025. [paper](https://arxiv.org/abs/2502.01416)

 *Grigoriy Ksenofontov, Alexander Korotin*

 > Categorical SBM：把 IMF 推到离散状态空间（categorical），文本/图/序列可用。

3. **Entering the Era of Discrete Diffusion Models: A Benchmark for Schrödinger Bridges and Entropic Optimal Transport.** ICLR, 2026. [paper](https://arxiv.org/abs/2509.23348), [code](https://github.com/gregkseno/catsbench)

 *Xavier Aramayo Carrasco, Grigoriy Ksenofontov, Aleksei Leonov, Iaroslav Koshelev, Alexander Korotin*

 > 离散空间 SB 基准：构造有解析解的分布对以严格评测离散 SB 求解器；副产品 DLightSB / DLightSB-M 与 α-CSBM。离散 SB 第一次有了 ground truth。

4. **Minimal-Action Discrete Schrödinger Bridge Matching for Peptide Sequence Design.** arXiv, 2026. [paper](https://arxiv.org/abs/2601.22408)

 *Shrey Goel, Pranam Chatterjee*

 > MadSBM：把肽序列生成写成氨基酸编辑图上的受控 CTMC，参考过程来自冻结的 ESM-2 logits，学时变控制场得到低作用量传输路径；首次在 SB 生成模型上做离散 classifier guidance。

5. **Generalized Schrödinger Bridge on Graphs.** arXiv, 2026. [paper](https://arxiv.org/abs/2602.04675)

 *Panagiotis Theodoropoulos, Juno Nam, Evangelos Theodorou, Jaemoo Choi*

 > GSBoG：图上的广义 SB——在状态代价增广动力学下学可执行的受控 CTMC 轨迹级策略，避免稠密全局求解器，随图规模与时域可扩展。

6. **Discrete diffusion samplers and bridges: Off-policy algorithms and applications in latent spaces.** ICML, 2026. [paper](https://arxiv.org/abs/2602.05961), [code](https://github.com/mmacosha/offpolicy-discrete-diffusion-samplers-and-bridges)

 *Arran Carter, Sanghyeok Choi, Kirill Tamogashev, Víctor Elvira, Esmeralda S. Whitammer*

 > 离散扩散采样器与桥的 off-policy 训练：把连续空间扩散采样器的 off-policy 算法推到离散空间，并应用于离散 latent 空间上的 data-to-energy SB。

7. **Discrete Diffusion Bridges for Spatiotemporally Aligned Image Translation and Generation.** ECCV, 2026. [paper](https://arxiv.org/abs/2608.29997)

 *Xing Xie, Jiawei Liu, Shijun Zhou, Huijie Fan, Zhi Han et al.*

 > DDB：离散扩散桥解决标准离散扩散在图像翻译中的时空错位——混合吸收态（mask 与源 token 的随机混合）注入源先验，信息引导的解码顺序替代随机掩码。

<a name="26-flow-matching-stochastic-interpolants-sb-unification"></a>
### [2.6. Flow Matching, Stochastic Interpolants & SB Unification](#content)
*流匹配 / 随机插值 / SB 统一*

1. **Neural Optimal Transport.** ICLR, 2023. [paper](https://arxiv.org/abs/2201.12220), [code](https://github.com/iamalexkorotin/NeuralOptimalTransport)

 *Alexander Korotin, Daniil Selikhanovych, Evgeny Burnaev*

 > Neural OT：对抗式弱 OT 求解器，SB/EOT 求解器的重要对照。

2. **Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow.** ICLR, 2023. [paper](https://arxiv.org/abs/2209.03003), [code](https://github.com/gnobitab/RectifiedFlow)

 *Xingchao Liu, Chengyue Gong, Qiang Liu*

 > Rectified Flow：reflow 直线化 + 蒸馏到一步。

3. **Building Normalizing Flows with Stochastic Interpolants.** ICLR, 2023. [paper](https://arxiv.org/abs/2209.15571)

 *Michael S. Albergo, Eric Vanden-Eijnden*

 > 随机插值短版：用插值构造 normalizing flow。

4. **Flow Matching for Generative Modeling.** ICLR, 2023. [paper](https://arxiv.org/abs/2210.02747), [code](https://github.com/facebookresearch/flow_matching)

 *Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le*

 > Flow Matching：条件回归训练 ODE 生成模型；SB 视角下它是 σ→0 的确定性极限。

5. **Improving and generalizing flow-based generative models with minibatch optimal transport.** TMLR, 2024. [paper](https://arxiv.org/abs/2302.00482), [code](https://github.com/atong01/conditional-flow-matching)

 *Alexander Tong, Kilian Fatras, Nikolay Malkin, Guillaume Huguet, Yanlei Zhang et al.*

 > OT-CFM：minibatch OT coupling 让 FM 轨迹变直、可做 unpaired 翻译。

6. **Simulation-free Schrödinger bridges via score and flow matching.** AISTATS, 2024. [paper](https://arxiv.org/abs/2307.03672), [code](https://github.com/atong01/conditional-flow-matching)

 *Alexander Tong, Esmeralda S. Whitammer, Kilian Fatras, Lazar Atanackovic, Yanlei Zhang et al.*

 > [SF]²M：score + flow matching 的 simulation-free SB 求解器。

7. **Optimal Flow Matching: Learning Straight Trajectories in Just One Step.** NeurIPS, 2024. [paper](https://arxiv.org/abs/2403.13117)

 *Nikita Kornilov, Petr Mokrov, Alexander Gasnikov, Alexander Korotin*

 > Optimal Flow Matching：一步得到直线 OT 轨迹。

8. **Stochastic Interpolants: A Unifying Framework for Flows and Diffusions.** JMLR, 2025. [paper](https://arxiv.org/abs/2303.08797)

 *Michael S. Albergo, Nicholas M. Boffi, Eric Vanden-Eijnden*

 > Stochastic Interpolants：统一 flow / diffusion / bridge / SB 的插值框架。

9. **A Unified Framework for Diffusion Bridge Problems: Flow Matching and Schrödinger Matching into One.** arXiv, 2025. [paper](https://arxiv.org/abs/2503.21756)

 *Minyoung Kim*

 > 统一视角：FM、minibatch-OT FM、minibatch SB-FM 与 DSBM 都是同一 bridge 框架的特例；含对 IMF 与 CFM 的简明技术回顾。

10. **Diffusion Bridge or Flow Matching? A Unifying Framework and Comparative Analysis.** arXiv, 2025. [paper](https://arxiv.org/abs/2509.24531)

 *Kaizhen Zhu, Mokai Pan, Zhechuan Yu, Jingya Wang, Jingyi Yu, Ye Shi*

 > Diffusion Bridge 还是 Flow Matching？用 SOC 视角统一两者并证明 diffusion bridge 的代价函数更低、轨迹更稳；从 OT 视角指出 FM 的 t/(1−t) 插值系数在小数据下失效。

11. **Flow Matching with Semidiscrete Couplings.** arXiv, 2025. [paper](https://arxiv.org/abs/2509.25519)

 *Alireza Mousavi-Hosseini, Stephen Y. Zhang, Michal Klein, Marco Cuturi*

 > 半离散耦合的 flow matching：OT-FM 只在大批次才见效；用半离散 OT 求解器给出可扩展的耦合替代。

12. **Curly Flow Matching for Learning Non-gradient Field Dynamics.** NeurIPS, 2025. [paper](https://arxiv.org/abs/2510.26645)

 *Katarina Petrović, Lazar Atanackovic, Viggo Moro, Kacper Kapuśniak, İsmail İlkan Ceylan et al.*

 > Curly-FM：现有 flow / bridge matching 基于最小作用量只能学梯度场；Curly-FM 设计可学非梯度、周期性动力学（如 scRNA 细胞周期）的匹配目标。

13. **Expected Batch Optimal Transport Plans and Consequences for Flow Matching.** arXiv, 2026. [paper](https://arxiv.org/abs/2605.12174)

 *Samuel Boïté, Julie Delon, Kimia Nadjahi*

 > 期望 minibatch OT 计划：把重复 minibatch OT 诱导的总体耦合形式化为期望批 OT 计划，证明大批次一致性并在半离散情形给出传输代价偏差与收敛速率——OT-CFM 的理论补丁。

14. **Lagrangian Flow Matching: A Least-Action Framework for Principled Path Design.** arXiv, 2026. [paper](https://arxiv.org/abs/2605.15419)

 *Shukai Du, Junzhe Zhang, Yiming Li*

 > Lagrangian flow matching：直线（rectified / OT）路径只是动能 Lagrangian 的自由粒子特例；按最小作用量原理用一般 Lagrangian 设计概率路径与速度场。

15. **Multiscale Supervised Unbalanced Optimal Transport Flow Matching.** arXiv, 2026. [paper](https://arxiv.org/abs/2605.16529)

 *Qiangwei Peng, Lezhi Chen, Peijie Zhou*

 > MUST-FM：利用单细胞数据的层级注释与转移先验（如谱系）做多尺度、可选监督的非平衡 OT flow matching，降低 UOT 的计算成本。

16. **Optimal Transport Flow Matching by Design.** arXiv, 2026. [paper](https://arxiv.org/abs/2606.04092)

 *Shimon Malnick, Matan Rusanovsky, Ohad Fried, Shai Avidan*

 > OT flow matching by design：不求 OT 耦合而是把先验当设计变量——许多先验与数据之间的恒等耦合已是 OT 最优，选一个可采样的即可得到直的非交叉轨迹。

17. **Multimarginal flow matching with optimal transport potentials.** ICML, 2026. [paper](https://arxiv.org/abs/2606.05327)

 *Raghav Kansal, David Crair, Nghia Nguyen, Scott Pope, Bradley Parry*

 > 多边缘 flow matching：借 FM ↔ 动态 OT 的联系，在动态 OT 作用量里加势项把流「软」引向中间边缘，得到 simulation-free 的多边缘算法——与 MSBM/MMtSBM 的 IMF 路线形成对照。

18. **A Lagrangian View of Flow Matching.** arXiv, 2026. [paper](https://arxiv.org/abs/2609.00198)

 *Peyman Milanfar*

 > Flow matching 的 Lagrangian（粒子）视角：由连续去噪器的局部 Taylor 展开导出「目标身份守恒」不变性，得到拟线性平流 PDE，特征线法给出单步生成的解析条件。

<a name="3-sampling-stochastic-optimal-control"></a>
## [3. Sampling & Stochastic Optimal Control](#content)
*采样与随机最优控制*

<a name="31-adjoint-soc-samplers-energy-only"></a>
### [3.1. Adjoint / SOC Samplers (energy-only)](#content)
*Adjoint / SOC 采样器（仅能量）*

1. **Adjoint Sampling: Highly Scalable Diffusion Samplers via Adjoint Matching.** ICML, 2025. [paper](https://arxiv.org/abs/2504.11713), [code](https://github.com/facebookresearch/adjoint_sampling), [📄 PDF](papers/2504.11713_adjoint_sampling.pdf), [📘 精读](reports/2504.11713_adjoint_sampling.md), [🀄 译本](papers_zh/2504.11713_adjoint_sampling.zh.pdf) ⭐

 *Aaron Havens, Benjamin Kurt Miller, Bing Yan, Carles Domingo-Enrich, Anuroop Sriram, Daniel Levine, Bin Hu, Brandon Amos, Brian Karrer, Xiang Fu, Guan-Horng Liu, Ricky T. Q. Chen 等*

2. **Non-equilibrium Annealed Adjoint Sampler.** NeurIPS, 2025. [paper](https://arxiv.org/abs/2506.18165)

 *Jaemoo Choi, Yongxin Chen, Molei Tao, Guan-Horng Liu*

 > NAAS：把退火（非平稳）参考动力学作为 SOC 的 base SDE，让参考轨迹自带朝目标推进的信息；交替优化两个控制并用 lean adjoint 训练，回放缓冲避免重要性加权；在经典能量景观与分子 Boltzmann 分布上优于 PIS / DDS。

3. **Adjoint Schrödinger Bridge Sampler.** NeurIPS, 2025 (Oral). [paper](https://arxiv.org/abs/2506.22565), [code](https://github.com/facebookresearch/adjoint_samplers), [📄 PDF](papers/2506.22565_adjoint_schrodinger_bridge_sampler.pdf), [📘 精读](reports/2506.22565_adjoint_schrodinger_bridge_sampler.md), [🀄 译本](papers_zh/2506.22565_adjoint_schrodinger_bridge_sampler.zh.pdf) ⭐

 *Guan-Horng Liu, Jaemoo Choi, Yongxin Chen, Benjamin Kurt Miller, Ricky T. Q. Chen*

4. **MDNS: Masked Diffusion Neural Sampler via Stochastic Optimal Control.** NeurIPS, 2025. [paper](https://arxiv.org/abs/2508.10684)

 *Yuchen Zhu, Wei Guo, Jaemoo Choi, Guan-Horng Liu, Yongxin Chen, Molei Tao*

 > MDNS（NeurIPS 2025，据作者主页与 CV）：CTMC 随机最优控制视角训练离散神经采样器（Ising/Potts），用免微分的路径测度对齐目标与加权去噪交叉熵实现高维可扩展；与 DASBS 同属离散能量采样线。

5. **Tilt Matching for Scalable Sampling and Fine-Tuning.** arXiv, 2025. [paper](https://arxiv.org/abs/2512.21829)

 *Peter Potaptchik, Cheuk-Kit Lee, Michael S. Albergo*

 > Tilt Matching：由「flow matching 速度」与「reward 倾斜后目标的速度」之间的动力学方程出发隐式解 SOC，同时服务于未归一化密度采样与生成模型微调；新速度是比 flow matching 方差更低的目标的极小点。

6. **Data-to-Energy Stochastic Dynamics.** ICLR, 2026. [paper](https://arxiv.org/abs/2509.26364)

 *Kirill Tamogashev, Esmeralda S. Whitammer*

 > Data-to-Energy：现有 SB 算法都要求两端有样本；本文处理一端只有样本、另一端只有能量的设定，学从数据到能量定义分布的随机动力学。

7. **Enhancing Diffusion-Based Sampling with Molecular Collective Variables (WT-ASBS).** ICLR, 2026. [paper](https://arxiv.org/abs/2510.11923), [code](https://github.com/facebookresearch/wt-asbs)

 *Juno Nam, Bálint Máté, Artur P. Toshev, Manasa Kaniselvan, Rafael Gómez-Bombarelli, Ricky T. Q. Chen, Brandon Wood, Guan-Horng Liu, Benjamin Kurt Miller*

 > WT-ASBS：把 well-tempered metadynamics 的在线排斥偏置沿集体变量（CV）加进 ASBS——内环训到有偏目标收敛、外环按 CV 投影叠加高斯核；发现稀有构象并重加权还原 Boltzmann 分布，首次用扩散采样器刻画含键断裂/形成的反应面。

8. **Functional Adjoint Sampler: Scalable Sampling on Infinite Dimensional Spaces.** ICML, 2026. [paper](https://arxiv.org/abs/2511.06239), [📄 PDF](papers/2511.06239_functional_adjoint_sampler.pdf), [📘 精读](reports/2511.06239_functional_adjoint_sampler.md), [🀄 译本](papers_zh/2511.06239_functional_adjoint_sampler.zh.pdf) ⭐

 *Byoungwoo Park, Juho Lee, Guan-Horng Liu*

9. **Discrete Adjoint Schrödinger Bridge Sampler.** ICML, 2026. [paper](https://arxiv.org/abs/2602.08243), [📄 PDF](papers/2602.08243_discrete_adjoint_schrodinger_bridge_sampler.pdf), [📘 精读](reports/2602.08243_discrete_adjoint_schrodinger_bridge_sampler.md), [🀄 译本](papers_zh/2602.08243_discrete_adjoint_schrodinger_bridge_sampler.zh.pdf) ⭐

 *Wei Guo, Yuchen Zhu, Xiaochen Du, Juno Nam, Yongxin Chen, Rafael Gómez-Bombarelli, Guan-Horng Liu, Molei Tao, Jaemoo Choi*

10. **MetaDNS: Enhancing Exploration in Discrete Neural Samplers via Well-Tempered Metadynamics.** ICML, 2026. [paper](https://arxiv.org/abs/2605.21722)

 *Xiaochen Du, Juno Nam, Jaemoo Choi, Wei Guo, Sathya Edamadaka et al.*

 > MetaDNS：MDNS 类离散神经采样器在高能垒间模式塌缩；把 well-tempered metadynamics 的自适应历史依赖偏置接进离散扩散/自回归采样器，用于自由能估计与相变（WT-ASBS 的离散对应）。

11. **Hard-Constrained Sampling on Embedded Riemannian Manifolds via Adjoint Schrödinger Bridges.** arXiv, 2026. [paper](https://arxiv.org/abs/2608.25838)

 *Mattia Mosso, Jaemoo Choi, Heng Yang*

 > 在嵌入的紧致黎曼流形上做 adjoint SB 采样：受控扩散在流形上内蕴定义、可行性在状态空间层面强制，面向流形支撑的 Boltzmann 分布（物理应用验证）。

<a name="32-soc-for-reward-fine-tuning-rl-adjoint-matching-lineage"></a>
### [3.2. SOC for Reward Fine-tuning & RL (Adjoint Matching lineage)](#content)
*SOC 奖励微调与 RL（Adjoint Matching 谱系）*

1. **Training Diffusion Models with Reinforcement Learning.** ICLR, 2024. [paper](https://arxiv.org/abs/2305.13301), [code](https://github.com/jannerm/ddpo)

 *Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, Sergey Levine*

 > DDPO：把去噪过程当 MDP 用 policy gradient 微调扩散模型。

2. **Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models with Memoryless Stochastic Optimal Control.** ICLR, 2025 (Spotlight). [paper](https://arxiv.org/abs/2409.08861), [📄 PDF](papers/2409.08861_adjoint_matching.pdf), [🀄 译本](papers_zh/2409.08861_adjoint_matching.zh.pdf)

 *Carles Domingo-Enrich, Michal Drozdzal, Brian Karrer, Ricky T. Q. Chen*

 > Adjoint Matching：memoryless SOC + lean adjoint 回归做 reward 微调；AS/ASBS/DAM/FAS 的共同源头。

3. **Fine-Tuning Discrete Diffusion Models via Reward Optimization with Applications to DNA and Protein Design.** arXiv, 2024. [paper](https://arxiv.org/abs/2410.13643)

 *Chenyu Wang, Masatoshi Uehara, Yichun He, Amy Wang, Tommaso Biancalani et al.*

 > DRAKES：对 masked 离散扩散做 reward 微调（Gumbel-Softmax 反传），目标是既像预训练分布又高 reward；DAM 的直接对照。

4. **TR2-D2: Tree Search Guided Trajectory-Aware Fine-Tuning for Discrete Diffusion.** arXiv, 2025. [paper](https://arxiv.org/abs/2509.25171)

 *Sophia Tang, Yuchen Zhu, Molei Tao, Pranam Chatterjee*

 > TR2-D2：SOC 式离散扩散微调依赖当前模型 rollout、易强化低回报轨迹；用树搜索引导的轨迹感知微调筛选高回报路径。

5. **Q-learning with Adjoint Matching.** arXiv, 2026. [paper](https://arxiv.org/abs/2601.14234)

 *Qiyang Li, Sergey Levine*

 > QAM：把「对参数化 Q 函数优化 diffusion/flow 策略」重写为带学习 critic 的 memoryless SOC，用 adjoint matching 利用 critic 的一阶信息而不反传多步去噪过程；连续动作 RL 的 TD 类算法。

6. **Discrete Adjoint Matching.** ICLR, 2026. [paper](https://arxiv.org/abs/2602.07132), [📄 PDF](papers/2602.07132_discrete_adjoint_matching.pdf), [📘 精读](reports/2602.07132_discrete_adjoint_matching.md), [🀄 译本](papers_zh/2602.07132_discrete_adjoint_matching.zh.pdf) ⭐

 *Oswin So, Brian Karrer, Chuchu Fan, Ricky T. Q. Chen, Guan-Horng Liu*

7. **A Sample-Wise Adjoint Regression Framework for Mean-Field Control with Connections to Adjoint Matching.** arXiv, 2026. [paper](https://arxiv.org/abs/2604.06675)

 *Hui Sun*

 > 平均场控制的样本级 adjoint 回归：不显式求解 adjoint 过程 (Y,Z)，而是构造其离散化的无偏样本估计并递归回归近似 Hamiltonian 控制梯度——与 adjoint matching 的「回归而非反传」思路同源。

8. **Adjoint Matching through the Lens of the Stochastic Maximum Principle in Optimal Control.** arXiv, 2026. [paper](https://arxiv.org/abs/2604.08580)

 *Carles Domingo-Enrich, Jiequn Han*

 > 从随机极大值原理（SMP）重推 Adjoint Matching：给出控制相关漂移/扩散与凸运行成本下的一般 Hamiltonian adjoint matching 目标，证明其一阶变分与原 SOC 目标一致；lean adjoint 是状态无关扩散下的特例。

9. **A unified perspective on fine-tuning and sampling with diffusion and flow models.** arXiv, 2026. [paper](https://arxiv.org/abs/2605.00229)

 *Carles Domingo-Enrich, Yuanqi Du, Michael S. Albergo*

 > 统一视角：把「从未归一化密度采样」与「reward 微调」都写成对 base 密度的指数倾斜，比较 SOC（adjoint / score matching）与非平衡热力学路线；给出偏差-方差分解（AM/AS 与 Novel Score Matching 梯度方差有限，Target/Conditional Score Matching 无界）与 lean adjoint ODE 的范数界。

10. **Entropy-Regularized Adjoint Matching for Offline Reinforcement Learning.** arXiv, 2026. [paper](https://arxiv.org/abs/2605.06156)

 *Abdelghani Ghanem, Mounir Ghogho*

 > ME-AM：QAM 受限于固定行为分布带来的流行度偏差与支撑绑定；最大熵 adjoint matching 在离线 RL 中放开对低密度高回报动作的抑制，避免退回单峰残差高斯策略。

11. **Improved techniques for fine-tuning flow models via adjoint matching: a deterministic control pipeline.** arXiv, 2026. [paper](https://arxiv.org/abs/2605.06583)

 *Zhengyi Guo, Jiayuan Sheng, David D. Yao, Wenpin Tang*

 > 确定性 AM：把 flow 模型的偏好对齐写成速度场上的最优控制，直接向价值梯度诱导的目标回归；截断 adjoint 只算轨迹末段，并推广到非 KL 正则；SiT-XL/2 与 FLUX 实验。

12. **Reinforce Adjoint Matching: Scaling RL Post-Training of Diffusion and Flow-Matching Models.** arXiv, 2026. [paper](https://arxiv.org/abs/2605.10759)

 *Andreas Bergmeister, Stefanie Jegelka, Nikolas Nüsken, Carles Domingo-Enrich, Jakiw Pidstrigach*

 > Reinforce AM：证明 KL 正则 reward 最大化的最优生成过程只倾斜干净端点分布、噪声律不变，结合 AM 最优性条件与 REINFORCE 恒等式，把 RL 后训练保留为预训练式的回归结构，免去昂贵 SDE rollout 与 reward 梯度。

13. **Efficient Adjoint Matching for Fine-tuning Diffusion Models.** arXiv, 2026. [paper](https://arxiv.org/abs/2605.11480)

 *Jeongwoo Shin, Dongsoo Shin, Yuchen Zhu, Wei Guo, Yongxin Chen et al.*

 > Efficient AM：指出 AM 的两大开销（memoryless 动力学下的全轨迹随机模拟、逐轨迹反向 adjoint ODE）都源自预训练模型的非平凡 base drift，据此设计更省算的变体。

14. **Trust Region Q Adjoint Matching.** arXiv, 2026. [paper](https://arxiv.org/abs/2605.27079)

 *Yonghoon Dong, Kyungmin Lee, Changyeon Kim, Jaehyuk Kim, Jinwoo Shin*

 > TRQAM：QAM 会放大病态 critic 的小误差导致崩溃；用投影对偶下降自适应控制与预训练 flow 策略间的路径空间 KL（优化 SOC 中的信赖域参数 λ），稳定 off-policy 微调。

15. **Unsupervised Diffusion Solver for Combinatorial Optimization via Combinatorial Adjoint Matching.** ICML, 2026. [paper](https://arxiv.org/abs/2605.30920)

 *Shengyu Feng, Tarun Suresh, Yiming Yang*

 > CAM：把扩散式组合优化写成 CTMC 上的随机控制，引入离散 adjoint 动力学在离散生成轨迹上传播优化信号，实现无监督（无需近优解标签）训练的离散扩散求解器。

16. **Scalable Maximum Entropy Reinforcement Learning for Diffusion Policies via Adjoint Matching.** arXiv, 2026. [paper](https://arxiv.org/abs/2606.22630)

 *Serge Thilges, Onur Celik, Denis Blessing, Emiliyan Gospodinov, Gerhard Neumann*

 > 用 adjoint matching 做在线最大熵 RL 的 diffusion 策略优化：simulation-free、无需显式似然或反传去噪过程，并给出若干稳定性扩展。

<a name="33-diffusion-samplers-boltzmann-generators-competitors"></a>
### [3.3. Diffusion Samplers, Boltzmann Generators & Competitors](#content)
*扩散采样器 / Boltzmann 生成器 / 竞品*

1. **Path Integral Sampler: a stochastic control approach for sampling.** ICLR, 2022. [paper](https://arxiv.org/abs/2111.15141), [code](https://github.com/qsh-zh/pis)

 *Qinsheng Zhang, Yongxin Chen*

 > PIS：路径积分控制视角的扩散采样器，SOC 采样线源头。

2. **Denoising Diffusion Samplers.** ICLR, 2023. [paper](https://arxiv.org/abs/2302.13834), [code](https://github.com/franciscovargas/denoising_diffusion_samplers)

 *Francisco Vargas, Will Grathwohl, Arnaud Doucet*

 > DDS：去噪扩散采样器，KL 控制目标。

3. **Transport meets Variational Inference: Controlled Monte Carlo Diffusions.** ICLR, 2024. [paper](https://arxiv.org/abs/2307.01050)

 *Francisco Vargas, Shreyas Padhy, Denis Blessing, Nikolas Nusken*

 > CMCD：controlled Monte Carlo diffusions，把 transport 与变分推断统一。

4. **Iterated Denoising Energy Matching for Sampling from Boltzmann Densities.** ICML, 2024. [paper](https://arxiv.org/abs/2402.06121), [code](https://github.com/jarridrb/DEM)

 *Tara Akhound-Sadegh, Jarrid Rector-Brooks, Avishek Joey Bose, Sarthak Mittal, Pablo Lemos et al.*

 > iDEM：simulation-free 能量采样竞品，Boltzmann 生成基准（DW4/LJ13/LJ55）。

5. **NETS: A Non-Equilibrium Transport Sampler.** ICML, 2025. [paper](https://arxiv.org/abs/2410.02711)

 *Michael S. Albergo, Eric Vanden-Eijnden*

 > NETS：非平衡 transport 采样器（annealed Langevin + 学习的 transport）。

6. **Underdamped Diffusion Bridges with Applications to Sampling.** ICLR, 2025. [paper](https://arxiv.org/abs/2503.01006)

 *Denis Blessing, Julius Berner, Lorenz Richter, Gerhard Neumann*

 > Underdamped Diffusion Bridges：欠阻尼动力学的 bridge 采样器。

7. **Proximal Diffusion Neural Sampler.** ICLR, 2026. [paper](https://arxiv.org/abs/2510.03824)

 *Wei Guo, Jaemoo Choi, Yuchen Zhu, Molei Tao, Yongxin Chen*

 > PDNS：把神经采样器训练视为路径测度上的 SOC，用近端点法（proximal point）处理多峰高能垒目标的模式塌缩。

8. **Reinforced sequential Monte Carlo for amortised sampling.** ICML, 2026. [paper](https://arxiv.org/abs/2510.11711), [code](https://github.com/hyeok9855/ReinforcedSMC)

 *Sanghyeok Choi, Sarthak Mittal, Víctor Elvira, Jinkyoo Park, Esmeralda S. Whitammer*

 > Reinforced SMC：建立序贯蒙特卡洛与最大熵 RL 训练的神经序贯采样器之间的联系（策略/价值 ↔ proposal/twist），用 SMC 样本作为行为策略做 off-policy 训练。

9. **One-Step Diffusion Samplers via Self-Distillation and Deterministic Flow.** AISTATS, 2026. [paper](https://arxiv.org/abs/2512.05251)

 *Pascal Jutras-Dube, Jiaru Zhang, Ziran Wang, Ruqi Zhang*

 > 一步扩散采样器：学步长条件化的 ODE，用状态空间一致性损失让一大步复现多小步轨迹；并指出常规 ELBO 估计在少步下失真。

10. **Diffusion-based Annealed Boltzmann Generators : benefits, pitfalls and hopes.** TMLR, 2026. [paper](https://arxiv.org/abs/2601.21026)

 *Louis Grenioux, Maxence Noble*

 > 基于退火 MC 的扩散式 Boltzmann 生成器：分析其相对重要性采样 BG 的收益、陷阱与前景（高维多峰、无需 tractable 似然）。

11. **Efficient Training of Boltzmann Generators Using Off-Policy Log-Dispersion Regularization.** arXiv, 2026. [paper](https://arxiv.org/abs/2602.03729)

 *Henrik Schopmans, Christopher von Klitzing, Pascal Friederich*

 > Off-policy log-dispersion 正则（LDR）：推广 log-variance 目标，提高 Boltzmann 生成器在昂贵能量评估下的数据效率。

12. **Coarse-Grained Boltzmann Generators.** ICML, 2026. [paper](https://arxiv.org/abs/2602.10637)

 *Weilong Chen, Bojun Zhao, Jan Eckwert, Julija Zavadlav*

 > CG-BG：粗粒化代理可扩到更大系统但缺重加权保证渐近正确统计；给出带重加权的粗粒化 Boltzmann 生成器框架。

13. **Bridge Matching Sampler: Scalable Sampling via Generalized Fixed-Point Diffusion Matching.** arXiv, 2026. [paper](https://arxiv.org/abs/2603.00530)

 *Denis Blessing, Lorenz Richter, Julius Berner, Egor Malitskiy, Gerhard Neumann*

 > Bridge Matching Sampler（BMS）：把近期各类最小二乘「匹配」式采样器统一为源于 Nelson 关系的不动点迭代，单一稳定目标学任意先验→目标的随机传输；阻尼变体缓解模式塌缩。

14. **Jeffreys Flow: Robust Boltzmann Generators for Rare Event Sampling via Parallel Tempering Distillation.** arXiv, 2026. [paper](https://arxiv.org/abs/2604.05303)

 *Guang Lin, Christian Moya, Di Qi, Xuda Ye*

 > Jeffreys Flow：反向 KL 训练的 BG 常模式塌缩；用对称 Jeffreys 散度蒸馏平行回火轨迹的经验数据，兼顾局部精度与全局模态覆盖。

15. **Autoregressive Boltzmann Generators.** ICML, 2026 (Spotlight). [paper](https://arxiv.org/abs/2606.27361)

 *Danyal Rehman, Charlie B. Tan, Yoshua Bengio, Avishek Joey Bose, Alexander Tong*

 > 自回归 Boltzmann 生成器：BG 依赖精确似然 + 重要性采样校正，但归一化流受可逆性或似然代价限制；改用自回归模型作为骨干。

16. **Few-Step Boltzmann Generators via Scalable Likelihood Flow Maps.** arXiv, 2026. [paper](https://arxiv.org/abs/2606.29110)

 *RuiKang OuYang, Hanlin Yu, Xinyue Ai, Yutong He, Nicholas M. Boffi et al.*

 > SCALLOP：少步 flow map 模型缺乏对应的少步似然估计；在 F2D2 基础上给出可扩展的似然蒸馏，替代高方差的 Hutchinson 估计，得到少步 Boltzmann 生成器。

17. **JANUS: A Multi-modal Foundation Neural Sampler for Disordered Materials.** arXiv, 2026. [paper](https://arxiv.org/abs/2608.19116)

 *Denis Blessing, Mouyang Cheng, Maximilian Schebek, Jutta Rogal, Mingda Li et al.*

 > JANUS：耦合连续与 masked 离散扩散的多模态神经采样器（等变 GNN），仅由能量评估训练；Ising 与等压 ΔμNPT 合金系统上以少三个量级的能量评估复现 MC 平衡观测量与自由能。

<a name="4-applications"></a>
## [4. Applications](#content)
*应用*

<a name="41-image-translation-restoration-editing"></a>
### [4.1. Image Translation, Restoration & Editing](#content)
*图像翻译 / 修复 / 编辑*

1. **SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations.** ICLR, 2022. [paper](https://arxiv.org/abs/2108.01073), [code](https://github.com/ermongroup/SDEdit)

 *Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu et al.*

 > SDEdit：噪声-去噪的零样本编辑基线。

2. **Dual Diffusion Implicit Bridges for Image-to-Image Translation.** ICLR, 2023. [paper](https://arxiv.org/abs/2203.08382), [code](https://github.com/suxuann/ddib)

 *Xuan Su, Jiaming Song, Chenlin Meng, Stefano Ermon*

 > DDIB：两段 DDIM 拼接实现零样本图像翻译；两段 SB 拼接≠跨域 OT 的对照案例。

3. **Unpaired Image-to-Image Translation via Neural Schrödinger Bridge.** ICLR, 2024. [paper](https://arxiv.org/abs/2305.15086), [code](https://github.com/cyclomon/UNSB)

 *Beomsu Kim, Gihyun Kwon, Kwanyoung Kim, Jong-Chul Ye*

 > UNSB：神经 SB 做 unpaired 图像翻译，配对抗正则与 patch 判别。

4. **Residual Diffusion Bridge Model for Image Restoration.** CVPR, 2026. [paper](https://arxiv.org/abs/2510.23116), [code](https://github.com/MiliLab/RDBM)

 *Hebaixu Wang, Jing Zhang, Haoyang Chen, Haonan Guo, Di Wang, Jiayi Ma, Bo Du*

 > RDBM：重新推导广义 diffusion bridge 的前/反向 SDE 闭式，用成对图像的残差调制噪声注入与去除，只扰动退化区域、保护完好区域；五类通用修复任务平均 +1.55 dB PSNR，并证明现有桥模型是其特例（CVF Open Access，pp. 8375–8386）。

5. **Energy-oriented Diffusion Bridge for Image Restoration with Foundational Diffusion Models.** ICLR, 2026. [paper](https://arxiv.org/abs/2604.10983)

 *Jinhui Hou, Zhiyu Zhu, Junhui Hou*

 > E-Bridge：更短时域 + 从「退化图像与高斯噪声的熵正则混合点」起步的桥过程，理论上降低所需轨迹能量；借 consistency 模型学单步映射。

6. **Stochastic Optimal Control Sampling for Diffusion Inverse Problems.** ECCV, 2026. [paper](https://arxiv.org/abs/2606.28785)

 *Jie Zhang, Youmei Qiu, Hanling Tian, Jingyuan Zhang, Xiang Yin, Xiaolin Huang*

 > SOCS：把去噪过程看作动力系统、用 SOC 注入控制信号引导采样轨迹贴合观测，避免此前 SOC 逆问题方法对整条轨迹优化的高开销。

7. **Bi-Bridge: Bidirectional Diffusion Bridges for Low-Light Image Enhancement.** CVPR, 2026. [paper](https://openaccess.thecvf.com/content/CVPR2026/html/Hua_Bi-Bridge_Bidirectional_Diffusion_Bridges_for_Low-Light_Image_Enhancement_CVPR_2026_paper.html)

 *Zeyu Hua, Hui Li, Yu Wang, Song Wang, Congchao Zhu, Caixia Zheng*

 > Bi-Bridge：利用 DDBM 高斯桥均值对端点的对称性，训练时随机交换起终点、单个 U-Net 同时学增强与退化两个方向，用双向一致性约束抑制单向生成模型对结构的扭曲（CVF Open Access，pp. 37455–37464）。

8. **Remote Sensing Image Super-Resolution via Progressive Diffusion Schrödinger Bridge.** Remote Sensing 18(3):532, 2026. [paper](https://www.mdpi.com/2072-4292/18/3/532)

 > PDSB：把大尺度 SR 切成级联子桥，逐级以小尺度结果为条件重建，缓解 LR/HR 配对中的几何位移违反高斯假设的问题；Gaofen-6 (2 m) ↔ Sentinel-2 (10 m) 数据上 FID 8.294，为第二名的一半。

<a name="42-video-3d-speech-audio-multimodal"></a>
### [4.2. Video, 3D, Speech, Audio & Multimodal](#content)
*视频 / 3D / 语音 / 音频 / 多模态*

1. **Schrodinger Bridges Beat Diffusion Models on Text-to-Speech Synthesis.** arXiv, 2023. [paper](https://arxiv.org/abs/2312.03491)

 *Zehua Chen, Guande He, Kaiwen Zheng, Xu Tan, Jun Zhu*

 > Bridge-TTS：用 SB 替代扩散做 TTS，少步下质量更优。

2. **Schrödinger Bridge for Generative Speech Enhancement.** Interspeech, 2024. [paper](https://arxiv.org/abs/2407.16074)

 *Ante Jukić, Roman Korostik, Jagadeesh Balam, Boris Ginsburg*

 > 语音增强的 SB 模型。

3. **Time-Correlated Video Bridge Matching.** arXiv, 2025. [paper](https://arxiv.org/abs/2510.12453)

 *Viacheslav Vasilev, Arseny Ivanov, Nikita Gushchin, Maria Kovaleva, Alexander Korotin*

 > TCVBM：把 bridge matching 推到时间相关的视频序列，在桥内显式建模序列间依赖，把时序相关性直接纳入采样。

4. **Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge.** NeurIPS, 2025. [paper](https://arxiv.org/abs/2510.20819)

 *Nimrod Berman, Omkar Joglekar, Eitan Kosman, Dotan Di Castro, Omri Azencot*

 > LDDBM：DDBM 的 latent 变量扩展，在共享 latent 空间学任意模态间的桥，不要求同维度、高斯源先验或模态专用架构。

5. **Walking the Schrödinger Bridge: A Direct Trajectory for Text-to-3D Generation.** NeurIPS, 2025. [paper](https://arxiv.org/abs/2511.05609), [code](https://github.com/emmaleee789/TraCe)

 *Ziying Li, Xuequan Lu, Xinkui Zhao, Guanjie Cheng, Shuiguang Deng, Jianwei Yin*

 > Walking the SB（TraCe）：证明 SDS 是 SB 反向过程在一端为高斯噪声时的退化特例，改为在当前渲染分布与目标分布间学直接传输轨迹，小 CFG 下缓解 text-to-3D 的过饱和/过平滑。

6. **Schrödinger Bridge Mamba for One-Step Speech Enhancement.** Interspeech, 2026. [paper](https://arxiv.org/abs/2510.16834)

 *Jing Yang, Sirui Wang, Chao Wu, Lei Guo, Fan Fan*

 > SB Mamba：SB 训练范式 + Mamba 骨干，联合去噪去混响一步推理即超过生成式与判别式基线，且实时因子满足流式。

7. **Rethinking Flow and Diffusion Bridge Models for Speech Enhancement.** AAAI, 2026. [paper](https://arxiv.org/abs/2602.18355)

 *Dahan Wang, Jun Gao, Tong Lei, Yuxiang Hu, Changbao Zhu et al.*

 > 把语音增强里的 flow / diffusion bridge 统一为成对数据间不同均值方差的高斯概率路径，指出数据预测损失下每一步采样在理论上等价于一次预测式增强，并据此给出改进的 bridge。

<a name="43-science-single-cell-molecules-chemistry-physics"></a>
### [4.3. Science: Single-cell, Molecules, Chemistry & Physics](#content)
*科学：单细胞 / 分子 / 化学 / 物理*

1. **TrajectoryNet: A Dynamic Optimal Transport Network for Modeling Cellular Dynamics.** ICML, 2020. [paper](https://arxiv.org/abs/2002.04461), [code](https://github.com/KrishnaswamyLab/TrajectoryNet)

 *Alexander Tong, Jessie Huang, Guy Wolf, David van Dijk, Smita Krishnaswamy*

 > TrajectoryNet：动态 OT 做单细胞轨迹推断的起点。

2. **Improving Generative Model-based Unfolding with Schrödinger Bridges.** Phys. Rev. D 109, 076011 (2024). [paper](https://arxiv.org/abs/2308.12351), [code](https://github.com/ViniciusMikuni/SBUnfold), [📄 PDF](papers/2308.12351_sb_unfold.pdf), [📘 精读](reports/2308.12351_sb_unfold.md), [🀄 译本](papers_zh/2308.12351_sb_unfold.zh.pdf) ⭐

 *Sascha Diefenbacher, Guan-Horng Liu, Vinicius Mikuni, Benjamin Nachman, Weili Nie*

3. **React-OT: Optimal Transport for Generating Transition State in Chemical Reactions.** Nature Machine Intelligence 7, 615-626 (2025). [paper](https://arxiv.org/abs/2404.13430), [code](https://github.com/deepprinciple/react-ot), [📄 PDF](papers/2404.13430_react_ot.pdf), [📘 精读](reports/2404.13430_react_ot.md), [🀄 译本](papers_zh/2404.13430_react_ot.zh.pdf) ⭐

 *Chenru Duan, Guan-Horng Liu, Yuanqi Du, Tianrong Chen, Qiyuan Zhao, Haojun Jia, Carla P. Gomes, Evangelos A. Theodorou, Heather J. Kulik*

4. **Departures: Distributional Transport for Single-Cell Perturbation Prediction with Neural Schrödinger Bridges.** arXiv, 2025. [paper](https://arxiv.org/abs/2511.13124)

 *Changxi Chi, Yufei Huang, Jun Xia, Jiangbin Zheng, Yunfan Liu et al.*

 > Departures：用神经 SB 直接对齐对照组与扰动组单细胞分布（unpaired，测序不可重复观测同一细胞），显式条件化扰动做预测；AAAI 2026 版本见 reports/deep_research_learning_resources.md。

5. **FreeBridge: Variational Schrödinger Bridges for Cellular Transition Dynamics.** MICCAI, 2026. [paper](https://arxiv.org/abs/2606.11286)

 *Xurui Wang, Qin Ren, Jun Ma, Haibin Ling, Chenyu You*

 > FreeBridge：高内涵成像下细胞被固定、只能看到两端边缘；以实例分割的单细胞表示为原子状态构成固定细胞流形，在其上学 SB 以避免中间态穿越无支撑区域。

<a name="44-embodied-ai-sim2real-cross-domain-transfer-rl"></a>
### [4.4. Embodied AI: Sim2Real, Cross-domain Transfer & RL](#content)
*具身智能：sim2real / 跨域迁移 / RL*

1. **Affine Transport for Sim-to-Real Domain Adaptation.** arXiv, 2021. [paper](https://arxiv.org/abs/2105.11739), [📄 PDF](papers/2105.11739_affine_transport_sim2real.pdf), [📘 精读](reports/2105.11739_affine_transport_sim2real.md), [🀄 译本](papers_zh/2105.11739_affine_transport_sim2real.zh.pdf) ⭐

 *Anton Mallasto, Karol Arndt, Markus Heinonen, Samuel Kaski, Ville Kyrki*

2. **One-Shot Imitation under Mismatched Execution.** ICRA, 2025. [paper](https://arxiv.org/abs/2409.06615), [code](https://github.com/portal-cornell/rhyme), [📄 PDF](papers/2409.06615_rhyme_one_shot_mismatched_execution.pdf), [📘 精读](reports/2409.06615_rhyme_one_shot_mismatched_execution.md), [🀄 译本](papers_zh/2409.06615_rhyme_one_shot_mismatched_execution.zh.pdf) ⭐

 *Kushal Kedia, Prithwish Dan, Angela Chao, Maximus A. Pace, Sanjiban Choudhury*

3. **Generalizable Domain Adaptation for Sim-and-Real Policy Co-Training.** NeurIPS, 2025. [paper](https://arxiv.org/abs/2509.18631), [project](https://ot-sim2real.github.io/), [📄 PDF](papers/2509.18631_guided_ot_sim_real_policy_cotraining.pdf), [📘 精读](reports/2509.18631_guided_ot_sim_real_policy_cotraining.md), [🀄 译本](papers_zh/2509.18631_guided_ot_sim_real_policy_cotraining.zh.pdf) ⭐

 *Shuo Cheng, Liqian Ma, Zhenyang Chen, Ajay Mandlekar, Caelan Garrett, Danfei Xu*

4. **EgoBridge: Domain Adaptation for Generalizable Imitation from Egocentric Human Data.** NeurIPS 2025 + CoRL 2025 (Oral). [paper](https://arxiv.org/abs/2509.19626), [📄 PDF](papers/2509.19626_egobridge.pdf), [📘 精读](reports/2509.19626_egobridge.md), [🀄 译本](papers_zh/2509.19626_egobridge.zh.pdf) ⭐

 *Ryan Punamiya, Dhruv Patel, Patcharapong Aphiwetsa, Pranav Kuppili, Lawrence Y. Zhu, Simar Kareer, Judy Hoffman, Danfei Xu*

5. **XFlowMP: Task-Conditioned Motion Fields for Generative Robot Planning with Schrodinger Bridges.** arXiv, 2025. [paper](https://arxiv.org/abs/2512.00022)

 *Khang Nguyen, Minh Nhat Vu*

 > XFlowMP：任务条件的生成式运动规划——用 SB 作为带 score 的条件流匹配学高阶动力学运动场，编码起止构型生成无碰撞、动力学可行的轨迹。

6. **BridgeDrive: Diffusion Bridge Policy for Closed-Loop Trajectory Planning in Autonomous Driving.** ICLR, 2026. [paper](https://arxiv.org/abs/2509.23589)

 *Shu Liu, Wenlin Chen, Weihao Li, Zheng Wang, Lijin Yang et al.*

 > BridgeDrive：锚点引导的 diffusion bridge 策略，把粗锚轨迹直接变换为上下文感知的精细规划，保持前向/反向过程一致性；面向自动驾驶闭环轨迹规划。

7. **Treatment Stitching with Schrödinger Bridge for Enhancing Offline Reinforcement Learning in Adaptive Treatment Strategies.** AAAI, 2026. [paper](https://arxiv.org/abs/2511.12075)

 *Dong-Hee Shin, Deok-Joong Lee, Young-Han Son, Tae-Eui Kam*

 > TreatStitch：用 SB 在相似中间患者状态间「缝合」既有治疗轨迹片段做离线 RL 数据增广（自适应治疗策略场景）。

8. **Sample from What You See: Visuomotor Policy Learning via Diffusion Bridge with Observation-Embedded Stochastic Differential Equation.** ICML, 2026. [paper](https://arxiv.org/abs/2512.07212), [code](https://github.com/jianghcsr/BridgePolicy)

 *Zhaoyang Liu, Mokai Pan, Zhongyi Wang, Kaizhen Zhu, Haotao Lu et al.*

 > BridgePolicy：把观测直接嵌入扩散 SDE，用 diffusion bridge 从'观测先验'而非高斯噪声出发采样动作；语义对齐器解决观测/动作维度异构；52 个仿真任务 + 5 个真机任务优于现有生成式策略。

9. **Distributional Reinforcement Learning with Diffusion Bridge Critics.** arXiv, 2026. [paper](https://arxiv.org/abs/2602.05783)

 *Shutong Ding, Yimiao Zhou, Ke Hu, Mokai Pan, Shan Zhong et al.*

 > DBC：用 diffusion bridge 直接建模 Q 值的逆 CDF 做分布式 critic，聚焦被忽视的扩散 critic 而非策略。

10. **FLAC: Maximum Entropy RL via Kinetic Energy Regularized Bridge Matching.** arXiv, 2026. [paper](https://arxiv.org/abs/2602.12829)

 *Lei Lv, Yunfei Li, Yu Luo, Fuchun Sun, Xiao Ma*

 > FLAC：把最大熵 RL 写成相对高熵参考过程（如均匀）的广义 SB，用速度场动能惩罚代替不可得的动作 log 密度，实现免似然的生成式策略熵正则。

11. **Bridging Dynamics Gaps via Diffusion Schrödinger Bridge for Cross-Domain Reinforcement Learning.** arXiv, 2026. [paper](https://arxiv.org/abs/2602.23737), [📄 PDF](papers/2602.23737_bdgxrl_diffusion_schrodinger_bridge.pdf), [📘 精读](reports/2602.23737_bdgxrl_diffusion_schrodinger_bridge.md), [🀄 译本](papers_zh/2602.23737_bdgxrl_diffusion_schrodinger_bridge.zh.pdf) ⭐

 *Hanping Zhang, Yuhong Guo*

12. **Path-Space Mirror Descent for On-Policy Reinforcement Learning under the Generalized Schrödinger Bridge.** arXiv, 2026. [paper](https://arxiv.org/abs/2603.21621)

 *Yuehu Gong, Zeyuan Wang, Yulin Chen, Shutong Ding, Qingyuan Zhou, Yanwei Fu*

 > GSB-MDPO：把 on-policy 生成式策略优化写成状态条件生成路径上的广义 SB，用路径空间镜像下降替代依赖动作似然的 PPO 式近端更新。

13. **Rectified Schrödinger Bridge Matching for Few-Step Visual Navigation.** arXiv, 2026. [paper](https://arxiv.org/abs/2604.05673)

 *Wuyang Luan, Junhui Li, Weiguang Zhao, Wenjian Zhang, Tieru Wu, Rui Ma*

 > RSBM：证明条件速度场在整个熵正则 ε 谱上函数形式不变（SB ε=1 ↔ OT ε→0 同一网络），降低 ε 线性减小速度方差；视觉导航 3 步积分达 92% 成功率、94.5% 余弦相似度，无需蒸馏。

14. **Optimal and Scalable MAPF via Multi-Marginal Optimal Transport and Schrödinger Bridges.** ICML, 2026 (Spotlight). [paper](https://arxiv.org/abs/2605.10917)

 *Usman A. Khan, Joseph W. Durham*

 > 把匿名多智能体寻路（MAPF）写成带 Markov 结构的多边缘 OT，指数规模坍缩为多项式 LP 且全单模；大规模时用 SB 概率化，化为可迭代求解的熵正则 MMOT。

<a name="45-optimal-transport-for-imitation-reward"></a>
### [4.5. Optimal Transport for Imitation & Reward](#content)
*最优传输用于模仿学习与奖励*

1. **Primal Wasserstein Imitation Learning.** ICLR, 2021. [paper](https://arxiv.org/abs/2006.04678), [code](https://github.com/google-research/google-research/tree/master/pwil), [📄 PDF](papers/2006.04678_primal_wasserstein_imitation_learning.pdf), [📘 精读](reports/2006.04678_primal_wasserstein_imitation_learning.md), [🀄 译本](papers_zh/2006.04678_primal_wasserstein_imitation_learning.zh.pdf) ⭐

 *Robert Dadashi, Léonard Hussenot, Matthieu Geist, Olivier Pietquin*

2. **Imitation Learning with Sinkhorn Distances.** ECML PKDD, 2022. [paper](https://arxiv.org/abs/2008.09167), [code](https://github.com/gpapagiannis/sinkhorn-imitation), [📄 PDF](papers/2008.09167_sinkhorn_imitation_learning.pdf), [📘 精读](reports/2008.09167_sinkhorn_imitation_learning.md), [🀄 译本](papers_zh/2008.09167_sinkhorn_imitation_learning.zh.pdf) ⭐

 *Georgios Papagiannis, Yunpeng Li*

3. **Cross-Domain Imitation Learning via Optimal Transport.** ICLR, 2022. [paper](https://arxiv.org/abs/2110.03684), [code](https://github.com/facebookresearch/gwil), [📄 PDF](papers/2110.03684_gwil_cross_domain_imitation_via_ot.pdf), [📘 精读](reports/2110.03684_gwil_cross_domain_imitation_via_ot.md), [🀄 译本](papers_zh/2110.03684_gwil_cross_domain_imitation_via_ot.zh.pdf) ⭐

 *Arnaud Fickinger, Samuel Cohen, Stuart Russell, Brandon Amos*

4. **Learn what matters: cross-domain imitation learning with task-relevant embeddings.** NeurIPS, 2022. [paper](https://arxiv.org/abs/2209.12093), [📄 PDF](papers/2209.12093_task_relevant_embeddings_cross_domain_il.pdf), [📘 精读](reports/2209.12093_task_relevant_embeddings_cross_domain_il.md), [🀄 译本](papers_zh/2209.12093_task_relevant_embeddings_cross_domain_il.zh.pdf) ⭐

 *Tim Franzmeyer, Philip H. S. Torr, João F. Henriques*

5. **Watch and Match: Supercharging Imitation with Regularized Optimal Transport.** CoRL, 2022 (PMLR 205:32-43, 2023). [paper](https://proceedings.mlr.press/v205/haldar23a.html), [code](https://github.com/siddhanthaldar/ROT), [project](https://rot-robot.github.io/), [📄 PDF](papers/2206.15469_rot_watch_and_match.pdf), [📘 精读](reports/2206.15469_rot_watch_and_match.md), [🀄 译本](papers_zh/2206.15469_rot_watch_and_match.zh.pdf) ⭐

 *Siddhant Haldar, Vaibhav Mathur, Denis Yarats, Lerrel Pinto*

6. **Offline Imitation from Observation via Primal Wasserstein State Occupancy Matching.** ICML, 2024. [paper](https://arxiv.org/abs/2311.01331), [code](https://github.com/KaiYan289/PW-DICE), [📄 PDF](papers/2311.01331_primal_wasserstein_state_occupancy.pdf), [📘 精读](reports/2311.01331_primal_wasserstein_state_occupancy.md), [🀄 译本](papers_zh/2311.01331_primal_wasserstein_state_occupancy.zh.pdf) ⭐

 *Kai Yan, Alexander G. Schwing, Yu-Xiong Wang*

7. **Robot Policy Learning with Temporal Optimal Transport Reward.** NeurIPS, 2024. [paper](https://arxiv.org/abs/2410.21795), [code](https://github.com/fuyw/TemporalOT), [📄 PDF](papers/2410.21795_temporal_ot_reward.pdf), [📘 精读](reports/2410.21795_temporal_ot_reward.md), [🀄 译本](papers_zh/2410.21795_temporal_ot_reward.zh.pdf) ⭐

 *Yuwei Fu, Haichao Zhang, Di Wu, Wei Xu, Benoit Boulet*

8. **Zero-Shot Offline Imitation Learning via Optimal Transport.** ICML, 2025. [paper](https://arxiv.org/abs/2410.08751), [code](https://github.com/martius-lab/zilot), [📄 PDF](papers/2410.08751_zero_shot_offline_il_ot.pdf), [📘 精读](reports/2410.08751_zero_shot_offline_il_ot.md), [🀄 译本](papers_zh/2410.08751_zero_shot_offline_il_ot.zh.pdf) ⭐

 *Thomas Rupf, Marco Bagatella, Nico Gürtler, Jonas Frey, Georg Martius*

<a name="5-codebases-benchmarks"></a>
## [5. Codebases & Benchmarks](#content)
*代码库与基准*

**Codebases**

1. [yuyang-shi/dsbm-pytorch](https://github.com/yuyang-shi/dsbm-pytorch) — NeurIPS 2023  
   DSBM / IMF 官方 PyTorch 实现（DSBM-IPF 与 DSBM-IMF，含自洽的高斯基准）。
2. [JTT94/diffusion_schrodinger_bridge](https://github.com/JTT94/diffusion_schrodinger_bridge) — NeurIPS 2021  
   DSB（IPF 求解）官方实现。
3. [ghliu/SB-FBSDE · ghliu/DeepGSB · facebookresearch/generalized-schrodinger-bridge-matching](https://github.com/ghliu) — ICLR 2022 / NeurIPS 2022 / ICLR 2024  
   Guan-Horng Liu 组的 SB 系列：SB-FBSDE（似然训练）、DeepGSB（平均场/广义 SB）、GSBM（任务代价的 SB 匹配）。
4. [NVlabs/I2SB](https://github.com/NVlabs/I2SB) — ICML 2023 · ⭐400+  
   I²SB 官方实现：paired 图像修复的 SB 基线（inpainting / deblurring / super-resolution / JPEG）。
5. [facebookresearch/adjoint_sampling · facebookresearch/adjoint_samplers](https://github.com/facebookresearch/adjoint_sampling) — ICML 2025 / NeurIPS 2025  
   Adjoint Sampling 与 Adjoint Schrödinger Bridge Sampler 官方实现，含分子构象生成基准（SPICE/eSEN）。
6. [ngushchin/LightSB · SKholkin/LightSB-Matching · Daniil-Selikhanovych/ASBM](https://github.com/ngushchin/LightSB) — ICLR 2024 / ICML 2024 / NeurIPS 2024  
   Korotin 组轻量 SB 求解器：LightSB（闭式高斯混合）、LightSB-M、ASBM（对抗式 D-IMF）。
7. [gregkseno/csbm · gregkseno/catsbench](https://github.com/gregkseno/catsbench) — ICML 2025 / ICLR 2026  
   离散空间 SB：Categorical SBM 实现与离散 SB 基准（DLightSB / DLightSB-M / α-CSBM）。
8. [thu-ml/DiffusionBridge](https://github.com/thu-ml/DiffusionBridge) — ICLR 2025 / NeurIPS 2024  
   DBIM（隐式 bridge 采样）与 CDBM（consistency bridge）统一代码库。
9. [UniDB-SOC/UniDB · 2769433owo/UniDB-plusplus](https://github.com/UniDB-SOC/UniDB) — ICML 2025 / 2025  
   SOC 统一的 diffusion bridge（UniDB）及其免训练快速采样（UniDB++）。
10. [sophtang/BranchSBM · bw-park/MSBM · panostheo98/3MSBM · TianrongChen/DMSB](https://github.com/sophtang/BranchSBM) — ICLR 2026 / 2025 / NeurIPS 2025 / NeurIPS 2023  
   结构化 SB：分叉（BranchSBM）、多边缘（MSBM、3MSBM、DMSB）——单细胞轨迹推断的主力代码。
11. [atong01/conditional-flow-matching (torchcfm)](https://github.com/atong01/conditional-flow-matching) — TMLR 2024 · ⭐1k+  
   Flow Matching / OT-CFM / [SF]²M（simulation-free SB）的通用库。
12. [facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching) — 2024  
   Flow Matching 官方库（含离散 FM），配套《Flow Matching Guide and Code》。
13. [PythonOT/POT · ott-jax/ott · jeanfeydy/geomloss](https://github.com/PythonOT/POT) — ⭐2.8k / 750 / 700  
   OT 工具箱：POT（Sinkhorn/UOT/GW）、OTT-JAX（可微 Sinkhorn、大规模）、GeomLoss（GPU 上的 Sinkhorn 散度）。
14. [jarridrb/DEM](https://github.com/jarridrb/DEM) — ICML 2024  
   iDEM 能量采样实现，附 GMM / DW4 / LJ13 / LJ55 等 Boltzmann 采样基准。
15. [jianghcsr/BridgePolicy](https://github.com/jianghcsr/BridgePolicy) — ICML 2026  
   BridgePolicy：观测嵌入 SDE 的 diffusion-bridge 视觉运动策略。

**Benchmarks & Datasets**

1. [catsbench（离散 SB 基准）](https://github.com/gregkseno/catsbench) — ICLR 2026  
   有解析解的离散分布对，严格评测离散 SB / EOT 求解器；目前唯一带 ground truth 的 SB 基准。
2. [能量采样基准（DEM / Adjoint Sampling）](https://github.com/facebookresearch/adjoint_sampling) — ICML 2024 / ICML 2025  
   GMM-40、Many-Well、DW4、LJ13/LJ55、alanine dipeptide，以及 SPICE 分子构象的 amortized 采样；指标 W₂ / TVD / ESS / 能量分布。
3. [unpaired 翻译常用协议](https://github.com/yuyang-shi/dsbm-pytorch) — —  
   EMNIST→MNIST、AFHQ 64/256（cat↔dog↔wild）、CelebA；指标 FID + NFE，DSBM / ASBM / SB Flow 共用。
4. [单细胞轨迹推断（EB / CITE-seq / Multiome）](https://github.com/KrishnaswamyLab/MIOFlow) — —  
   多时间点 scRNA-seq 快照，留一时间点评测（EMD / SWD）；TrajectoryNet、MIOFlow、DMSB、3MSBM、MSBM、BranchSBM 的共同战场。
5. [SimplerEnv（sim2real 策略评测）](https://github.com/simpler-env/SimplerEnv) — ⭐1.1k  
   仿真评估与真机成功率排序一致性（MMRV / Pearson）；具身 SB 方法落地评测的参考协议（见 topics/E11）。

**Workshops & Communities**

1. [ICLR 2026 DeLTa Workshop（Deep Learning for Theory & Applications）](https://www.iclr.cc/virtual/2026/workshop/10000780) — 2026  
   2026 年离散 SB / adjoint 采样多篇早期版本的发表地（如 DASBS workshop 版）。


<a name="6-chinese-deep-dive-reports-topic-notes"></a>
## [6. Chinese Deep-dive Reports & Topic Notes](#content)
*中文精读报告与专题笔记*

Every core paper has a Chinese deep-dive report (基本信息 / 一句话总结 / 方法核心 / 实验与结果 / 局限性 / 与相关方向的关系) and a layout-preserving Chinese translation. Start from [reports/INDEX.md](reports/INDEX.md) and the synthesis documents:

- [综合文献地图：OT / SB 如何迁移具身跨域数据](reports/synthesis.md)
- [Adjoint / Generalized / Structured Schrödinger Bridge 扩展文献综述](reports/sb_adjoint_extended_synthesis.md)
- [SB × OT × Sim2Real：深度调研、前沿论文与学习资源导航](reports/deep_research_learning_resources.md)
- [Guan-Horng Liu 研究工作专题：从最优控制到 SB、Adjoint Sampling 与 LLM Post-training](reports/guan_horng_liu_research_roadmap.md)

| arXiv | Paper | Venue | 精读 | 英文 PDF | 中文译本 |
|---|---|---|---|---|---|
| 2006.04678 | Primal Wasserstein Imitation Learning | ICLR 2021 | [📘](reports/2006.04678_primal_wasserstein_imitation_learning.md) | [📄](papers/2006.04678_primal_wasserstein_imitation_learning.pdf) | [🀄](papers_zh/2006.04678_primal_wasserstein_imitation_learning.zh.pdf) |
| 2105.11739 | Affine Transport for Sim-to-Real Domain Adaptation | arXiv | [📘](reports/2105.11739_affine_transport_sim2real.md) | [📄](papers/2105.11739_affine_transport_sim2real.pdf) | [🀄](papers_zh/2105.11739_affine_transport_sim2real.zh.pdf) |
| 2008.09167 | Imitation Learning with Sinkhorn Distances | ECML PKDD 2022 | [📘](reports/2008.09167_sinkhorn_imitation_learning.md) | [📄](papers/2008.09167_sinkhorn_imitation_learning.pdf) | [🀄](papers_zh/2008.09167_sinkhorn_imitation_learning.zh.pdf) |
| 2110.03684 | Cross-Domain Imitation Learning via Optimal Transport | ICLR 2022 | [📘](reports/2110.03684_gwil_cross_domain_imitation_via_ot.md) | [📄](papers/2110.03684_gwil_cross_domain_imitation_via_ot.pdf) | [🀄](papers_zh/2110.03684_gwil_cross_domain_imitation_via_ot.zh.pdf) |
| 2209.09893 | Deep Generalized Schrödinger Bridge | NeurIPS 2022 (Oral) | [📘](reports/2209.09893_deep_generalized_schrodinger_bridge.md) | [📄](papers/2209.09893_deep_generalized_schrodinger_bridge.pdf) | [🀄](papers_zh/2209.09893_deep_generalized_schrodinger_bridge.zh.pdf) |
| 2209.12093 | Learn what matters: cross-domain imitation learning with task-relevant embeddings | NeurIPS 2022 | [📘](reports/2209.12093_task_relevant_embeddings_cross_domain_il.md) | [📄](papers/2209.12093_task_relevant_embeddings_cross_domain_il.pdf) | [🀄](papers_zh/2209.12093_task_relevant_embeddings_cross_domain_il.zh.pdf) |
| 2206.15469 | Watch and Match: Supercharging Imitation with Regularized Optimal Transport | CoRL 2022 (PMLR 205:32-43, 2023) | [📘](reports/2206.15469_rot_watch_and_match.md) | [📄](papers/2206.15469_rot_watch_and_match.pdf) | [🀄](papers_zh/2206.15469_rot_watch_and_match.zh.pdf) |
| 2302.05872 | I²SB: Image-to-Image Schrödinger Bridge | ICML 2023 | [📘](reports/2302.05872_i2sb.md) | [📄](papers/2302.05872_i2sb.pdf) | [🀄](papers_zh/2302.05872_i2sb.zh.pdf) |
| 2308.12351 | Improving Generative Model-based Unfolding with Schrödinger Bridges | Phys. Rev. D 109, 076011 (2024) | [📘](reports/2308.12351_sb_unfold.md) | [📄](papers/2308.12351_sb_unfold.pdf) | [🀄](papers_zh/2308.12351_sb_unfold.zh.pdf) |
| 2310.02233 | Generalized Schrödinger Bridge Matching | ICLR 2024 (Poster) | [📘](reports/2310.02233_generalized_schrodinger_bridge_matching.md) | [📄](papers/2310.02233_generalized_schrodinger_bridge_matching.pdf) | [🀄](papers_zh/2310.02233_generalized_schrodinger_bridge_matching.zh.pdf) |
| 2311.01331 | Offline Imitation from Observation via Primal Wasserstein State Occupancy Matching | ICML 2024 | [📘](reports/2311.01331_primal_wasserstein_state_occupancy.md) | [📄](papers/2311.01331_primal_wasserstein_state_occupancy.pdf) | [🀄](papers_zh/2311.01331_primal_wasserstein_state_occupancy.zh.pdf) |
| 2409.09347 | Schrödinger Bridge Flow for Unpaired Data Translation | NeurIPS 2024 (Spotlight) | [📘](reports/2409.09347_schrodinger_bridge_flow_unpaired_translation.md) | [📄](papers/2409.09347_schrodinger_bridge_flow_unpaired_translation.pdf) | [🀄](papers_zh/2409.09347_schrodinger_bridge_flow_unpaired_translation.zh.pdf) |
| 2410.21795 | Robot Policy Learning with Temporal Optimal Transport Reward | NeurIPS 2024 | [📘](reports/2410.21795_temporal_ot_reward.md) | [📄](papers/2410.21795_temporal_ot_reward.pdf) | [🀄](papers_zh/2410.21795_temporal_ot_reward.zh.pdf) |
| 2404.13430 | React-OT: Optimal Transport for Generating Transition State in Chemical Reactions | Nature Machine Intelligence 7, 615-626 (2025) | [📘](reports/2404.13430_react_ot.md) | [📄](papers/2404.13430_react_ot.pdf) | [🀄](papers_zh/2404.13430_react_ot.zh.pdf) |
| 2409.06615 | One-Shot Imitation under Mismatched Execution | ICRA 2025 | [📘](reports/2409.06615_rhyme_one_shot_mismatched_execution.md) | [📄](papers/2409.06615_rhyme_one_shot_mismatched_execution.pdf) | [🀄](papers_zh/2409.06615_rhyme_one_shot_mismatched_execution.zh.pdf) |
| 2410.08751 | Zero-Shot Offline Imitation Learning via Optimal Transport | ICML 2025 | [📘](reports/2410.08751_zero_shot_offline_il_ot.md) | [📄](papers/2410.08751_zero_shot_offline_il_ot.pdf) | [🀄](papers_zh/2410.08751_zero_shot_offline_il_ot.zh.pdf) |
| 2504.11713 | Adjoint Sampling: Highly Scalable Diffusion Samplers via Adjoint Matching | ICML 2025 | [📘](reports/2504.11713_adjoint_sampling.md) | [📄](papers/2504.11713_adjoint_sampling.pdf) | [🀄](papers_zh/2504.11713_adjoint_sampling.zh.pdf) |
| 2506.10168 | Momentum Multi-Marginal Schrödinger Bridge Matching | NeurIPS 2025 | [📘](reports/2506.10168_momentum_multi_marginal_sbm.md) | [📄](papers/2506.10168_momentum_multi_marginal_sbm.pdf) | [🀄](papers_zh/2506.10168_momentum_multi_marginal_sbm.zh.pdf) |
| 2506.22565 | Adjoint Schrödinger Bridge Sampler | NeurIPS 2025 (Oral) | [📘](reports/2506.22565_adjoint_schrodinger_bridge_sampler.md) | [📄](papers/2506.22565_adjoint_schrodinger_bridge_sampler.pdf) | [🀄](papers_zh/2506.22565_adjoint_schrodinger_bridge_sampler.zh.pdf) |
| 2509.18631 | Generalizable Domain Adaptation for Sim-and-Real Policy Co-Training | NeurIPS 2025 | [📘](reports/2509.18631_guided_ot_sim_real_policy_cotraining.md) | [📄](papers/2509.18631_guided_ot_sim_real_policy_cotraining.pdf) | [🀄](papers_zh/2509.18631_guided_ot_sim_real_policy_cotraining.zh.pdf) |
| 2509.19626 | EgoBridge: Domain Adaptation for Generalizable Imitation from Egocentric Human Data | NeurIPS 2025 + CoRL 2025 (Oral) | [📘](reports/2509.19626_egobridge.md) | [📄](papers/2509.19626_egobridge.pdf) | [🀄](papers_zh/2509.19626_egobridge.zh.pdf) |
| 2511.06239 | Functional Adjoint Sampler: Scalable Sampling on Infinite Dimensional Spaces | ICML 2026 | [📘](reports/2511.06239_functional_adjoint_sampler.md) | [📄](papers/2511.06239_functional_adjoint_sampler.pdf) | [🀄](papers_zh/2511.06239_functional_adjoint_sampler.zh.pdf) |
| 2602.07132 | Discrete Adjoint Matching | ICLR 2026 | [📘](reports/2602.07132_discrete_adjoint_matching.md) | [📄](papers/2602.07132_discrete_adjoint_matching.pdf) | [🀄](papers_zh/2602.07132_discrete_adjoint_matching.zh.pdf) |
| 2602.08243 | Discrete Adjoint Schrödinger Bridge Sampler | ICML 2026 | [📘](reports/2602.08243_discrete_adjoint_schrodinger_bridge_sampler.md) | [📄](papers/2602.08243_discrete_adjoint_schrodinger_bridge_sampler.pdf) | [🀄](papers_zh/2602.08243_discrete_adjoint_schrodinger_bridge_sampler.zh.pdf) |
| 2602.23737 | Bridging Dynamics Gaps via Diffusion Schrödinger Bridge for Cross-Domain Reinforcement Learning | arXiv | [📘](reports/2602.23737_bdgxrl_diffusion_schrodinger_bridge.md) | [📄](papers/2602.23737_bdgxrl_diffusion_schrodinger_bridge.pdf) | [🀄](papers_zh/2602.23737_bdgxrl_diffusion_schrodinger_bridge.zh.pdf) |

**Topic notes (`topics/`, 20 份专题笔记)** — 方法谱系、基线协议与评测方案：

- [E01](topics/E01_dsbm_solver_lineage.md) E01 扩充报告：DSBM 精读 + SB 求解器谱系（IPF/DSB → IMF/DSBM → α-DSBM/SB Flow → bridge matching 系）
- [E02](topics/E02_ddbm_idbm.md) E02 扩充报告：DDBM 精读 + IDBM（bridge matching 理论源头）笔记
- [E03](topics/E03_asbm_lightsb_costs.md) E03 文献扩充：ASBM + LightSB 轻量 SB 求解器对照（NFE / 训练成本 / 适用维度）
- [E04](topics/E04_flow_matching_coupling.md) E04：Flow Matching 精读 + minibatch coupling 设计笔记（unpaired sim↔real）
- [E05](topics/E05_rf_stochastic_interpolants.md) E05 扩充报告：Rectified Flow 与 Stochastic Interpolants（附 FM ↔ RF ↔ SI ↔ SB 理论桥）
- [E06](topics/E06_adjoint_matching_origin.md) Adjoint Matching：Adjoint 谱系的源头精读（reward fine-tuning 的 memoryless SOC）
- [E07](topics/E07_diffusion_reward_alignment.md) E07 扩充报告：Diffusion Reward 对齐谱系 —— DDPO 精读 + 四路线综述
- [E08](topics/E08_dr_gan_baselines.md) E08：Domain Randomization + GAN 翻译经典基线 —— RCAN / RetinaGAN 精读与 SB-Render-Lite 基线协议规格
- [E09](topics/E09_splatsim_rialto_real2sim.md) E09：SplatSim + RialTo 精读 —— 重建/渲染 real2sim 竞品路线（上篇）
- [E10](topics/E10_lucidsim_xsim_interface.md) E10 精读：LucidSim + X-Sim —— 生成式增广与 real-to-sim-to-real 系统对 SB latent transport 的接口
- [E11](topics/E11_simplerenv_eval_protocol.md) E11：SimplerEnv 评测协议精读 + SB-Render-Lite 评测方案草案
- [E12](topics/E12_worldmodel_data_engine.md) E12｜世界模型数据引擎：DreamGen 精读 + UniSim 半精读 + 评估逻辑笔记
- [E13](topics/E13_diffusion_semantic_aug.md) E13 扩散语义增广基线：ROSIE 精读 + inpainting 增广对照协议
- [E14](topics/E14_soc_sampler_origins.md) E14｜SOC 采样器源头：PIS / DDS / CMCD 与 Adjoint 线的谱系定位
- [E15](topics/E15_energy_sampler_competitors.md) E15 · Simulation-free 能量采样竞品：iDEM / NETS / Sendera vs AS / ASBS
- [E16](topics/E16_latent_bridge_fewstep.md) E16 扩充报告：Latent Bridge 与少步部署
- [E17](topics/E17_zeroshot_translation_baselines.md) E17：Zero-shot 翻译基线（DDIB 精读 + SDEdit 收录）
- [E18](topics/E18_transport_policy_interface.md) E18 扩充报告：扩散/流策略下游接口 —— transport→policy 接口与 co-training 配比
- [E19](topics/E19_ot_theory_toolbox.md) E19 扩充笔记：OT 理论工具箱 —— UOT / GW 系变体 / Neural OT 求解器与评估陷阱
- [E20](topics/E20_sb_inverse_trajectory.md) E20：SB 逆问题 × Trajectory Inference 横断综述

<a name="7-trend-report-slides"></a>
## [7. Trend Report & Slides](#content)
*趋势报告与汇报*

- **Trend report (2025–2026)**: [survey/SB_TREND_REPORT_2026.md](survey/SB_TREND_REPORT_2026.md) · [PDF](survey/SB_TREND_REPORT_2026.pdf) — 五条主线的进展盘点、证据表与 insight。
- **Raw survey notes**: [survey/raw/](survey/raw/) — WebSearch 证据记录（`S1_*`）与 arXiv 近 12 个月扫描雷达表（`S2_*`，277 篇候选，含未入选项）；复扫命令 `python3 scripts/arxiv_scan.py --months 12`。
- **Slides**: [slides/awesome_sb_report.html](slides/awesome_sb_report.html)（HTML，←/→ 翻页，可打印）· [slides/awesome_sb_report.pdf](slides/awesome_sb_report.pdf) · Beamer 版 [slides/beamer/awesome_sb_beamer.pdf](slides/beamer/awesome_sb_beamer.pdf)

<a name="8-contributing-citation-license"></a>
## [8. Contributing, Citation & License](#content)
*贡献、引用与许可*

**Contributing** — PRs are welcome. Add a row to `metadata/extended.tsv` (or `resources.tsv`) and run `python3 scripts/build_readme.py`; please verify the venue on arXiv/OpenReview and link the official code when it exists. Chinese reports follow the template in `reports/` (基本信息 → 一句话总结 → 方法核心 → 实验与结果 → 局限性 → 关系).

**Citation**

```bibtex
@misc{awesome_schrodinger_bridge,
  title  = {Awesome Schr\"odinger Bridge: Papers, Code, Chinese Deep-dive Reports and Translations},
  author = {Li, Yufeng},
  year   = {2026},
  howpublished = {\url{https://github.com/asimfish/awesome_Schrodinger_Bridge}}
}
```

**License** — Curated text, reports and slides are released under [CC BY 4.0](LICENSE); scripts under MIT. Paper PDFs in `papers/` are the arXiv versions (see each paper's arXiv license); translated PDFs in `papers_zh/` are derivative works provided for non-commercial research use only — please cite the original papers.

**Acknowledgements** — Format inspired by [awesome-ml4co](https://github.com/Thinklab-SJTU/awesome-ml4co). Translations by [SuperTranslate](https://github.com/asimfish/super_translate); writing polished with [shuorenhua](https://github.com/MrGeDiao/shuorenhua) and [anti-defensive-writing](https://github.com/Kiterlin/anti-defensive-writing); reports organised in the spirit of [PaperOrchestra](https://github.com/Ar9av/PaperOrchestra); slides built on [ppt-master](https://github.com/hugohe3/ppt-master) design tokens and [beamer-skill](https://github.com/Noi1r/beamer-skill).
