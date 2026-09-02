# S1 · 2026-09-01 WebSearch 增量检索记录（主会话）

检索日：2026-09-01。方式：WebSearch（8 组关键词）+ 直接打开 arXiv / ICLR 2026 / ICML 2025–2026 / PMLR 页面核对 ID、标题、venue；GitHub 仓库经 `gh api` 核验存在。arXiv API 与 Semantic Scholar 当日限流，故以页面直读为证据。作者不确定的条目未写作者。

## 已入库（metadata/extended.tsv，source=survey2026）

| 条目 | 证据页 | venue | 代码 |
|---|---|---|---|
| Branched Schrödinger Bridge Matching | https://iclr.cc/virtual/2026/poster/10008461 · https://sophtang.github.io/branch-sbm/ | ICLR 2026 | github.com/sophtang/BranchSBM |
| Multi-Marginal Schrödinger Bridge Matching | https://arxiv.org/abs/2510.16587 | arXiv 2025 | github.com/bw-park/MSBM |
| Entering the Era of Discrete Diffusion Models: A Benchmark for SB and EOT | https://iclr.cc/virtual/2026/poster/10008954 | ICLR 2026 | github.com/gregkseno/catsbench |
| Minimal-Action Discrete SBM for Peptide Sequence Design | https://arxiv.org/abs/2601.22408 | arXiv 2026 | — |
| MDNS: Masked Diffusion Neural Sampler via SOC | https://arxiv.org/abs/2508.10684 | arXiv 2025 | — |
| Adjoint Matching through the Lens of the Stochastic Maximum Principle | https://arxiv.org/abs/2604.08580 | arXiv 2026 | — |
| Fine-Tuning Discrete Diffusion Models via Reward Optimization (DRAKES) | https://arxiv.org/abs/2410.13643 | arXiv 2024 | — |
| UniDB: A Unified Diffusion Bridge Framework via SOC | https://proceedings.mlr.press/v267/zhu25o.html · https://openreview.net/forum?id=uqCfoVXb67 | ICML 2025 | github.com/UniDB-SOC/UniDB |
| UniDB++ / A Unified and Fast-Sampling Diffusion Bridge Framework via SOC | https://arxiv.org/abs/2505.21528 | arXiv 2025 | github.com/2769433owo/UniDB-plusplus |
| Rectified Schrödinger Bridge Matching for Few-Step Visual Navigation | https://arxiv.org/abs/2604.05673 | arXiv 2026 | — |
| Sample from What You See (BridgePolicy) | https://arxiv.org/abs/2512.07212 · https://icml.cc/virtual/2026/poster/61138 | ICML 2026 | github.com/jianghcsr/BridgePolicy |
| A Unified Framework for Diffusion Bridge Problems | https://arxiv.org/abs/2503.21756 | arXiv 2025 | — |
| Foundations of Schrödinger Bridges for Generative Modeling | https://arxiv.org/abs/2603.18992 | arXiv 2026 | — |
| Notes on generative modeling: FM, diffusion, OT and SB（Peyresq 2026） | https://arxiv.org/abs/2606.30053 | arXiv 2026 | — |
| Non-equilibrium Annealed Adjoint Sampler (NAAS) | https://arxiv.org/abs/2506.18165 · https://openreview.net/forum?id=ay7WDSq0Kb | NeurIPS 2025 | — |
| Enhancing Diffusion-Based Sampling with Molecular Collective Variables (WT-ASBS) | https://arxiv.org/abs/2510.11923 · ghliu.github.io 新闻栏 | ICLR 2026 | github.com/facebookresearch/wt-asbs |
| Reflected Schrödinger Bridge Matching | https://arxiv.org/abs/2607.03626 | arXiv 2026 | — |
| Residual Diffusion Bridge Model for Image Restoration (RDBM) | https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Residual_Diffusion_Bridge_Model_for_Image_Restoration_CVPR_2026_paper.html · arXiv 2510.23116 | CVPR 2026 | github.com/MiliLab/RDBM |
| Bi-Bridge: Bidirectional Diffusion Bridges for Low-Light Image Enhancement | https://openaccess.thecvf.com/content/CVPR2026/html/Hua_Bi-Bridge_Bidirectional_Diffusion_Bridges_for_Low-Light_Image_Enhancement_CVPR_2026_paper.html | CVPR 2026 | — |
| Remote Sensing Image SR via Progressive Diffusion Schrödinger Bridge (PDSB) | https://www.mdpi.com/2072-4292/18/3/532 | Remote Sensing 2026 | — |
| Modeling Cell Dynamics and Interactions with Unbalanced Mean Field SB (CytoBridge) | https://arxiv.org/abs/2505.11197 · papers.nips.cc hash cbf552bd | NeurIPS 2025 | github.com/zhenyiizhang/CytoBridge-NeurIPS |
| （更正）MDNS venue → NeurIPS 2025 | https://ghliu.github.io/（新闻栏 09/2025 与 CV [C8]） | NeurIPS 2025 | — |

## 见到但未收录（未能打开原页 / 无 ID）

- FreeBridge、MMDSBM、Departures（AAAI 2026，已在 reports/deep_research_learning_resources.md 收录）
- XFlowMP（SB-based flow matching 运动规划）
- Off-policy training for discrete diffusion samplers（data-to-energy SB in discrete latent space）

## 第二轮关键词（补空白轴）

"Reflected Schrödinger Bridge Matching ICLR 2026" · "Schrödinger bridge image restoration 2026 CVPR 2026 bridge" · "CytoBridge mean-field unbalanced single-cell" · "Guan-Horng Liu 2026 SB LLM post-training adjoint" · "Non-equilibrium Annealed Adjoint Sampler" · "WT-ASBS"

## 第一轮关键词

"Schrödinger bridge matching 2026 arXiv new solver ICLR 2026" · "Schrödinger bridge 2025 arXiv discrete masked diffusion language model bridge" · "adjoint sampling 2026 Boltzmann diffusion sampler" · "Schrödinger bridge robot policy sim-to-real 2026 diffusion bridge imitation learning" · "UniDB unified diffusion bridges stochastic optimal control 2025" · "Schrödinger bridge single-cell trajectory inference 2026 unbalanced multi-marginal" · "stochastic optimal control fine-tuning discrete diffusion language model 2026" · "Schrödinger bridge survey 2025 tutorial diffusion bridge review"
