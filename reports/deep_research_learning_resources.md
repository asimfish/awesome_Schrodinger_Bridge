# Schrödinger Bridge × Optimal Transport × Sim2Real

## 深度调研、前沿论文与学习资源导航

> 检索与核验日期：2026-07-31  
> 覆盖重点：Schrödinger Bridge（SB）、Entropic Optimal Transport（EOT）、Diffusion / Flow Matching、Adjoint Sampling、跨域生成、机器人与具身 Sim2Real  
> 时间范围：以 2024–2025 两个完整会议年度为核心，同时纳入截至检索日已经正式公开的 2026 主会论文

---

## 0. 先说结论：这条研究线应该怎样学

如果目标是尽快从“会读论文”进入“能设计 SB-Sim2Real 实验”，最有效的知识链不是逐篇追最新论文，而是：

```text
静态 OT / Sinkhorn
    ↓
动态 OT / Benamou–Brenier
    ↓
SDE、Fokker–Planck、Girsanov、随机最优控制
    ↓
Schrödinger problem / Schrödinger Bridge
    ↓
Score-based diffusion / Flow Matching
    ↓
DSB、I²SB、SB Matching、GSBM
    ↓
Adjoint / Discrete / Functional / Multi-Marginal SB
    ↓
视觉域迁移、轨迹迁移、策略条件约束与 Sim2Real
```

对当前项目，最值得长期追踪的是四条线：

1. **带任务代价的路径输运**：GSBM、Feedback SBM、Topological SBM、Smooth SB。
2. **可扩展能量引导采样**：Adjoint Sampling、ASBS、Functional / Discrete Adjoint。
3. **非配对视觉与几何域迁移**：UNSB、DehazeSB、P2P-Bridge、视觉域泛化 SB。
4. **轨迹与具身迁移**：3MSBM、Grasp2Grasp、Offline RL stitching、sim-to-real / real-to-sim-to-real。

---

## 1. 标签、优先级与会议口径

### 1.1 标签

| 标签 | 含义 |
|---|---|
| `核心-SB` | 直接研究 Schrödinger Bridge、bridge matching 或其理论/算法 |
| `核心-Bridge` | bridge 方法本身是论文核心（如 diffusion / Brownian bridge 的模型、理论或系统比较），但不属于严格 entropic-OT / SB 一族 |
| `相邻-Bridge` | 借用 bridge 构造的相邻方法，非严格 entropic-OT / SB 一族，按应用价值选读 |
| `Adjoint` | 直接研究 adjoint matching / adjoint sampler |
| `相邻-OT/FM` | 不是 SB 主论文，但对 coupling、路径设计、flow matching 或控制有直接借鉴价值 |
| `直接-Sim2Real` | 研究真实/仿真域迁移、具身策略迁移或机器人部署 |
| `应用` | 图像、3D、分子、生物、物理反演等可迁移的 SB 应用 |
| `主会` | 已核验为正式 conference paper |
| `Workshop` | Workshop 论文，不能与主会等同引用 |

### 1.2 推荐优先级

- `S`：当前课题必须精读，建议推公式或复现实验。
- `A`：构成方法版图，至少精读方法与实验。
- `B`：按具体应用选读。
- `C`：扩展视野或作为检索入口。

### 1.3 “最近两年顶会”的口径

- **完整覆盖**：2024、2025。
- **增量覆盖**：截至 2026-07-31 已正式公开的 2026 论文。
- **NeurIPS 2026** 尚未举行，因此不存在可核验的 2026 正式 proceedings。
- **ECCV 2026** 尚未举行；最近一届完整 proceedings 是 ECCV 2024。
- **ICCV** 为双年会，最近一届是 ICCV 2025。
- 对只有少量直接 SB 论文的会议，另列 `相邻-OT/FM` 或 `直接-Sim2Real`，不为了“凑会议”把弱相关论文冒充 SB 论文。

---

## 2. 十五篇核心阅读链

这组论文可以作为正式读最新文献前的主干。

| 顺序 | 论文 | 读它解决什么问题 | 优先级 |
|---:|---|---|:---:|
| 1 | [Computational Optimal Transport](https://optimaltransport.github.io/book/)（选读 Ch. 2/4/5/9，与 §4.1 口径一致，不必通读全书） | Wasserstein、Kantorovich、Sinkhorn、EOT 的计算基础 | S |
| 2 | [A Survey of the Schrödinger Problem and Some of Its Connections with Optimal Transport](https://arxiv.org/abs/1308.0215) | SB 的概率、熵最小化与 OT 全景 | S |
| 3 | [On the Relation Between Optimal Transport and Schrödinger Bridges: A Stochastic Control Viewpoint](https://arxiv.org/abs/1412.4430) | 动态 OT、随机控制与 SB 的统一 | S |
| 4 | [Score-Based Generative Modeling through SDEs](https://openreview.net/forum?id=PxTIG12RRHS) | 反向 SDE、score、probability-flow ODE | S |
| 5 | [Flow Matching for Generative Modeling](https://openreview.net/forum?id=PqvMRDCJT9t) | 从回归向量场理解生成路径 | S |
| 6 | [Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling](https://proceedings.neurips.cc/paper/2021/hash/940392f5f32a7ade1cc201767cf83e31-Abstract.html) | 现代神经 SB 的经典算法入口 | S |
| 7 | [I²SB: Image-to-Image Schrödinger Bridge](https://proceedings.mlr.press/v202/liu23ai.html) | paired image restoration / translation 的标准桥模型 | S |
| 8 | [Diffusion Schrödinger Bridge Matching](https://proceedings.neurips.cc/paper_files/paper/2023/hash/c428adf74782c2092d254329b6b02482-Abstract-Conference.html) | bridge matching 与 Iterative Markovian Fitting | S |
| 9 | [Simulation-Free Schrödinger Bridges via Score and Flow Matching](https://proceedings.mlr.press/v238/tong24a.html) | 用 EOT coupling 避免训练中反复模拟 SDE | A |
| 10 | [Generalized Schrödinger Bridge Matching](https://proceedings.iclr.cc/paper_files/paper/2024/hash/3e91dd686ea22cb51b8732e3c7e9fc8e-Abstract-Conference.html) | 把几何、状态、控制代价写进路径目标 | S |
| 11 | [Light and Optimal Schrödinger Bridge Matching](https://proceedings.mlr.press/v235/gushchin24a.html) | 单次 matching、最优参数化与轻量求解 | A |
| 12 | [Adjoint Sampling](https://openreview.net/forum?id=6Eg1OrHmg2) | 不依赖目标样本、只用未归一化能量的可扩展采样 | S |
| 13 | [Adjoint Schrödinger Bridge Sampler](https://proceedings.neurips.cc/paper_files/paper/2025/hash/174692c52dc84fad2b2e99dd8637ce6a-Abstract-Conference.html) | 将 adjoint sampler 扩展到任意 source prior | S |
| 14 | [Momentum Multi-Marginal Schrödinger Bridge Matching](https://proceedings.neurips.cc/paper_files/paper/2025/hash/7c3875b86bd2b0639ab1e858c678af40-Abstract-Conference.html) | 多时间点、动量与轨迹平滑 | S |
| 15 | [Functional Adjoint Sampler](https://openreview.net/forum?id=XcJk9un0Tb) | 把 energy-only adjoint 采样从有限维向量扩展到函数空间 / 无限维对象（属 Adjoint 谱系，非时间多边缘 SB） | A |

本地已有的逐篇报告可从 [报告索引](./INDEX.md) 进入；本节负责搭骨架，不替代逐篇深读。

### 2.1 本轮指定论文的本地专题入口

| 方向 | 本地报告 |
|---|---|
| 连续空间 Adjoint | [Adjoint Sampling](./2504.11713_adjoint_sampling.md) · [Adjoint Schrödinger Bridge Sampler](./2506.22565_adjoint_schrodinger_bridge_sampler.md) |
| 离散 / 函数空间 Adjoint | [Discrete Adjoint Matching](./2602.07132_discrete_adjoint_matching.md) · [Discrete Adjoint Schrödinger Bridge Sampler](./2602.08243_discrete_adjoint_schrodinger_bridge_sampler.md) · [Functional Adjoint Sampler](./2511.06239_functional_adjoint_sampler.md) |
| Generalized / Multi-Marginal SB | [Deep Generalized Schrödinger Bridge](./2209.09893_deep_generalized_schrodinger_bridge.md) · [Generalized Schrödinger Bridge Matching](./2310.02233_generalized_schrodinger_bridge_matching.md) · [Momentum Multi-Marginal Schrödinger Bridge Matching](./2506.10168_momentum_multi_marginal_sbm.md) |
| 图像 / 科学 / 化学应用 | [I²SB](./2302.05872_i2sb.md) · [SBUnfold](./2308.12351_sb_unfold.md) · [React-OT](./2404.13430_react_ot.md) |
| 跨论文总结 | [Adjoint / Generalized / Structured SB 扩展综述](./sb_adjoint_extended_synthesis.md) |
| 研究者专题 | [Guan-Horng Liu：从最优控制到 SB、Adjoint Sampling 与 LLM Post-training](./guan_horng_liu_research_roadmap.md) |

---

## 3. 2024–2026 顶会前沿雷达

## 3.1 NeurIPS

### 2025：直接相关

| 论文 | 标签 | 为什么值得读 | 优先级 |
|---|---|---|:---:|
| [Adjoint Schrödinger Bridge Sampler](https://proceedings.neurips.cc/paper_files/paper/2025/hash/174692c52dc84fad2b2e99dd8637ce6a-Abstract-Conference.html) | `核心-SB` `Adjoint` `主会` | 用任意可采样 source prior 连接能量目标，是 energy-guided sim2real 的关键方法 | S |
| [Momentum Multi-Marginal Schrödinger Bridge Matching](https://proceedings.neurips.cc/paper_files/paper/2025/hash/7c3875b86bd2b0639ab1e858c678af40-Abstract-Conference.html) | `核心-SB` `主会` | 同时满足多个时间边际，并用动量状态改善轨迹连续性 | S |
| [Grasp2Grasp: Vision-Based Dexterous Grasp Translation via Schrödinger Bridges](https://proceedings.neurips.cc/paper_files/paper/2025/hash/18b1a1ea7278c9429e96a11f960e30f2-Abstract-Conference.html) | `核心-SB` `直接-Sim2Real` `主会` | 在异构灵巧手之间翻译视觉抓取，和 embodiment gap 高度一致 | S |
| [Degradation-Aware Dynamic Schrödinger Bridge for Unpaired Image Restoration](https://proceedings.neurips.cc/paper_files/paper/2025/hash/039c30e9af8039fbd1b58da9d04f38e9-Abstract-Conference.html) | `核心-SB` `应用` `主会` | 非配对恢复中显式建模退化路径，可借鉴到 sim-render degradation | A |
| [Optical Coherence Tomography Harmonization with Anatomy-Guided Latent Metric Schrödinger Bridges](https://proceedings.neurips.cc/paper_files/paper/2025/hash/08b60b4af0b8163b18553b15f5ce25d2-Abstract-Conference.html) | `核心-SB` `应用` `主会` | 通过任务结构定义 latent metric，示范“保持语义/几何再做域迁移” | A |
| [Learning a Cross-Modal Schrödinger Bridge for Visual Domain Generalization](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f4fba41b554f9aaa013c4062a1c40518-Abstract-Conference.html) | `核心-SB` `应用` `主会` | 用跨模态信息构造域泛化桥，对视觉 sim2real 很直接 | S |
| [Fractional Diffusion Bridge Models](https://proceedings.neurips.cc/paper_files/paper/2025/hash/a66caa1703fe34705a4368c3014c1966-Abstract-Conference.html) | `核心-SB` `主会` | 非 Markov、长记忆参考过程，适合带时序相关的数据与轨迹 | A |
| [Modeling Cell Dynamics and Interactions with Unbalanced Mean Field Schrödinger Bridge](https://proceedings.neurips.cc/paper_files/paper/2025/hash/cbf552bd72c0dd301605d3f620fe0c3a-Abstract-Conference.html) | `核心-SB` `应用` `主会` | 同时处理质量变化与群体交互，是 unbalanced / mean-field 扩展范例 | B |

### 2024：直接相关

| 论文 | 标签 | 为什么值得读 | 优先级 |
|---|---|---|:---:|
| [Schrödinger Bridge Flow for Unpaired Data Translation](https://arxiv.org/abs/2409.09347) | `核心-SB` `主会` | 把 SB 求解写成单一 flow（α-IMF / 在线版 α-DSBM），免去 DSBM 的样本缓存与交替训练，是本项目 unpaired 基线主力（§8/§9.2 首选之一；NeurIPS 2024 Spotlight）；本地已有[逐篇报告](./2409.09347_schrodinger_bridge_flow_unpaired_translation.md) | S |

### 2024–2025：重要相邻方法

| 论文 | 标签 | 价值 |
|---|---|---|
| [Optimal Flow Matching](https://proceedings.neurips.cc/paper_files/paper/2024/file/bc8f76d9caadd48f77025b1c889d2e2d-Paper-Conference.pdf) | `相邻-OT/FM` `主会` | 从路径/coupling 的最优性理解 flow matching |
| [Curly Flow Matching for Learning Non-gradient Field Dynamics](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b00692865928625ace212e89ded2decd-Abstract-Conference.html) | `相邻-OT/FM` `主会` | 允许非梯度、旋转或周期动力学，适合机器人相空间与循环任务 |
| [Value Gradient Guidance for Flow Matching Alignment](https://proceedings.neurips.cc/paper_files/paper/2025/hash/10b7e27c8eb9571fbbd2ae6a9f8c3855-Abstract-Conference.html) | `相邻-OT/FM` `主会` | 用 value gradient 引导生成流，可类比任务奖励/策略价值引导 |

## 3.2 ICML

| 年份 | 论文 | 标签 | 为什么值得读 | 优先级 |
|---:|---|---|---|:---:|
| 2024 | [Light and Optimal Schrödinger Bridge Matching](https://proceedings.mlr.press/v235/gushchin24a.html) | `核心-SB` `主会` | 讨论从任意 plan 恢复 SB、单次 matching 与轻量参数化 | A |
| 2025 | [Adjoint Sampling: Highly Scalable Diffusion Samplers via Adjoint Matching](https://openreview.net/forum?id=6Eg1OrHmg2) | `Adjoint` `主会` | 以能量函数代替目标样本，核心是高扩展性 adjoint matching | S |
| 2025 | [Categorical Schrödinger Bridge Matching](https://proceedings.mlr.press/v267/ksenofontov25a.html) | `核心-SB` `主会` | 离散时间 IMF 的收敛与 VQ / token / 分子类别空间 | S |
| 2025 | [Trajectory Inference with Smooth Schrödinger Bridges](https://proceedings.mlr.press/v267/hong25f.html) | `核心-SB` `主会` | 使用 Matérn 等平滑 GP reference，并提升到 phase space 求解 | S |
| 2026 | [Functional Adjoint Sampler: Scalable Sampling on Infinite Dimensional Spaces](https://openreview.net/forum?id=XcJk9un0Tb) | `Adjoint` `主会` | 对函数、场、轨迹等无限维对象做 amortized energy sampling | A |
| 2026 | [Discrete Adjoint Schrödinger Bridge Sampler](https://arxiv.org/abs/2602.08243) | `核心-SB` `Adjoint` `主会` | 从任意离散 source 到 unnormalized target；最新版由作者主页与 CV 列为 ICML 2026 | A |

补充但不属于 ICML：[[SF]²M（AISTATS 2024）](https://proceedings.mlr.press/v238/tong24a.html) 是连接 flow matching 与 SB 最重要的 simulation-free 方法之一。

## 3.3 ICLR

| 年份 | 论文 | 标签 | 为什么值得读 | 优先级 |
|---:|---|---|---|:---:|
| 2024 | [Generalized Schrödinger Bridge Matching](https://proceedings.iclr.cc/paper_files/paper/2024/hash/3e91dd686ea22cb51b8732e3c7e9fc8e-Abstract-Conference.html) | `核心-SB` `主会` | 用 conditional stochastic optimal control 处理非线性路径代价 | S |
| 2024 | [Unpaired Image-to-Image Translation via Neural Schrödinger Bridge](https://proceedings.iclr.cc/paper_files/paper/2024/hash/5491280797f3192b895bce84eb83df8d-Abstract-Conference.html) | `核心-SB` `应用` `主会` | 高分辨率非配对 I2I 的实用基线 | S |
| 2025 | [Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models with Memoryless Stochastic Optimal Control](https://arxiv.org/abs/2409.08861) | `Adjoint` `主会` | AS/ASBS/DAM/FAS 的共同方法论源头（ICLR 2025 Spotlight）：memoryless SOC 形式化 reward 微调 + lean adjoint 回归 | S |
| 2025 | [Feedback Schrödinger Bridge Matching](https://proceedings.iclr.cc/paper_files/paper/2025/hash/94398c15080cca93180416a52989949a-Abstract-Conference.html) | `核心-SB` `主会` | 用少于 8% 的预对齐样本指导其余非配对输运（ICLR 2025 Oral） | S |
| 2025 | [Topological Schrödinger Bridge Matching](https://proceedings.iclr.cc/paper_files/paper/2025/hash/7de665476d0adc8a54d3b8744f932bbf-Abstract-Conference.html) | `核心-SB` `主会` | 在图与单纯复形上使用 topology-aware reference dynamics（ICLR 2025 Spotlight） | A |
| 2025 | [Discrete Diffusion Schrödinger Bridge Matching for Graph Transformation](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2438d634f0ed1640934d31376c110a92-Abstract-Conference.html) | `核心-SB` `主会` | CTMC、离散 IMF 与 graph edit cost 的统一 | A |
| 2025 | [Underdamped Diffusion Bridges with Applications to Sampling](https://proceedings.iclr.cc/paper_files/paper/2025/hash/08342dc6ab69f23167b4123086ad4d38-Abstract-Conference.html) | `核心-SB` `主会` | 相空间、退化扩散与更少离散步数，适合动力学系统 | S |
| 2025 | [Diffusion Bridge Implicit Models](https://proceedings.iclr.cc/paper_files/paper/2025/hash/cb8a878dc5afad501474efa554926771-Abstract-Conference.html) | `核心-SB` `应用` `主会` | 无额外训练的快速 bridge sampler，报告最高 25× 加速 | A |
| 2025 | [Physics-aligned Field Reconstruction with Diffusion Bridge](https://proceedings.iclr.cc/paper_files/paper/2025/hash/9421261e06f1a63a352b068f1ac90609-Abstract-Conference.html) | `核心-SB` `应用` `主会` | 将物理结构纳入场重建，适合作为 functional / field SB 案例 | A |
| 2026 | [Discrete Adjoint Matching](https://openreview.net/forum?id=VXB4xxAgOf) | `Adjoint` `主会` | 将 adjoint matching 扩展到离散状态空间 | S |
| 2026 | [Count Bridges Enable Modeling and Deconvolving Transcriptomic Data](https://openreview.net/forum?id=4nOZBufbLC) | `核心-SB` `应用` `主会` | 对整数计数数据建立 bridge，展示非欧连续空间之外的建模方式 | B |
| 2026 | [Discrete Adjoint Schrödinger Bridge Sampler（早期版本）](https://openreview.net/forum?id=6gVec6LNtu) | `核心-SB` `Adjoint` `Workshop` | 早期 DeLTa Workshop 版本；完整论文后由作者列为 ICML 2026 主会，正式引用应采用最新版 | C |

## 3.4 ECCV

ECCV 2024 没有形成像 ICLR/NeurIPS 那样密集的 SB 方法簇，但视觉 bridge 应用很有借鉴价值。

| 论文 | 标签 | 为什么值得读 | 优先级 |
|---|---|---|:---:|
| [P2P-Bridge: Diffusion Bridges for 3D Point Cloud Denoising](https://eccv.ecva.net/virtual/2024/poster/207) | `应用` `核心-Bridge` `主会` | 把 bridge 用于 3D 点云恢复，关注几何保持而非只看像素 | A |
| [EBDM: Exemplar-guided Image Translation with Brownian-bridge Diffusion Models](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/2096_ECCV_2024_paper.php) | `应用` `相邻-Bridge` `主会` | exemplar-conditioned Brownian bridge；不是严格 EOT-SB，但对条件域翻译实用 | B |
| [Towards Robust Event-based Networks for Nighttime via Unpaired Day-to-Night Event Translation](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08432.pdf) | `应用` `直接-Sim2Real` `主会` | 用 neural SB 做非配对昼夜 event translation，体现传感器域迁移 | A |

## 3.5 ICCV

| 论文 | 标签 | 为什么值得读 | 优先级 |
|---|---|---|:---:|
| [When Schrödinger Bridge Meets Real-World Image Dehazing with Unpaired Training](https://openaccess.thecvf.com/content/ICCV2025/html/Lan_When_Schrodinger_Bridge_Meets_Real-World_Image_Dehazing_with_Unpaired_Training_ICCV_2025_paper.html) | `核心-SB` `应用` `主会` | 真实退化、非配对训练与视觉域恢复的直接案例 | S |
| [CounterPC: Counterfactual Feature Realignment for Unsupervised Domain Adaptation on Point Clouds](https://openaccess.thecvf.com/content/ICCV2025/html/Yang_CounterPC_Counterfactual_Feature_Realignment_for_Unsupervised_Domain_Adaptation_on_Point_ICCV_2025_paper.html) | `相邻-OT/FM` `直接-Sim2Real` `主会` | 点云 UDA 与特征重对齐，适合几何 sim2real 对照 | A |
| [Contrastive Flow Matching](https://openaccess.thecvf.com/content/ICCV2025/html/Stoica_Contrastive_Flow_Matching_ICCV_2025_paper.html) | `相邻-OT/FM` `主会` | 研究 pairing / representation 对 flow 路径的影响 | B |
| [EmbodiedSplat: Personalized Real-to-Sim-to-Real Navigation with Gaussian Splats](https://openaccess.thecvf.com/content/ICCV2025/html/Chhablani_EmbodiedSplat_Personalized_Real-to-Sim-to-Real_Navigation_with_Gaussian_Splats_from_a_Mobile_ICCV_2025_paper.html) | `直接-Sim2Real` `主会` | 从真实扫描构建个性化模拟环境再迁回现实，是系统路线的重要对照 | S |
| [Taming Flow Matching with Unbalanced Optimal Transport into Fast Pansharpening](https://www.openaccess.thecvf.com/content/ICCV2025/papers/Cao_Taming_Flow_Matching_with_Unbalanced_Optimal_Transport_into_Fast_Pansharpening_ICCV_2025_paper.pdf) | `相邻-OT/FM` `应用` `主会` | unbalanced OT 与 FM 结合，适合质量不守恒或支持集不一致问题 | A |

## 3.6 CVPR

| 年份 | 论文 | 标签 | 为什么值得读 | 优先级 |
|---:|---|---|---|:---:|
| 2025 | [Finding Local Diffusion Schrödinger Bridge using Kolmogorov-Arnold Network](https://openaccess.thecvf.com/content/CVPR2025/html/Qiu_Finding_Local_Diffusion_Schrodinger_Bridge_using_Kolmogorov-Arnold_Network_CVPR_2025_paper.html) | `核心-SB` `主会` | 通过局部 bridge 与 KAN 处理复杂映射 | B |
| 2025 | [A Unified Latent Schrödinger Bridge Diffusion Model for Unsupervised Anomaly Detection and Localization](https://openaccess.thecvf.com/content/CVPR2025/html/Akshay_A_Unified_Latent_Schrodinger_Bridge_Diffusion_Model_for_Unsupervised_Anomaly_CVPR_2025_paper.html) | `核心-SB` `应用` `主会` | latent bridge 同时做异常检测与定位 | B |
| 2025 | [Enhancing Virtual Try-On with Synthetic Pairs and Error-Aware Noise Scheduling](https://openaccess.thecvf.com/content/CVPR2025/html/Li_Enhancing_Virtual_Try-On_with_Synthetic_Pairs_and_Error-Aware_Noise_Scheduling_CVPR_2025_paper.html) | `应用` `主会` | 合成配对与 error-aware SB noise schedule，可类比 sim-real mixed pairs | A |
| 2025 | [Vid2Sim: Realistic and Interactive Simulation from Video for Urban Navigation](https://openaccess.thecvf.com/content/CVPR2025/html/Xie_Vid2Sim_Realistic_and_Interactive_Simulation_from_Video_for_Urban_Navigation_CVPR_2025_paper.html) | `直接-Sim2Real` `主会` | 从视频构建可交互模拟器，是视觉 real-to-sim 路线强对照 | S |
| 2025 | [Optimal Transport-Guided Source-Free Adaptation for Face Anti-Spoofing](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_Optimal_Transport-Guided_Source-Free_Adaptation_for_Face_Anti-Spoofing_CVPR_2025_paper.pdf) | `相邻-OT/FM` `主会` | source-free adaptation 的 OT 设计可借鉴到无法保留仿真源数据的场景 | B |
| 2026 | [Inconsistency-aware Multimodal Schrödinger Bridge for Deepfake Localization](https://openaccess.thecvf.com/content/CVPR2026/html/Xiong_Inconsistency-aware_Multimodal_Schrodinger_Bridge_for_Deepfake_Localization_CVPR_2026_paper.html) | `核心-SB` `应用` `主会` | 多模态不一致性作为 bridge 信号，适合研究多传感器对齐 | A |
| 2026 | [Opening the Sim-to-Real Door for Humanoid Pixel-to-Action Policy Transfer](https://openaccess.thecvf.com/content/CVPR2026/html/Xue_Opening_the_Sim-to-Real_Door_for_Humanoid_Pixel-to-Action_Policy_Transfer_CVPR_2026_paper.html) | `直接-Sim2Real` `主会` | 像素到动作的人形机器人策略迁移，直接对应最终下游评估 | S |
| 2026 | [Vision-Language Model Guided Source-Free Domain Adaptation via Optimal Transport](https://openaccess.thecvf.com/content/CVPR2026/html/Han_Vision-Language_Model_Guided_Source-Free_Domain_Adaptation_via_Optimal_Transport_CVPR_2026_paper.html) | `相邻-OT/FM` `主会` | 用 VLM 语义指导 OT，可用于建立 task-relevant transport cost | A |

## 3.7 AAAI

| 年份 | 论文 | 标签 | 为什么值得读 | 优先级 |
|---:|---|---|---|:---:|
| 2026 | [Departures: Distributional Transport for Single-Cell Perturbation Prediction with Neural Schrödinger Bridges](https://ojs.aaai.org/index.php/AAAI/article/view/39190) | `核心-SB` `应用` `主会` | 将 intervention / perturbation 看作分布输运 | B |
| 2026 | [Bridging Day and Night: Target-Class Hallucination Suppression in Unpaired Image Translation](https://ojs.aaai.org/index.php/AAAI/article/view/37570) | `核心-SB` `应用` `主会` | 针对昼夜非配对翻译中的目标类幻觉，和语义保持高度相关 | A |
| 2026 | [BridgeShape: Latent Diffusion Schrödinger Bridge for 3D Shape Completion](https://ojs.aaai.org/index.php/AAAI/article/view/37493) | `核心-SB` `应用` `主会` | 3D 几何补全与 latent SB | A |
| 2026 | [Rethinking Flow and Diffusion Bridge Models for Speech Enhancement](https://ojs.aaai.org/index.php/AAAI/article/view/40630) | `核心-Bridge` `相邻-OT/FM` `主会` | 系统比较 flow / diffusion bridge 的条件生成取舍 | B |
| 2026 | [Treatment Stitching with Schrödinger Bridge for Enhancing Offline Reinforcement Learning in Adaptive Treatment Strategies](https://ojs.aaai.org/index.php/AAAI/article/view/38826) | `核心-SB` `主会` | 用 SB 补全难以直接拼接的轨迹段，对 offline robot trajectory augmentation 很有启发 | S |

### 会议覆盖检查

| 会议 | 2024 | 2025 | 2026 截止检索日 | 直接 SB / Bridge | 相邻 OT/FM / Sim2Real |
|---|:---:|:---:|:---:|:---:|:---:|
| NeurIPS | ✓ | ✓ | 尚未举行 | ✓ | ✓ |
| ICML | ✓ | ✓ | ✓ | ✓ | — |
| ICLR | ✓ | ✓ | ✓ | ✓ | — |
| ECCV | ✓ | — | 尚未举行 | ✓ | ✓ |
| ICCV | — | ✓ | — | ✓ | ✓ |
| CVPR | — | ✓ | ✓ | ✓ | ✓ |
| AAAI | — | — | ✓ | ✓ | ✓ |

---

## 4. 基础理论资料

## 4.1 Optimal Transport 与 Entropic OT

| 资源 | 类型 / 难度 | 建议读法 |
|---|---|---|
| [Computational Optimal Transport — Peyré & Cuturi](https://optimaltransport.github.io/book/) | 免费教材；★★ | 先读 Ch. 2、4、5、9；重点是 Kantorovich、Sinkhorn、动态 OT 与 ML 应用 |
| [Optimal Transport for Applied Mathematicians — Santambrogio](https://www.math.univ-toulouse.fr/~santambr/OTAM-cvgmt.pdf) | 教材；★★★ | 补充严格理论、Wasserstein 空间与变分结构 |
| [Optimal Transport: Old and New — Villani](https://link.springer.com/book/10.1007/978-3-540-71050-9) | 经典专著；★★★ | 用作定理字典，不建议第一遍线性通读 |
| [Optimal Transport on Discrete Domains — Solomon](https://people.csail.mit.edu/jsolomon/assets/optimal_transport.pdf) | 课程讲义；★★ | 适合图、点云、离散几何与数值实现 |
| [Sinkhorn Distances: Lightspeed Computation of Optimal Transport](https://proceedings.neurips.cc/paper/2013/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html) | 基础论文；★★ | 理解为什么熵正则使 OT 可规模化 |
| [A Computational Fluid Mechanics Solution to the Monge–Kantorovich Mass Transfer Problem](https://link.springer.com/article/10.1007/s002110050002) | 经典论文；★★★ | 动态 OT / Benamou–Brenier 公式的原始来源 |
| [Topics in Optimal Transportation — Villani](https://bookstore.ams.org/gsm-58)（AMS GSM 58, 2003） | 入门专著；★★ | 比《Old and New》易读一个量级，适合线性通读的第一本 OT 专著 |
| An Invitation to Optimal Transport, Wasserstein Distances, and Gradient Flows — Figalli & Glaudo（EMS, 2021；EMS Press 检索书名） | 现代教材；★★★ | 精简（约 150 页）而严格，恰好补 Santambrogio 与 Villani 之间的层次 |

掌握标准：

- 能写出 Kantorovich primal / dual。
- 能解释 entropic regularization 的统计和计算含义。
- 能从 continuity equation 写出动态 OT。
- 能区分 transport plan、transport map、coupling、path measure。

## 4.2 SDE、随机过程与最优控制

| 资源 | 类型 / 难度 | 建议读法 |
|---|---|---|
| [Applied Stochastic Differential Equations — Särkkä & Solin](https://users.aalto.fi/~ssarkka/pub/sde_book.pdf) | 免费教材；★★ | 首选入门；重点 Ch. 3–8，尤其 Fokker–Planck、Girsanov、Doob h-transform |
| [Stochastic Differential Equations — Øksendal](https://link.springer.com/book/10.1007/978-3-642-14394-6) | 教材；★★★ | 补严谨 Itô calculus、Girsanov、Feynman–Kac |
| [Stochastic Optimal Control: The Discrete-Time Case — Bertsekas & Shreve](https://www.athenasc.com/dpbook.html) | 控制教材；★★★ | 用于理解 Bellman、policy、value 与随机控制 |
| [Path Integral Control and KL Control](https://homes.cs.washington.edu/~todorov/papers/TodorovCDC06.pdf) | 论文；★★★ | 连接控制代价、KL 与可线性化控制 |
| [Path integrals and symmetry breaking for optimal control theory — Kappen](https://arxiv.org/abs/physics/0505066) | 教程论文；★★ | 为 GSBM / adjoint 系列补路径积分控制直觉 |
| [A Generalized Path Integral Control Approach to Reinforcement Learning — Theodorou et al.](https://jmlr.org/papers/v11/theodorou10a.html) | 论文（JMLR 2010）；★★ | 路径积分控制通向 RL（PI²）的桥 |
| [Controlled Markov Processes and Viscosity Solutions — Fleming & Soner](https://link.springer.com/book/10.1007/0-387-31071-1) | 控制教材；★★★ | 连续时间 SOC 标准教材（HJB、粘性解）；Bertsekas–Shreve 只覆盖离散时间，而 GSBM / adjoint 系列全在连续时间框架 |
| [Stochastic Controls: Hamiltonian Systems and HJB Equations — Yong & Zhou](https://link.springer.com/book/10.1007/978-1-4612-1466-3) | 控制教材；★★★ | Pontryagin 极大值原理 / adjoint 方程与 HJB 的对偶，正是 Adjoint Matching 一族的数学出处 |

掌握标准：

- 会从 SDE 写 generator 与 Fokker–Planck equation。
- 能解释 forward / reverse-time SDE。
- 理解 Girsanov 如何把漂移控制转成路径 KL。
- 知道 HJB、value function、Pontryagin / adjoint 的角色。

## 4.3 Schrödinger Bridge 的理论入口

| 资源 | 定位 |
|---|---|
| [Léonard 综述](https://arxiv.org/abs/1308.0215) | 最完整的经典 SB 文献入口；重点是 path-space KL、Schrödinger system 与 OT 联系 |
| [OT 与 SB 的随机控制视角](https://arxiv.org/abs/1412.4430) | 从控制与流体动力学统一动态 OT / SB |
| [From the Schrödinger Problem to the Monge–Kantorovich Problem](https://arxiv.org/abs/1011.2564) | 理解小噪声极限与 Γ-convergence |
| [Diffusion Schrödinger Bridge](https://proceedings.neurips.cc/paper/2021/hash/940392f5f32a7ade1cc201767cf83e31-Abstract.html) | 现代机器学习 SB 的算法起点 |
| [Building the Bridge of Schrödinger](https://openreview.net/forum?id=OHimIaixXk) | 连续 EOT benchmark，适合比较求解器而非只看生成质量 |
| [Soft-constrained Schrödinger Bridge](https://proceedings.mlr.press/v238/garg24a.html) | 当边际约束不完全可信或需要软约束时的扩展 |
| [Reflected Schrödinger Bridge for Constrained Generative Modeling](https://proceedings.mlr.press/v244/deng24b.html) | 有硬状态约束/安全集合时的重要方向 |
| [Stochastic Control Liaisons: Richard Sinkhorn Meets Gaspard Monge on a Schrödinger Bridge — Chen, Georgiou & Pavon](https://arxiv.org/abs/2005.10963)（SIAM Review, 2021） | 现代 SB–SOC–Sinkhorn 联系的标准综述，比 Léonard 更贴近 ML 读者，是本方向被引最多的综述之一 |
| Optimal Transport in Systems and Control — Chen, Georgiou & Pavon（Annu. Rev. Control, Robotics, and Autonomous Systems, 2021；Annual Reviews 检索标题） | 面向控制的 OT/SB 综述，与机器人下游最贴 |

历史源头（C 级选读）：Schrödinger 1931/1932 原文与 Föllmer《Random fields and diffusion processes》（Saint-Flour 1988 讲义），检索标题即可。

## 4.4 Diffusion、Score 与 Flow Matching

| 资源 | 定位 |
|---|---|
| [Denoising Diffusion Probabilistic Models](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html) | DDPM 标准入口 |
| [Score-Based Generative Modeling through SDEs](https://openreview.net/forum?id=PxTIG12RRHS) | 将 score、reverse SDE、ODE 统一起来 |
| [Flow Matching for Generative Modeling](https://openreview.net/forum?id=PqvMRDCJT9t) | 连续归一化流与条件向量场回归 |
| [Building Normalizing Flows with Stochastic Interpolants](https://openreview.net/forum?id=li7qeBbCR1t) | ICLR 2023 短版：用插值直接构造 flow 的入口 |
| [Stochastic Interpolants: A Unifying Framework for Flows and Diffusions](https://arxiv.org/abs/2303.08797) | 完整框架版（已正式发表于 JMLR 26(209), 2025）：用统一插值视角联系 flow、score 与 stochastic dynamics |
| [Simulation-Free Schrödinger Bridges via Score and Flow Matching](https://proceedings.mlr.press/v238/tong24a.html) | SB、score、flow matching 的直接交汇点 |
| [An Introduction to Flow Matching and Diffusion Models](https://arxiv.org/abs/2506.02070) | 2025/2026 最友好的系统讲义之一 |
| [Flow Matching Guide and Code](https://arxiv.org/abs/2412.06264) · [facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching) | FM 原作者团队的官方指南（约 90 页）+ 配套代码库，是 2506.02070 讲义之外最权威的系统材料 |
| [Understanding Diffusion Models: A Unified Perspective — Calvin Luo](https://arxiv.org/abs/2208.11970) | ELBO→VDM→score 一条线推导最细的教程论文，比博客严格、比原论文平缓 |
| [Elucidating the Design Space of Diffusion-Based Generative Models（EDM）](https://arxiv.org/abs/2206.00364) | 实现/调参圣经：σ 参数化、采样器、预条件设计；复现 bridge 模型时的对照系（NeurIPS 2022） |

---

## 5. 课程、讲义与视频

## 5.1 首选课程

| 课程 | 内容 | 适合谁 | 优先级 |
|---|---|---|:---:|
| [MIT 6.S184: Introduction to Flow Matching and Diffusion Models](https://diffusion.csail.mit.edu/2026/) | ODE/SDE、Fokker–Planck、score、flow matching、DiT；含录播、讲义和从零实验 | 想快速补齐生成动力学理论与实现 | S |
| [Computational Optimal Transport](https://optimaltransport.github.io/) | Peyré & Cuturi 教材、代码与数学资源 | OT 零基础到研究级 | S |
| [Stanford CS236: Deep Generative Models](https://deepgenerativemodels.github.io/) | VAE、flow、EBM、diffusion 等生成模型全景 | 需要补生成建模背景 | A |
| [Hugging Face Diffusion Course](https://huggingface.co/learn/diffusion-course/unit0/1) | 以 notebook 和 Diffusers 为中心的实践课程 | 想先跑通训练/采样代码 | A |
| [MIT OCW: Diffusion and Score-Based Generative Models](https://ocw.mit.edu/courses/res-9-008-brain-and-cognitive-sciences-computational-tutorials/pages/diffusion-and-score-based-generative-models/) | Yang Song 的 score / SDE 直觉讲座 | 论文前快速建立直觉 | A |
| [UIUC ECE 598ZZ: Generative Modeling with Diffusion and Flow Matching](https://courses.grainger.illinois.edu/ECE598ZZ/sp2026/) | 2026 研究型课程，覆盖理论、采样、图像与蛋白应用 | 已懂基础，希望追前沿 | A |
| [CMU 10-799: Diffusion & Flow Matching](https://kellyyutonghe.github.io/10799S26/) | 2026 研究课程，含离散模型、distillation 与前沿论文 | 需要论文 seminar 式路线 | A |
| [A Primer on Optimal Transport — Cuturi & Solomon（NIPS 2017 Tutorial）](https://marcocuturi.net/) | 3 小时视频教程：OT→Sinkhorn→ML 应用的最快入口（slideslive/YouTube 检索 "A Primer on Optimal Transport NIPS 2017"） | 想用最短时间建立 OT 全景 | A |
| [Mathematical Tours 主站 +《Mathematical Foundations of Data Sciences》讲义 — Peyré](https://mathematical-tours.github.io/) | Peyré 成体系课程讲义（含 OT 章节）与全部 numerical tours 的母站（§6 的 MATLAB 页面是其子页面） | 想要讲义 + 可运行练习成套材料 | A |

## 5.2 机器人、RL 与 Sim2Real

| 课程 | 内容 | 建议用途 |
|---|---|---|
| [Stanford CS331B: Interactive Simulation for Robot Learning](https://web.stanford.edu/class/cs331b/) | 仿真、机器人学习、domain randomization、sim-to-real 专题 | 最直接的 sim2real 课程阅读表 |
| [UC Berkeley CS 285: Deep Reinforcement Learning](https://rail.eecs.berkeley.edu/deeprlcourse/) | RL、model-based RL、offline RL、imitation learning | 补策略学习和下游评价 |
| [MIT Underactuated Robotics](https://underactuated.csail.mit.edu/) | 动力学、轨迹优化、控制与学习 | 补“视觉桥之外”的物理/控制约束 |
| [Stanford CS224R: Deep Reinforcement Learning](https://cs224r.stanford.edu/) | 深度 RL 与机器人学习 | 跟踪最新机器人策略学习 |
| [Stanford Sim2Real Robot Learning Notes](https://web.stanford.edu/~srliu/homepage/notes/cs224r/12-sim2real-robot-learning/) | domain randomization、adaptation、calibration | 一次读完 sim2real 方法谱系 |

### Sim2Real 综述与奠基论文

| 论文 | 定位 |
|---|---|
| [Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey — Zhao, Queralta & Westerlund（2020）](https://arxiv.org/abs/2009.13303) | 被引最多的 sim2real 综述，方法谱系（DR / DA / 系统辨识）一文打底 |
| [Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World — Tobin et al.（IROS 2017）](https://arxiv.org/abs/1703.06907) | Domain randomization 奠基论文，§8 所有 DR 对照的源头 |
| [Robot Learning from Randomized Simulations: A Review — Muratore et al.（2022）](https://arxiv.org/abs/2111.00956) | DR 一支更系统的近年 review，补足"随机化分布如何选"的方法论 |

### 推荐学习组合

- **数学优先**：Computational OT → Särkkä & Solin → Léonard → GSBM。
- **实现优先**：MIT 6.S184 labs → TorchCFM → I²SB / UNSB → GSBM。
- **机器人优先**：CS331B → CS285 offline/model-based RL → ManiSkill / Isaac Lab → SB 轨迹实验。

---

## 6. 优质博客与可视化教程

| 资源 | 亮点 | 阅读时机 |
|---|---|---|
| [Diffusion Meets Flow Matching: Two Sides of the Same Coin](https://diffusionflow.github.io/) | 交互式解释 Gaussian FM 与 diffusion 的等价和参数化转换 | 学完基本 diffusion 后立即读 |
| [Yang Song: Generative Modeling by Estimating Gradients of the Data Distribution](https://yang-song.net/blog/2021/score/) | score matching、Langevin 与 SDE 直觉最好的作者教程之一 | 读 SDE 论文前 |
| [Lilian Weng: What Are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) | 公式和模型谱系整理清楚 | 快速查 DDPM / score 术语 |
| [Sander Dieleman: Perspectives on Diffusion](https://sander.ai/2023/07/20/perspectives.html) | 从多个视角解释相同 diffusion 机制 | 已懂基本公式后加深直觉 |
| [Primer on Score-Based Generative Models](https://argmax.blog/posts/primer-score-based/) | 比原论文更平缓的 score 入门 | 概率背景较弱时 |
| [A Pedagogical Introduction to Score Models](https://ericmjl.github.io/score-models/) | notebook 风格、重视可实现性 | 想自己写小模型时 |
| [POT Tutorials](https://pythonot.github.io/auto_examples/index.html) | 从离散 OT、Sinkhorn 到 domain adaptation 的可运行例子 | OT 实践阶段 |
| [Mathematical Tours: Optimal Transport](https://www.numerical-tours.com/matlab/optimaltransp_1_linprog/) | 可视化和数值练习丰富 | 第一次实现 OT |
| [CVPR 2022 Tutorial: Denoising Diffusion-based Generative Modeling — Kreis, Gao & Vahdat](https://cvpr2022-tutorial-diffusion-models.github.io/) | 视频 + slides 全套，SDE/ODE/引导采样一次讲完 | 系统过一遍 diffusion 基础时 |
| [CVPR 2023 Tutorial: Denoising Diffusion Models: A Generative Learning Big Bang](https://cvpr2023-tutorial-diffusion-models.github.io/) | 2022 版续作，覆盖更新的架构与应用 | 读完 2022 版之后 |

博客用于建立直觉，不应替代论文中的假设、定理和实验设置。

---

## 7. 代码库与复现实验工具箱

## 7.1 OT、SDE 与 Flow 基础设施

| 代码库 | 用途 | 推荐程度 |
|---|---|:---:|
| [POT: Python Optimal Transport](https://pythonot.github.io/) | NumPy/PyTorch/JAX 风格 OT、Sinkhorn、Gromov-Wasserstein、domain adaptation | S |
| [OTT-JAX](https://ott-jax.readthedocs.io/) | JAX 上高性能、可微的 OT 工具 | S |
| [GeomLoss](https://www.kernel-operations.io/geomloss/) | 大规模 Sinkhorn divergence 与 point cloud / measure loss | S |
| [TorchCFM](https://github.com/atong01/conditional-flow-matching) | Conditional FM、OT-CFM、[SF]²M | S |
| [torchsde](https://github.com/google-research/torchsde) | GPU 可微 SDE solver 与 adjoint sensitivity | A |
| [Diffusers](https://github.com/huggingface/diffusers) | 工业级 diffusion / flow pipeline 与 scheduler | A |

## 7.2 SB 官方或作者实现

| 代码库 | 对应论文 / 备注 | 从哪里开始 |
|---|---|---|
| [NVlabs/I2SB](https://github.com/NVlabs/I2SB) | I²SB 官方 PyTorch 实现 | 先跑 paired restoration |
| [cyclomon/UNSB](https://github.com/cyclomon/UNSB) | UNSB 官方实现 | 跑 unpaired Cat↔Dog 或自建 sim↔real |
| [facebookresearch/generalized-schrodinger-bridge-matching](https://github.com/facebookresearch/generalized-schrodinger-bridge-matching) | GSBM；仓库已归档但代码可用 | 先运行 `example_CondSOC`，再改 state cost |
| [facebookresearch/adjoint_sampling](https://github.com/facebookresearch/adjoint_sampling) | Adjoint Sampling 官方实现 | 先看 toy energy，再看分子 conformer |
| [TorchCFM](https://github.com/atong01/conditional-flow-matching) | [SF]²M 收录在同一库 | 先比较 independent / OT coupling |
| [thu-ml/DiffusionBridge](https://github.com/thu-ml/DiffusionBridge) | DBIM / diffusion bridge 快速采样 | 对比 NFE 与质量 |
| [LightSB](https://github.com/SKholkin/LightSB-Matching) | Light / Optimal SBM | 适合先做低维与轻量基线 |
| [yuyang-shi/dsbm-pytorch](https://github.com/yuyang-shi/dsbm-pytorch) | DSBM（§2 #8，S 级） | 对照 bridge matching / IMF 训练流程 |
| [gregkseno/csbm](https://github.com/gregkseno/csbm) | Categorical SBM（ICML 2025，S 级） | VQ / token / 类别空间 SB 实验 |
| [WanliHongC/Smooth_SB](https://github.com/WanliHongC/Smooth_SB) | Smooth SB（ICML 2025，S 级） | 平滑 GP reference 的轨迹推断 |
| [panostheo98/3MSBM](https://github.com/panostheo98/3MSBM) | 3MSBM（NeurIPS 2025，S 级） | 多时间边际 + 动量的轨迹实验 |
| [n3il666/grasp2grasp](https://github.com/n3il666/grasp2grasp) | Grasp2Grasp（NeurIPS 2025，S 级，对本项目最直接） | 异构灵巧手视觉抓取翻译复现 |
| [DenisBless/UnderdampedDiffusionBridges](https://github.com/DenisBless/UnderdampedDiffusionBridges) | Underdamped DB（ICLR 2025，S 级） | 相空间采样与更少离散步数 |
| [cookbook-ms/topological_SB_matching](https://github.com/cookbook-ms/topological_SB_matching) | Topological SBM（ICLR 2025 Spotlight） | 图 / 单纯复形上的 SB |
| [ywxjm/DehazeSB](https://github.com/ywxjm/DehazeSB) | DehazeSB（ICCV 2025，S 级） | 真实非配对去雾复现 |
| [AlexandreGUO2001/DASBS](https://github.com/AlexandreGUO2001/DASBS) | Discrete ASBS（ICML 2026） | 任意离散 source → 能量目标采样 |

### 复现时必须记录

- source / target 是否 paired、semi-paired、unpaired。
- 使用什么 coupling：independent、minibatch OT、全局 OT、learned plan。
- reference process、noise scale、time discretization 与 NFE。
- 目标是边际匹配、条件映射，还是 path / trajectory optimality。
- 评估是否只看 FID / LPIPS，还是同时看语义、几何、动作和下游 policy success。

---

## 8. Sim2Real 平台、数据与实践资源

| 平台 / 数据 | 擅长方向 | 与 SB 项目的结合方式 |
|---|---|---|
| [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) | GPU 物理仿真、并行 RL、机器人部署 | 建立 dynamics / visual randomization source distribution |
| [ManiSkill](https://maniskill.readthedocs.io/) | 高吞吐视觉操作、heterogeneous scenes、sim2real 示例 | 快速构造大规模 sim trajectories 与视觉对 |
| [robosuite](https://robosuite.ai/) | MuJoCo 操作、控制器、domain randomization | 做轻量视觉/状态/动作联合分布桥 |
| [AI Habitat](https://aihabitat.org/) | 具身导航、真实扫描环境、RGB-D | 测试 image/feature bridge 是否改善真实扫描泛化 |
| [LeRobot](https://github.com/huggingface/lerobot) | 真实机器人数据、策略训练与统一数据格式 | 把真实端数据与模拟端轨迹接入同一 pipeline |
| [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | 多任务终身机器人学习 | 测试 bridge 是否保持任务语义和跨任务迁移 |
| [MimicGen](https://github.com/NVlabs/mimicgen) | 从少量示范扩增大量机器人轨迹 | 与 SB trajectory augmentation 做公平对照 |

### 建议的最小 Sim2Real benchmark

```text
数据：
  sim RGB/RGB-D + state + action + task id
  少量 real RGB/RGB-D + state/action（可选）

对照：
  no adaptation
  domain randomization
  CycleGAN / CUT 或 feature UDA
  deterministic OT / OT-CFM
  I²SB（paired）
  UNSB / SB Flow（unpaired）
  GSBM（task cost）

评价：
  image: FID/KID, LPIPS, DINO distance
  geometry: depth/pose/keypoint consistency
  action: action consistency / inverse dynamics error
  trajectory: smoothness, endpoint error, collision/safety
  policy: real success rate, robustness, calibration
```

必须坚持：**真实环境策略成功率是主指标，视觉指标只是诊断指标。**

---

## 9. 按研究问题选论文

### 9.1 我有 paired sim-real 数据

优先顺序：

1. I²SB。
2. Diffusion Bridge Implicit Models。
3. Feedback SBM（将大量数据视为 unpaired、少量数据为 paired）。
4. GSBM（加入 depth / pose / action consistency）。

### 9.2 我只有 unpaired sim 与 real 图像

优先顺序：

1. UNSB。
2. Schrödinger Bridge Flow for Unpaired Data Translation。
3. Degradation-Aware Dynamic SB。
4. DehazeSB / day-to-night event translation。
5. OT-CFM / [SF]²M 作为高效对照。

### 9.3 我要迁移整条机器人轨迹

优先顺序：

1. Smooth Schrödinger Bridges。
2. 3MSBM。
3. Underdamped Diffusion Bridges。
4. GSBM with state/action/safety costs。
5. Treatment Stitching 与 MimicGen 作为数据增广对照。

### 9.4 我只有目标能量、奖励或可微代价

优先顺序：

1. Adjoint Sampling。
2. Adjoint Schrödinger Bridge Sampler。
3. Functional Adjoint Sampler（场/轨迹）。
4. Value Gradient Guidance for FM。
5. GSBM（当条件控制问题可直接求或近似求时）。

### 9.5 我的空间不是普通欧氏向量

| 数据结构 | 首选方向 |
|---|---|
| token / categorical / VQ | Categorical SBM、Discrete Adjoint Matching |
| graph / molecule | DDSBM、Topological SBM |
| point cloud / 3D | P2P-Bridge、BridgeShape、CounterPC |
| function / physical field | Functional Adjoint Sampler、PalSB |
| multi-time trajectory | Smooth SB、3MSBM |
| constrained state space | Reflected SB、GSBM |

---

## 10. 十二周快速学习与研究计划

先按时间预算选路径：**时间预算 < 1 个月 → 走下方"4 周速通路径"；≥ 3 个月 → 走十二周表**。两条路径的实操主线与 §13"本周四步"相同，只是粒度不同。

| 周 | 目标 | 阅读 / 实践 | 产出 |
|---:|---|---|---|
| 1 | OT 直觉与离散求解 | Computational OT Ch. 2/4；POT tutorial | 实现 primal OT 与 Sinkhorn |
| 2 | 动态 OT 与 EOT | Benamou–Brenier；Léonard 综述导论 | 一页公式关系图 |
| 3 | SDE 基础 | Särkkä & Solin Ch. 3–7 | 模拟 OU / controlled diffusion |
| 4 | Diffusion / score | Song SDE；MIT 6.S184 Lab 1–2 | 自写 2D score model |
| 5 | Flow matching | FM、Stochastic Interpolants、交互博客 | 自写 conditional FM |
| 6 | 经典神经 SB | DSB、I²SB、DSBM | 跑通一个 paired bridge |
| 7 | 非配对 SB | UNSB、SB Flow、[SF]²M | 跑通 sim↔real 小数据实验 |
| 8 | 任务代价 | GSBM、Feedback SBM | 加 geometry / action cost |
| 9 | Adjoint | Adjoint Sampling、ASBS | toy energy 与任意 prior 实验 |
| 10 | 结构化空间 | discrete / functional / topological SB | 选一个与数据结构匹配的分支 |
| 11 | 轨迹与机器人 | Smooth SB、3MSBM、Grasp2Grasp | trajectory prototype |
| 12 | 严格比较 | ablation、OOD、real policy evaluation | 可投稿式实验表与失败分析 |

### 4 周速通路径（fast-track）

面向"4 周内上手 SB + sim2real 并跑出第一个对照实验"的读者；每周主线对应 §13"本周四步"中的一步（W1→第 1 步，……，W4→第 4 步）。

| 周 | 主线（做） | 支线（读） | 完成标准 |
|---|---|---|---|
| W1 | MIT 6.S184 Lab 1–3；用 POT / GeomLoss 在自己的 sim-real 特征上比较 independent / Sinkhorn coupling（§13 第 1 步） | Yang Song 博客、Computational OT Ch. 2/4（只读这两章） | 能自写 2D score/FM 模型；能解释 ε 对 coupling 的影响 |
| W2 | 跑通 NVlabs/I2SB（paired）与 cyclomon/UNSB（unpaired）各一个数据集（§13 第 2 步）；用 DBIM / thu-ml 把 NFE 降到 20 | DSBM 方法节、[SF]²M 摘要与实验节 | 有第一张 sim→real 翻译图 + FID/LPIPS 基线表 |
| W3 | GSBM 官方 repo 的 `example_CondSOC` → 换成自己的 state cost（depth / keypoint / action 一致性，§13 第 3 步） | GSBM 正文、FSBM（半配对思想）、Adjoint Matching 摘要 | 带任务代价的桥 vs 纯视觉桥的对照曲线 |
| W4 | 按 §8 最小 benchmark 搭 no-adapt / DR / UNSB / GSBM 四路对照，以 real policy success 为主指标（§13 第 4 步） | Grasp2Grasp、EmbodiedSplat（系统路线对照） | 一页实验报告 + 失败案例清单 |

理论债（Girsanov、HJB、Léonard 综述、Fleming–Soner 等）在速通路径里**按需回补**，不作为前置：遇到推导卡点时再回 §4 对应小节。

### 每篇论文的统一笔记模板

```markdown
# 论文标题

## 一句话问题
## 输入、输出、监督类型
## 静态 / 动态 / path-space 目标
## reference process 与 coupling
## 训练目标和采样算法
## 理论保证及其假设
## 计算复杂度、NFE、显存
## 实验数据和强基线
## 失败模式
## 对当前 Sim2Real 项目的可复用点
## 最小复现实验
```

---

## 11. 值得持续追踪的开放问题

1. **任务正确而非视觉逼真**  
   如何把动作可执行性、接触、深度、关键点、碰撞和策略价值写进动态输运代价，并避免“图像更真但控制更差”。

2. **少量 paired + 大量 unpaired**  
   Feedback SBM 已给出清晰方向，但机器人数据中的配对通常有时延、标定和 embodiment mismatch，需要弱配对或不确定 coupling。

3. **路径级而非帧级迁移**  
   逐帧翻译会破坏时序一致性；Smooth SB、3MSBM、fractional bridge 与 functional sampler 是更合适的基础。

4. **不同状态/动作空间之间的桥**  
   异构机器人并不共享同一欧氏空间。需要 task-relevant latent、topological / Gromov-Wasserstein 结构或 learned correspondence。

5. **安全与约束**  
   Reflected SB、GSBM 和 constrained SOC 可用于工作空间、碰撞、关节限位、稳定性等硬/软约束。

6. **能量模型的可信性**  
   Adjoint sampler 的能力取决于 energy / reward。若能量在 OOD 区域失真，sampler 可能高效地产生错误样本。

7. **求解器误差与真实收益之间的关系**  
   边际 KL、path cost、NFE、FID 与真实策略成功率之间通常没有单调关系，需要专门建立诊断矩阵。

8. **在线更新与闭环数据采集**  
   静态 sim→real bridge 之后，可以用真实失败数据反向更新 simulator、coupling、cost 或 target energy，形成 real-to-sim-to-real 闭环。

---

## 12. 如何持续更新这份清单

### 固定检索入口

- [NeurIPS Proceedings](https://proceedings.neurips.cc/)
- [PMLR / ICML](https://proceedings.mlr.press/)
- [ICLR Proceedings](https://proceedings.iclr.cc/)
- [OpenReview](https://openreview.net/)
- [CVF Open Access：CVPR / ICCV](https://openaccess.thecvf.com/)
- [ECVA：ECCV](https://www.ecva.net/papers.php)
- [AAAI Proceedings](https://ojs.aaai.org/index.php/AAAI)
- [arXiv cs.LG](https://arxiv.org/list/cs.LG/recent)
- [arXiv cs.RO](https://arxiv.org/list/cs.RO/recent)

### 推荐关键词

```text
"Schrödinger bridge"
"bridge matching"
"diffusion bridge"
"adjoint matching" OR "adjoint sampler"
"entropic optimal transport" AND generative
"flow matching" AND optimal transport
"unpaired translation" AND bridge
"trajectory inference" AND Schrödinger
"sim-to-real" AND optimal transport
"real-to-sim-to-real"
"cross-domain imitation" AND transport
```

### 收录规则

1. 先核验主会 proceedings / OpenReview 决定，后看 arXiv。
2. 主会、Workshop、预印本分开标记。
3. 保存正式版本、代码、项目页与补充材料。
4. 至少记录：监督类型、空间类型、coupling、reference、目标、复杂度、数据集和主要限制。
5. 对 Sim2Real 只保留能明确影响感知、动作、轨迹或策略迁移的论文。

---

## 13. 当前最推荐的实际起点

若本周就开始做实验，建议只做下面四步：

1. 用 POT / GeomLoss 在当前 sim-real feature 上比较 independent、Sinkhorn、Gromov-Wasserstein coupling。
2. 用 I²SB 或 UNSB 建立 paired / unpaired 图像桥基线。
3. 用 GSBM 加入 depth、keypoint、action 或 policy feature consistency。
4. 最终只以真实域 policy success、robustness 和安全失败率决定方法是否有效。

这四步跑通后，再根据瓶颈进入 Adjoint、Multi-Marginal、Functional 或 Discrete 分支，能避免过早把研究复杂化。

这四步的周粒度展开与完成标准见 §10"4 周速通路径"（W1–W4 与上述四步一一对应）；时间预算 ≥ 3 个月则按 §10 十二周表推进。
