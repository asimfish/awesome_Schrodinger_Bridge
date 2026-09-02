# Guan-Horng Liu 研究工作专题调研

## 从最优控制神经优化器到 Schrödinger Bridge、Adjoint Sampling 与 LLM Post-training

> 调研对象：[Guan-Horng Liu 个人主页](https://ghliu.github.io/)  
> 核验日期：2026-07-31  
> 主要依据：作者[个人主页](https://ghliu.github.io/)、[CV](https://ghliu.github.io/assets/pdf/cv.pdf)、[博士论文](https://repository.gatech.edu/entities/publication/895044f4-a1e1-403b-8414-1f6aabacadbf)、正式会议页面、论文原文与作者代码  
> 本文目标：解释研究脉络、方法之间的因果关系，以及这些工作应如何学习和用于 `SB-Render-Lite / Sim2Real`

---

## 0. 一页结论

Guan-Horng Liu 的研究主线不应被简单概括成“做 Schrödinger Bridge”。更准确的理解是：

> **以动力系统和最优控制为统一语言，把难以直接求解的路径空间优化问题，转化为可扩展、可监督回归的神经算法，并针对具体领域加入正确的结构。**

他的工作大致形成了四个连续阶段：

```text
机器人、规划与控制
    ↓
把神经网络训练视作动力系统最优控制
    ↓
用 FBSDE / stochastic control 求解 Schrödinger Bridge
    ↓
用 matching / adjoint 把路径控制变成可扩展的监督学习
    ↓
连续、离散、函数空间以及科学计算和 LLM reward sampling
```

如果只选最值得学习的七项工作，建议按以下顺序：

1. [博士论文：Large-Scale Optimization for Deep Neural Network Architecture](https://repository.gatech.edu/entities/publication/895044f4-a1e1-403b-8414-1f6aabacadbf)——理解统一思想。
2. [Likelihood Training of Schrödinger Bridge using Forward-Backward SDEs](https://arxiv.org/abs/2110.11291)——理解 FBSDE 如何进入生成建模。
3. [Deep Generalized Schrödinger Bridge](https://arxiv.org/abs/2209.09893)——理解 state cost、mean-field interaction 与社会动力学。
4. [I²SB](https://arxiv.org/abs/2302.05872)——理解如何利用 paired endpoints 得到可扩展图像 bridge。
5. [Generalized Schrödinger Bridge Matching](https://arxiv.org/abs/2310.02233)——理解 conditional stochastic optimal control 与 matching 的结合。
6. [Adjoint Sampling](https://arxiv.org/abs/2504.11713)——理解只有 energy/reward、没有目标样本时怎样训练 sampler。
7. [Adjoint Schrödinger Bridge Sampler](https://arxiv.org/abs/2506.22565)——理解怎样从 arbitrary source prior 到 energy target 建立完整 SB。

对当前 Sim2Real 课题，最直接的组合不是照搬单篇论文，而是：

```text
GSBM 的任务路径代价
    + I²SB / Feedback SBM 的配对数据利用方式
    + ASBS 的 target energy 引导
    + 3MSBM 的时序多边缘扩展与 FAS 的函数空间 energy-only 扩展
```

---

## 1. 人物与研究定位

### 1.1 当前身份

根据作者主页与 2026 年 CV：

- 现任 Meta Superintelligence Labs（MSL）Research Scientist，2024 年加入 Meta。
- 2024 年于 Georgia Tech 获 Machine Learning PhD，导师包括 Evangelos A. Theodorou。
- 曾在 Meta FAIR、NVIDIA Research 实习。
- 当前兴趣横跨：
  - post-training、reinforcement learning 与 large reasoning models；
  - AI for Science、Boltzmann / diffusion sampling、分子生成；
  - stochastic optimal control、Schrödinger Bridge 与 dynamic optimal transport。

作者自己对研究的概括很重要：一类工作研究 **data-driven Schrödinger Bridge**，如 I²SB、DeepGSB；另一类研究 **data-efficient adjoint diffusion samplers**，如 ASBS、DAM；近期则开始探索 LLM post-training 中的 RL science。

### 1.2 研究能力画像

| 能力 | 在论文中的体现 | 对学习者的启示 |
|---|---|---|
| 从控制论重写 ML 问题 | DDPNOpt、Dynamic Game Optimizer、SB-FBSDE | 先问“优化对象与状态动力学是什么”，再选网络 |
| 从最优性条件构造训练目标 | FBSDE、GSBM、Adjoint Matching | 推导不是装饰，而是决定可训练 loss |
| 利用问题结构降低计算量 | I²SB、Mirror Diffusion、ASBS | paired endpoint、对称性、先验分布都应进入算法 |
| 从连续空间推广到新对象 | multi-marginal、discrete、function space | 区分空间结构后再选择 SDE、CTMC 或 Hilbert-space 方法 |
| 跨领域验证同一原则 | 图像、分子、粒子物理、群体动力学、语言 | 先抽象共同的 path-control 问题，再做领域适配 |

### 1.3 不要把他的所有工作都叫作 SB

需要保持以下边界：

- [React-OT](https://arxiv.org/abs/2404.13430) 是确定性 optimal transport，不是 Schrödinger Bridge。
- [Mirror Diffusion Models](https://arxiv.org/abs/2310.01236) 是面向凸约束的 structured diffusion，不是 SB 求解器。
- DDPNOpt、SNOpt 等是 neural optimizer，与生成式 SB 共用控制论思想，但任务不同。
- DAM、MDNS、DASBS 都在离散空间工作，但优化目标和路径构造并不相同。
- 当前 LLM post-training 是新的延伸方向，公开论文积累尚不像 SB 主线那样完整。

---

## 2. 研究谱系：每个阶段解决了什么

### 2.1 第一阶段：机器人与真实动力系统，2013–2017

早期工作涉及 Kangaroo robot、无人水面艇、越野自主驾驶和多模态传感器策略。这个阶段不是后续学习的必读文献，但解释了三个长期特点：

1. 研究对象从一开始就是带动力学和约束的系统，而不是静态预测。
2. 对多传感器、复杂环境、模型误差和实际控制计算量敏感。
3. 后续反复出现的 state、control、trajectory、feedback policy 都有机器人学来源。

对于只想学习 SB 的读者，可以读 CV 和 CMU 硕士论文摘要，不必逐篇复现。

### 2.2 第二阶段：把神经网络视为动力系统，2019–2021

#### DDPNOpt：训练网络是一类 trajectory optimization

[DDPNOpt（ICLR 2021 Spotlight）](https://arxiv.org/abs/2002.08809) 将网络层看作离散动力系统的时间步，将参数更新与 Differential Dynamic Programming 联系起来。

核心贡献不是“又一个 optimizer”，而是：

- backpropagation 可以看作一阶局部控制更新；
- DDP 提供二阶近似和 layer-wise feedback policy；
- 优化器可以利用网络深度方向的动力学结构，而不只把所有参数看成一个大向量。

#### Dynamic Game Theoretic Neural Optimizer：跳连意味着多主体依赖

[Dynamic Game Theoretic Neural Optimizer（ICML 2021 Oral）](https://proceedings.mlr.press/v139/liu21d.html) 进一步处理 skip connection 等非纯 Markov 依赖：不同层可视作相互作用的 players，以动态博弈而非单一最优控制描述。

这项工作体现了一个反复出现的方法论：

> 当标准动力系统假设被网络结构破坏时，不是忽略结构，而是换一个更适合的控制模型。

#### SNOpt：从 discrete backprop 到 differential programming

[Second-Order Neural ODE Optimizer（NeurIPS 2021 Spotlight）](https://arxiv.org/abs/2109.14158) 面向 Neural ODE：

- 推导高阶 backward ODE；
- 使用低秩 / Kronecker 结构控制二阶计算；
- 保留 continuous-depth 模型的低内存优势；
- 官方实现：[ghliu/snopt](https://github.com/ghliu/snopt)。

这条线最终由 [Optimal Control Theoretic Neural Optimizer（TPAMI 2026）](https://pubmed.ncbi.nlm.nih.gov/41259159/) 作了更完整的统一总结。

#### 这一阶段与 SB 的真正连接

它训练了后续工作的三项关键技术：

```text
动力系统建模
    + 最优性条件 / adjoint 推导
    + 把求解控制方程转化为可训练神经算法
```

如果没有这部分背景，很容易把后续 FBSDE 和 adjoint matching 误解为孤立技巧。

---

## 3. 核心主线一：FBSDE 与 Generalized Schrödinger Bridge

### 3.1 SB-FBSDE：从 bridge optimality 到 likelihood training

[Likelihood Training of Schrödinger Bridge using Forward-Backward SDEs（ICLR 2022）](https://arxiv.org/abs/2110.11291) 是研究转折点。

它把 Schrödinger Bridge 写成随机最优控制，并通过 forward-backward SDE 表达最优性条件。其意义在于：

- 不再只把 SB 看成静态边际耦合；
- forward 与 backward dynamics 共同决定路径分布；
- 可构造 likelihood-oriented objectives；
- score-based generative modeling 可被视为这一框架的特殊情形。

官方代码：[ghliu/SB-FBSDE](https://github.com/ghliu/SB-FBSDE)。

阅读时应重点掌握四个对象：

1. reference diffusion；
2. forward / backward control；
3. potential、score 与 value function 的关系；
4. endpoint constraints 如何通过 FBSDE 被满足。

### 3.2 DeepGSB：把状态代价和群体相互作用放进 bridge

[Deep Generalized Schrödinger Bridge（NeurIPS 2022 Oral）](https://proceedings.neurips.cc/paper_files/paper/2022/hash/3d17b7f7d52c83ab6e97e2dc0bda2e71-Abstract-Conference.html) 不再只最小化相对 reference path measure 的 KL，而是加入一般 state cost 和 mean-field interaction。

它解决的问题包括：

- 运输路径需要避障或满足领域代价；
- 个体代价依赖总体分布；
- 目标既是生成建模，也可能是 mean-field game；
- 高维 PDE 难以直接求解。

论文以 FBSDE 和近似 actor–critic / temporal-difference 结构求解，并展示了最高 1000 维的 opinion depolarization。

官方代码：[ghliu/DeepGSB](https://github.com/ghliu/DeepGSB)。

对 Sim2Real 最重要的思想是：

> source 和 target 边际匹配并不足够；中间路径应受到 geometry、action consistency、collision risk 或 task semantics 的约束。

### 3.3 Momentum Multi-Marginal SB：从两端点到多个观测时刻

[Deep Momentum Multi-Marginal Schrödinger Bridge（NeurIPS 2023）](https://arxiv.org/abs/2303.01751) 将问题扩展到：

- 多个时间边际；
- position–velocity phase space；
- 只有位置 snapshot，速度可能不可观测；
- 需要更平滑、更符合惯性的插值轨迹。

这为 2025 年的 3MSBM 奠定背景。对机器人、视频、细胞轨迹而言，“起点到终点”通常过于粗糙，多边际才是自然问题。

### 3.4 GSBM：把昂贵的全局 GSB 变成 conditional SOC + matching

[Generalized Schrödinger Bridge Matching（ICLR 2024）](https://proceedings.iclr.cc/paper_files/paper/2024/hash/3e91dd686ea22cb51b8732e3c7e9fc8e-Abstract-Conference.html) 是这一主线最适合工程化的一篇。

其核心分解为：

```text
采样 endpoint pairs / coupling
        ↓
对每对 endpoints 解 conditional stochastic optimal control
        ↓
用 matching objective 学习全局 Markov drift
        ↓
更新 coupling / 重复
```

相比只给定简单 Brownian reference 的 bridge matching，它允许：

- task-specific state cost；
- nonlinear dynamics；
- 更合理的中间路径；
- crowd navigation、opinion dynamics、LiDAR、图像等不同结构。

官方代码：[facebookresearch/generalized-schrodinger-bridge-matching](https://github.com/facebookresearch/generalized-schrodinger-bridge-matching)。仓库已归档为只读，但仍是重要参考实现。

#### GSBM 的局限

- conditional SOC 本身可能昂贵；
- state cost 通常需要可微或有可用梯度；
- 高维 RGB 直接求解未必现实，latent / feature space 更合适；
- cost 设计错误会得到“数学上优化、任务上有害”的路径。

### 3.5 适合回顾的统一资料

[Deep Generalized Schrödinger Bridges: From Image Generation to Solving Mean-Field Games](https://arxiv.org/abs/2412.20279) 是回看这条线的好入口。建议先读 SB-FBSDE 与 GSBM，再读综述；否则容易只记分类，不理解各算法为何出现。

---

## 4. 核心主线二：结构化生成与科学应用

### 4.1 I²SB：paired endpoints 是算法资源，不只是数据设定

[I²SB（ICML 2023）](https://proceedings.mlr.press/v202/liu23ai.html) 面向 image-to-image translation / restoration。当训练数据给出成对 degraded–clean endpoints 时，条件 bridge 的中间分布可以解析采样，从而避免昂贵的 iterative fitting。

关键思想：

- source 不是无信息 Gaussian noise，而是观测图像；
- paired endpoints 让 conditional bridge tractable；
- 训练时直接构造中间状态和目标；
- 适合 restoration、colorization、JPEG recovery 等任务。

官方项目与代码：

- [I²SB 项目页](https://i2sb.github.io/)
- [NVlabs/I2SB](https://github.com/NVlabs/I2SB)

对 Sim2Real 的启示是：如果仿真能提供相同场景的 controlled paired render，应该利用这种配对结构，而不是强行当作完全 unpaired translation。

### 4.2 Mirror Diffusion Models：约束应改变建模空间

[Mirror Diffusion Models（NeurIPS 2023）](https://arxiv.org/abs/2310.01236) 使用 mirror map 把凸约束域映射到更易处理的 dual space，在保留 diffusion 可训练性的同时满足原空间约束。

官方代码：[ghliu/mdm](https://github.com/ghliu/mdm)。

这项工作提示：

- 有效域、simplex、bounded control 或 watermark constraints 不应仅靠后处理；
- 正确的坐标和几何有时比更大的生成网络更重要；
- “选择 reference process / state space”本身就是算法设计。

### 4.3 SBUnfold：simulation-trained inverse problem 的近邻案例

[Improving Generative Model-based Unfolding with Schrödinger Bridges（Physical Review D 2024）](https://arxiv.org/abs/2308.12351) 将 detector-level 分布映射回 particle-level 分布，训练依赖模拟数据。

官方代码：[ViniciusMikuni/SBUnfold](https://github.com/ViniciusMikuni/SBUnfold)。

它与 Sim2Real 的结构相似：

```text
模拟生成 latent truth
    → detector / observation domain
    → 用 bridge 学习逆向校正
    → 在真实观测上应用
```

这篇论文的价值不在视觉质量，而在于展示如何验证 distributional correction、uncertainty 与下游物理量。

### 4.4 React-OT：如果映射接近唯一，确定性 OT 可能更合适

[React-OT（Nature Machine Intelligence 2025）](https://www.nature.com/articles/s42256-025-01010-0) 从反应物与产物生成 transition state，利用物理与几何结构构造 transport。

其重要方法论意义是：

- 并非所有过渡生成都需要随机 SB；
- 当 endpoints 强配对且目标路径接近确定性时，OT map 可能更直接；
- 化学对称性、等变性与结构表示对性能至关重要；
- 评价应落在 transition-state accuracy 和计算成本，而非通用生成指标。

对 Sim2Real，应保留 deterministic OT / regression baseline。若同一仿真状态几乎唯一对应真实状态，复杂随机桥可能没有必要。

### 4.5 Feedback SBM：少量配对数据如何引导大量非配对数据

[Feedback Schrödinger Bridge Matching（ICLR 2025 Oral）](https://arxiv.org/abs/2410.14055) 用少量 pre-aligned / paired samples 给其余 unpaired transport 提供 feedback。

它对现实数据设定尤其关键：

```text
大量便宜 simulation data
    + 大量未配对 real observations
    + 少量 calibration pairs
    → semi-paired bridge
```

作者报告少于 8% 的配对反馈即可显著影响整体 transport。对于机器人，这是比“纯 paired”或“纯 unpaired”更现实的设定。

---

## 5. 核心主线三：Adjoint Sampling

### 5.1 问题变化：从“有目标样本”到“只有目标能量”

传统生成模型通常有目标样本 \(x \sim p_{\text{data}}\)。但科学计算、reward fine-tuning 和 constrained generation 常只有：

\[
\pi(x) \propto \exp(-E(x))
\]

其中 \(E(x)\) 可计算但归一化常数未知，目标样本也可能极其昂贵。

这时真正的问题是：

- 怎样从简单或已有 prior 采样到 \(\pi\)；
- 怎样减少对昂贵 energy / reward oracle 的调用；
- 怎样复用已生成轨迹做多次 optimizer update；
- 怎样保留对称性、周期性或离散结构。

### 5.2 Adjoint Sampling：on-policy 轨迹与 adjoint regression

[Adjoint Sampling: Highly Scalable Diffusion Samplers via Adjoint Matching（ICML 2025）](https://arxiv.org/abs/2504.11713) 将 diffusion sampler 训练写成随机控制，并使用 adjoint matching 学习控制。

最值得理解的工程思想是：

> 一批昂贵的 on-policy trajectories 可以支持多次神经网络更新，而不是一次模拟只做一次更新。

这对 energy evaluation 昂贵的场景尤为重要。论文同时考虑分子结构中的对称性与周期边界。

官方代码：[facebookresearch/adjoint_sampling](https://github.com/facebookresearch/adjoint_sampling)。

#### 风险

- energy 定义不可靠时，sampler 会忠实地放大错误偏好；
- reverse-KL 风格优化可能更偏 mode-seeking；
- on-policy 训练的稳定性依赖 reference、初始化与探索；
- 高维视觉“真实感能量”通常比物理能量难以校准。

### 5.3 ASBS：任意 source prior 到 energy target 的完整桥

[Adjoint Schrödinger Bridge Sampler（NeurIPS 2025 Oral）](https://arxiv.org/abs/2506.22565) 解决 Adjoint Sampling 中 source 受限的问题，允许从 arbitrary sampleable prior 出发。

它的关键扩展包括：

- source 不必是单点或特殊 memoryless prior；
- 同时兼顾 source boundary 和 unnormalized target；
- 通过 adjoint 与 corrector / bridge 更新逼近完整 SB；
- 适合把预训练生成器、仿真分布或经验数据分布作为 source。

官方资源：

- [facebookresearch/adjoint_samplers](https://github.com/facebookresearch/adjoint_samplers)
- [作者 NeurIPS 讲稿](https://ghliu.github.io/assets/pdf/asbs_talk_neurips.pdf)
- [SlidesLive 报告](https://slideslive.com/39055157/adjoint-schrodinger-bridge-sampler)

对当前项目，ASBS 的价值在于：

```text
已有 simulation / renderer prior
    + 可微的 realness / task energy
    → 不必先收集大量 target-domain samples
    → 学习从模拟先验到目标分布的受控路径
```

但这只有在 target energy 比“直接收集真实样本”更可靠、更便宜时才成立。

### 5.4 Non-equilibrium Annealed Adjoint Sampler

[NAAS（NeurIPS 2025）](https://arxiv.org/abs/2506.18165) 设计非平衡退火 reference trajectories，使需要学习的 corrective control 更小。

它强调了 Adjoint 系列另一个核心原则：

> reference process 越接近目标路径，需要学习的 residual control 越简单。

这对 Sim2Real 可转化为：用 domain randomization、粗糙物理校准或现有 renderer adaptation 先构造“较好的 reference”，再由 adjoint 学剩余差距。

### 5.5 Momentum Multi-Marginal SBM

[Momentum Multi-Marginal Schrödinger Bridge Matching（3MSBM，NeurIPS 2025）](https://arxiv.org/abs/2506.10168) 在 phase space 中连接多个时间边际，强调平滑、惯性与 sparse snapshots。

官方代码：[panostheo98/3MSBM](https://github.com/panostheo98/3MSBM)。

它特别适合：

- 机器人 trajectory translation；
- 视频或动态场景；
- 多时刻 sim / real calibration；
- 只观测位置、需要恢复隐式速度的任务。

### 5.6 从连续向量到离散状态与函数空间

#### Discrete Adjoint Matching

[Discrete Adjoint Matching（ICLR 2026）](https://arxiv.org/abs/2602.07132) 将 adjoint matching 推广到 CTMC / discrete state space，并用于离散 diffusion 与数学推理 reward fine-tuning。

它是从科学 sampling 走向 diffusion language model 的关键桥梁：

```text
continuous SDE control
    → discrete CTMC control
    → token-level reward fine-tuning
```

#### Discrete ASBS

[Discrete Adjoint Schrödinger Bridge Sampler（ICML 2026）](https://arxiv.org/abs/2602.08243) 将 ASBS 推向离散空间，采用适合群结构的 CTMC reference，并处理离散 energy-based sampling。

需要注意：

- 早期版本曾出现在 ICLR 2026 DeLTa Workshop；
- 作者最新版主页与 CV 已将完整论文列为 **ICML 2026 主会**；
- 它与 DAM 同属 discrete adjoint 大方向，但不是同一个 objective。

#### Masked Diffusion Neural Sampler

[Masked Diffusion Neural Sampler（NeurIPS 2025）](https://arxiv.org/abs/2508.10684) 使用 masked CTMC 与 path-measure stochastic control 处理 unnormalized discrete distributions。

阅读它的目的不是重复 DAM，而是比较：

- masked reference 与 group / additive reference；
- terminal reward 与 bridge constraints；
- discrete generative modeling 的路径设计。

#### Functional Adjoint Sampler

[Functional Adjoint Sampler（ICML 2026）](https://arxiv.org/abs/2511.06239) 从有限维向量推广到 Hilbert function space，使用 stochastic maximum principle 与可扩展的函数表示学习整条 path / field 的 Gibbs distribution。

适合：

- transition path sampling；
- spatiotemporal field；
- PDE solution / functional posterior；
- 连续时间机器人轨迹。

它不是入门论文。应在掌握 Adjoint Sampling、ASBS 和基础 functional analysis 后阅读。

### 5.7 分子 collective variables：采样不只是命中，还要探索

[Enhancing Diffusion-Based Sampling with Molecular Collective Variables（ICLR 2026）](https://arxiv.org/abs/2510.11923) 在低维 collective-variable space 中加入顺序 repulsive bias，以发现更多 metastable modes，并保留正确 reweighting。

这项工作补上了 energy sampler 的一个常见短板：

- 命中低能区域不等于覆盖所有重要模态；
- 高维空间中的探索可借助可解释低维变量；
- bias 必须配合统计校正，不能只追求新颖性。

对机器人，可类比为在 task-relevant latent variables 中推动 coverage，例如接触模式、抓取类别、视角或场景拓扑。

---

## 6. 方法关系图

下面的箭头表示“主要概念或技术承接”，不等同于每篇论文都直接引用上一项：

```text
机器人规划 / 真实动力系统
            │
            ▼
DNN as dynamical system
    ├── DDPNOpt
    ├── Dynamic Game Optimizer
    └── SNOpt / differential programming
            │
            ▼
nonlinear Feynman–Kac / FBSDE / stochastic control
            │
            ├── SB-FBSDE
            │      ├── DeepGSB ── mean-field / state cost
            │      ├── Momentum multi-marginal SB
            │      └── GSBM ── conditional SOC + matching
            │
            ├── I²SB ── paired conditional bridge
            ├── Mirror Diffusion ── convex geometry
            └── scientific applications
                   ├── SBUnfold
                   └── React-OT

stochastic maximum principle / adjoint matching
            │
            ├── Adjoint Sampling
            │      └── ASBS ── arbitrary source prior
            │             ├── NAAS ── better reference
            │             └── molecular CV ── better exploration
            │
            ├── 3MSBM ── phase-space multi-marginal paths
            ├── FAS ── Hilbert function space
            └── discrete branch
                   ├── DAM ── discrete reward fine-tuning
                   ├── MDNS ── masked CTMC sampler
                   └── DASBS ── discrete source-to-energy bridge
```

---

## 7. 代表性论文地图

### 7.1 会议论文主线

| 年份 | 论文 | 会议 | 主题 | 建议 |
|---:|---|---|---|:---:|
| 2026 | Discrete Adjoint Schrödinger Bridge Sampler | ICML | 离散 ASBS、能量采样 | A |
| 2026 | Functional Adjoint Sampler | ICML | 函数空间、transition paths | A |
| 2026 | Discrete Adjoint Matching | ICLR | CTMC、reward fine-tuning、LLM | A |
| 2026 | Enhancing Diffusion-Based Sampling with Molecular Collective Variables | ICLR | 探索、多模态、reweighting | B |
| 2025 | Adjoint Schrödinger Bridge Sampler | NeurIPS Oral | arbitrary source → energy target | S |
| 2025 | Non-equilibrium Annealed Adjoint Sampler | NeurIPS | reference path / annealing | A |
| 2025 | Momentum Multi-Marginal SBM | NeurIPS | 平滑时序、多边际 | S |
| 2025 | Masked Diffusion Neural Sampler | NeurIPS | 离散 unnormalized target | B |
| 2025 | Adjoint Sampling | ICML | 高扩展性能量采样 | S |
| 2025 | Feedback SBM | ICLR Oral | 少量 paired + 大量 unpaired | S |
| 2024 | GSBM | ICLR | general cost + matching | S |
| 2024 | Robust Differential Neural ODE Optimizer | ICLR | 鲁棒 continuous-depth 优化 | C |
| 2023 | Mirror Diffusion Models | NeurIPS | 凸约束与 mirror geometry | A |
| 2023 | Deep Momentum Multi-Marginal SB | NeurIPS | phase space、多时间边际 | A |
| 2023 | I²SB | ICML | paired image bridge | S |
| 2022 | DeepGSB | NeurIPS Oral | generalized SB、mean-field game | S |
| 2022 | SB-FBSDE | ICLR | FBSDE 与 likelihood training | S |
| 2021 | Second-Order Neural ODE Optimizer | NeurIPS Spotlight | differential programming | B |
| 2021 | Dynamic Game Theoretic Neural Optimizer | ICML Oral | 网络结构与动态博弈 | B |
| 2021 | DDPNOpt | ICLR Spotlight | DDP 与网络训练 | B |

#### 其他早期会议工作

- Variational Inference MPC using Tsallis Divergence（RSS 2021）
- Multimodal Sensor Policies for Autonomous Navigation（CoRL 2017）
- 无人水面艇与 Kangaroo robot 控制工作（2013–2014）

这些工作用于理解研究起点，不必放在 SB 精读主线前面。

### 7.2 期刊与长文

| 年份 | 工作 | 定位 |
|---:|---|---|
| 2026 | Optimal Control Theoretic Neural Optimizer | TPAMI；对 neural optimizer 主线的统一总结 |
| 2025 | React-OT | Nature Machine Intelligence；结构化确定性 OT |
| 2024 | SBUnfold | Physical Review D；simulation-trained scientific inverse problem |
| 2024 | PhD thesis | 将 dynamic programming、Feynman–Kac、path integral 与神经优化统一 |

### 7.3 如何判断“作者的核心工作”

建议依据作者 CV 的作者标记、共同一作 / core contributor 说明和论文作者顺序，而不是只根据主页展示位置猜测。

从公开信息可较有把握地把以下工作视作个人研究主线：

- neural optimizer 系列；
- SB-FBSDE、DeepGSB、GSBM；
- I²SB；
- Adjoint Sampling、ASBS；
- discrete / functional adjoint 扩展。

React-OT、SBUnfold、molecular CV 等更适合称为重要合作或应用延伸，除非讨论具体论文贡献分工时有更直接的公开依据。

---

## 8. 研究“DNA”：值得模仿的五种做法

### 8.1 从 optimality structure 出发，而不是从网络模块出发

常见思路是先选择 U-Net、Transformer 或 GNN，再寻找训练目标。Liu 的工作通常反过来：

```text
定义 path-space objective
    → 推导 HJB / FBSDE / stochastic maximum principle
    → 识别需要学习的 score、value、control 或 adjoint
    → 才选择函数逼近器
```

### 8.2 把难求解的控制问题改写为 supervised regression

SB-FBSDE、GSBM 和 Adjoint Matching 形式不同，但共同追求：

- 轨迹或条件子问题产生训练 targets；
- 神经网络通过 regression / matching 学 global policy；
- 避免每个新样本都重新求昂贵最优控制。

### 8.3 把数据设定当作可利用结构

- paired endpoints → I²SB；
- 少量 paired + 大量 unpaired → Feedback SBM；
- arbitrary source prior → ASBS；
- sparse temporal marginals → 3MSBM；
- only unnormalized energy → Adjoint Sampling；
- discrete group / token space → DAM / DASBS；
- whole functions or paths → FAS。

### 8.4 认真设计 reference process

reference 不是无关紧要的默认 Brownian motion：

- underdamped / momentum reference 带来平滑性；
- mirror geometry 处理约束；
- annealed non-equilibrium reference 减少 corrective control；
- CTMC reference 决定离散状态的可达性和 inductive bias。

### 8.5 评价最终科学或任务目标

优秀应用不是只报告生成分数：

- React-OT 看 transition-state 质量；
- SBUnfold 看物理分布恢复；
- 分子 sampling 看 coverage、free energy 与 reweighting；
- mean-field game 看群体动力学目标；
- Sim2Real 最终应看真实机器人任务成功率。

---

## 9. 对 `SB-Render-Lite / Sim2Real` 的直接价值

### 9.1 优先级分层

#### S：直接形成实验设计

| 工作 | 可转化组件 |
|---|---|
| GSBM | 把 geometry、action、collision、task feature 写成路径状态代价 |
| I²SB | 利用同场景 sim–real / synthetic-corrupted paired samples |
| Feedback SBM | 小量 calibration pairs 指导大量 unpaired data |
| SBUnfold | 模拟训练、真实端应用以及 distributional validation |
| ASBS | 从 renderer / simulation prior 到 realness 或 task energy target |
| 3MSBM | 轨迹、视频、多时刻观测和速度一致性 |

#### A：作为方法支撑或强对照

| 工作 | 用途 |
|---|---|
| SB-FBSDE | 统一 score、control、forward/backward dynamics |
| DeepGSB | mean-field、群体交互与一般 state cost |
| React-OT | 近确定性配对映射的简单强基线 |
| Mirror Diffusion | bounded action、simplex 或几何约束 |
| FAS | 连续轨迹、时空场或 function-valued observation |
| NAAS | 用更好的 sim reference 降低 domain correction 难度 |

#### B：特定数据类型再读

- DAM / DASBS / MDNS：当 action、skill、language command 或环境状态被 token 化时。
- Molecular CV：当需要以少量 task variables 提升多模态 coverage 时。
- neural optimizer 系列：理解思想即可，不必优先复现。

### 9.2 推荐的项目建模

设：

- \(x_0 \sim p_{\text{sim}}\)：仿真图像、feature 或 trajectory；
- \(x_1 \sim p_{\text{real}}\)：真实域数据；
- \(a_t\)：同步动作或控制；
- \(\phi(x_t)\)：任务相关特征；
- \(E_{\text{real}}(x)\)：真实感、几何一致性或策略价值能量。

可构造：

\[
\mathcal{J}(u)
=
\mathbb{E}\left[
\int_0^1
\frac{1}{2}\|u_t\|^2
+\lambda_g c_{\mathrm{geom}}(x_t)
+\lambda_a c_{\mathrm{action}}(x_t,a_t)
+\lambda_\phi c_{\mathrm{task}}(\phi(x_t))
\,dt
+\lambda_E E_{\mathrm{real}}(x_1)
\right].
\]

对应方法选择：

- 有可靠 paired endpoints：先做 I²SB。
- 只有少量 pairs：做 Feedback SBM。
- 中间路径有任务约束：做 GSBM。
- 没有 target samples、只有可信 energy：做 ASBS。
- 有多个时间点和动力学：做 3MSBM。
- 轨迹本身是连续函数、且目标只能写成能量泛函而非轨迹样本：再考虑 FAS（energy-only，不适用于样本驱动的轨迹桥）。

### 9.3 最小可证伪实验

```text
Dataset
  paired sim / real-like calibration subset
  + large unpaired sim and real sets

Baselines
  deterministic regression / OT
  I²SB
  unpaired SB
  GSBM with task cost
  Feedback SBM

Ablations
  no task cost
  no paired feedback
  pixel vs policy-latent transport
  Brownian vs dynamics-aware reference

Metrics
  image/feature distribution distance
  geometry and temporal consistency
  action preservation
  real-domain policy success
  compute, NFE, memory, paired-data efficiency
```

必须把真实域 policy success 放在最终指标中。若生成质量上升而控制成功率下降，bridge 没有完成 Sim2Real 目标。

---

## 10. 精读路线：10 周形成研究能力

### 第 1 周：只补必要的控制背景

阅读：

- 作者博士论文的 abstract、introduction、统一方法章节和 conclusion；
- DDPNOpt 方法概览；
- stochastic control 中 HJB、Pontryagin / adjoint、Feynman–Kac。

产出：

- 用一页图解释 backprop、dynamic programming、adjoint 的关系；
- 不要求复现 neural optimizer。

### 第 2 周：SB-FBSDE

阅读：

- SB-FBSDE 正文与附录推导；
- 本地 SB 基础资源。

产出：

- 自己推一次 controlled diffusion 的 Girsanov cost；
- 写清 forward / backward potential 与 score 的关系；
- 跑官方低维例子。

### 第 3 周：DeepGSB

阅读：

- [本地 DeepGSB 报告](./2209.09893_deep_generalized_schrodinger_bridge.md)；
- [官方论文](https://arxiv.org/abs/2209.09893)；
- [NeurIPS slides](https://neurips.cc/media/neurips-2022/Slides/54873.pdf)。

产出：

- 对比 classical SB、GSB、mean-field GSB 的 objective；
- 复现一个 obstacle 或 opinion toy problem。

### 第 4 周：I²SB

阅读：

- [本地 I²SB 报告](./2302.05872_i2sb.md)；
- 官方代码的数据构造和 scheduler。

产出：

- 推导 paired conditional bridge 的均值与方差；
- 在小型 restoration 数据上跑通训练与采样；
- 记录 paired data 的真实需求。

### 第 5 周：GSBM

阅读：

- [本地 GSBM 报告](./2310.02233_generalized_schrodinger_bridge_matching.md)；
- conditional SOC、matching loss 和 coupling 更新。

产出：

- 给自己的 Sim2Real 问题定义三个可微 state costs；
- 先在 2D obstacle toy 上验证 cost 是否真正改变路径。

### 第 6 周：Feedback SBM 与应用

阅读：

- Feedback SBM；
- [SBUnfold 本地报告](./2308.12351_sb_unfold.md)；
- [React-OT 本地报告](./2404.13430_react_ot.md)。

产出：

- 构造 paired ratio 为 0%、2%、5%、10%、100% 的实验；
- 与 deterministic OT / regression 比较。

### 第 7 周：Adjoint Sampling

阅读：

- [本地 Adjoint Sampling 报告](./2504.11713_adjoint_sampling.md)；
- stochastic maximum principle 与 adjoint matching。

产出：

- 在二维多峰 energy 上训练 sampler；
- 统计每次 energy evaluation 支持的 optimizer updates；
- 检查 mode coverage，而不只看平均 energy。

### 第 8 周：ASBS

阅读：

- [本地 ASBS 报告](./2506.22565_adjoint_schrodinger_bridge_sampler.md)；
- 作者讲稿和代码。

产出：

- 用非 Gaussian、可直接采样的 source prior 做 toy bridge；
- 比较 Adjoint Sampling 与 ASBS 对 source boundary 的保持。

### 第 9 周：按课题选择一个扩展

- trajectory / video：读 [3MSBM](./2506.10168_momentum_multi_marginal_sbm.md)。
- function / field：读 [FAS](./2511.06239_functional_adjoint_sampler.md)。
- token / discrete action：读 [DAM](./2602.07132_discrete_adjoint_matching.md) 与 [DASBS](./2602.08243_discrete_adjoint_schrodinger_bridge_sampler.md)。

### 第 10 周：做自己的最小研究原型

目标不是继续读论文，而是完成：

1. 明确 source、target、coupling 与 data regime；
2. 定义 task-aware path cost；
3. 至少两个简单基线；
4. downstream real-domain metric；
5. 一个能够证伪核心假设的 ablation。

---

## 11. 复现路线与代码入口

| 阶段 | 代码 | 建议先跑什么 | 预期收获 |
|---|---|---|---|
| 1 | [SB-FBSDE](https://github.com/ghliu/SB-FBSDE) | 低维 toy / Gaussian mixture | forward-backward 训练结构 |
| 2 | [DeepGSB](https://github.com/ghliu/DeepGSB) | obstacle / opinion toy | state cost 与 mean-field |
| 3 | [I²SB](https://github.com/NVlabs/I2SB) | 小型 paired restoration | conditional bridge 与采样 |
| 4 | [GSBM](https://github.com/facebookresearch/generalized-schrodinger-bridge-matching) | `example_CondSOC` | conditional SOC + matching |
| 5 | [Adjoint Sampling](https://github.com/facebookresearch/adjoint_sampling) | toy energy | energy-only sampler |
| 6 | [Adjoint Samplers](https://github.com/facebookresearch/adjoint_samplers) | ASBS toy | arbitrary source prior |
| 7 | [3MSBM](https://github.com/panostheo98/3MSBM) | sparse snapshots | phase-space multi-marginal |
| 对照 | [Mirror Diffusion](https://github.com/ghliu/mdm) | constrained toy | domain geometry |
| 应用 | [SBUnfold](https://github.com/ViniciusMikuni/SBUnfold) | synthetic detector setup | simulation-trained correction |

### 复现时统一记录

- commit、环境和数据版本；
- source / target 是否 paired、semi-paired 或 unpaired；
- reference process 与 noise schedule；
- SDE / ODE / CTMC solver、时间步数与 NFE；
- energy / reward evaluation 次数；
- 每批轨迹的 optimizer reuse 次数；
- marginal fit、path cost、mode coverage 和 downstream metric；
- 训练显存、总时间和采样吞吐。

---

## 12. 阅读陷阱与批判性问题

### 12.1 不要用“理论统一”代替实验选择

FBSDE、control、SB 与 adjoint 可以统一描述很多方法，但具体问题仍需回答：

- 目标是 sample generation、domain mapping，还是 trajectory optimization？
- 是否必须保持 source–target conditional identity？
- 是否有 paired data？
- target samples 和 target energy 哪个更可信？
- 中间路径是否真的有物理意义？

### 12.2 Adjoint 方法的 energy 是最大风险源

在分子系统中，energy 往往有明确物理含义；在 Sim2Real 视觉域中，“真实感”常由判别器或 embedding 近似，可能产生：

- adversarial shortcuts；
- mode collapse；
- task-irrelevant photorealism；
- 几何和动作语义破坏。

因此 energy 必须分解并单独验证，而不是只优化一个黑箱分数。

### 12.3 路径漂亮不等于条件映射正确

边际匹配优秀的模型可能错误配对实例。机器人数据尤其要检查：

- 同一动作是否保持相同结果语义；
- 对象 identity 与 pose 是否稳定；
- temporal ordering 是否保留；
- policy latent 是否漂移。

### 12.4 高维数据不一定应在原空间做 bridge

RGB 像素空间的欧氏路径可能没有任务含义。优先比较：

- pixel bridge；
- pretrained visual latent bridge；
- policy feature bridge；
- state–action joint bridge。

### 12.5 新会议状态要以最新官方来源为准

预印本、workshop 与正式主会版本可能跨年变化。引用时依次核验：

1. 作者最新 CV / 主页；
2. 正式 proceedings；
3. OpenReview venue；
4. arXiv 历史版本。

---

## 13. 作者当前向 LLM Post-training 的延伸

作者主页明确表示近期开始研究 post-training LLMs 中的 RL science。现阶段最清楚的技术连接是 DAM：

```text
unnormalized energy / reward
    + discrete CTMC sampler
    + adjoint matching
    → diffusion language model reward fine-tuning
    → mathematical reasoning
```

这不是对 SB 主线的放弃，而是把“只依赖 reward 的路径控制”迁移到 token space。

作者目前所在的 Meta Superintelligence Labs 也在公开推进 [Muse Spark](https://ai.meta.com/blog/introducing-muse-spark-meta-model-api/) 一类大模型 post-training 与 agentic reasoning 系统。应把这视为其当前团队环境和可能的研究方向信号；在没有公开作者列表或具体论文前，不应把 Muse Spark 的某项技术直接归因于个人。

建议关注三个后续问题：

1. adjoint matching 是否能成为比 policy-gradient 更稳定的 reward optimization 方法；
2. discrete path reference 如何影响推理探索、长度和 mode coverage；
3. energy-based sampling、RL 与 inference-time search 是否会形成统一框架。

---

## 14. 推荐资料清单

### 14.1 作者官方入口

- [个人主页](https://ghliu.github.io/)
- [完整 CV](https://ghliu.github.io/assets/pdf/cv.pdf)
- [Google Scholar](https://scholar.google.com/citations?user=2Dt0VJ4AAAAJ)
- [GitHub](https://github.com/ghliu)
- [Georgia Tech 博士论文](https://repository.gatech.edu/entities/publication/895044f4-a1e1-403b-8414-1f6aabacadbf)

### 14.2 最值得看的讲稿

- [Adjoint Schrödinger Bridge Sampler — NeurIPS slides](https://ghliu.github.io/assets/pdf/asbs_talk_neurips.pdf)
- [Adjoint Schrödinger Bridge Sampler — SlidesLive](https://slideslive.com/39055157/adjoint-schrodinger-bridge-sampler)
- [DeepGSB — NeurIPS slides](https://neurips.cc/media/neurips-2022/Slides/54873.pdf)
- [Generalized Schrödinger Bridge: From Generative Modeling to Mean-Field Games](https://slideslive.com/38993572/generalized-schrodinger-bridge-from-generative-modeling-to-meanfield-games)

### 14.3 本地配套阅读

- [完整学习资源与顶会前沿导航](./deep_research_learning_resources.md)
- [Adjoint / Generalized / Structured SB 跨论文总结](./sb_adjoint_extended_synthesis.md)
- [DeepGSB](./2209.09893_deep_generalized_schrodinger_bridge.md)
- [I²SB](./2302.05872_i2sb.md)
- [GSBM](./2310.02233_generalized_schrodinger_bridge_matching.md)
- [Adjoint Sampling](./2504.11713_adjoint_sampling.md)
- [ASBS](./2506.22565_adjoint_schrodinger_bridge_sampler.md)
- [3MSBM](./2506.10168_momentum_multi_marginal_sbm.md)
- [FAS](./2511.06239_functional_adjoint_sampler.md)
- [DAM](./2602.07132_discrete_adjoint_matching.md)
- [DASBS](./2602.08243_discrete_adjoint_schrodinger_bridge_sampler.md)
- [React-OT](./2404.13430_react_ot.md)
- [SBUnfold](./2308.12351_sb_unfold.md)

---

## 15. 最终学习建议

最不建议的做法是按年份把他的所有论文线性读完。更高效的方法是围绕三个问题建立自己的推导与实验：

1. **怎样从 path-space objective 推导出可训练的 control / score / adjoint？**
2. **当前数据提供了什么结构：paired、energy、prior、multi-marginal、discrete 还是 function-valued？**
3. **什么评价能证明 transport 保留了真正的任务语义？**

学习成果的合格标准不是“知道每个 acronym”，而是能够面对一个新问题，明确选择：

```text
OT 还是 SB？
sample matching 还是 energy sampling？
Brownian、momentum、annealed 还是 CTMC reference？
pixel、latent、state-action 还是 function space？
paired、semi-paired、unpaired 还是 energy-only？
```

如果能对这些选择给出可证伪的理由，就已经真正学到了这条研究线的核心。
