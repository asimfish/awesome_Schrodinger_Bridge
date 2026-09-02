# E14｜SOC 采样器源头：PIS / DDS / CMCD 与 Adjoint 线的谱系定位

> **选题定位**（来自 R09 缺口分析 G9/G8，扩充计划第 14 项）：库内 Adjoint 线（AS/ASBS/FAS/DASBS）的报告把 Adjoint Sampling 当作叙事起点，但 AS 论文本身是对 PIS/DDS 这类 SOC（stochastic optimal control）采样器可扩展性瓶颈的回应。缺了源头，"为什么 adjoint matching 是突破"讲不清楚，也无法回答审稿人"为什么不用现成 neural sampler 做 energy-guided transport"。本报告补齐三篇源头论文：**PIS 精读**，**DDS / CMCD 半精读**（CMCD 重点澄清其与 IPF/EM 的关系），并给出 PIS/DDS/CMCD → AS/ASBS 的**谱系定位图**。
>
> **阅读方式**：三篇均通过 arXiv abs 页的 HTML 全文获取成功（含 PIS 附录 A–G、DDS 主文+附录节选、CMCD 主文；CMCD 附录未逐节精读）。无"无法获取全文"情况。venue 已于 **2026-08-14** web 复核（见 §4）。

## TL;DR

1. **PIS（ICLR 2022）是"把采样写成 SOC/SB"的深度学习开山作**：Dirac 初态 + pinned Brownian motion 参考过程下，Schrödinger bridge 退化为一个单侧 half-bridge，可用反向 path KL（= 控制能量 + 终端能量代价）端到端训练；代价是**每次梯度更新都要模拟完整 SDE 并对轨迹反传**（Neural SDE adjoint / 缓存全图），这正是后来 Adjoint Matching 拆掉的瓶颈。
2. **DDS（ICLR 2023）换参考过程不换范式**：用平稳 OU（VP）参考替代 pinned BM，drift 不再陡峭、训练显著更稳，并修正了离散化对 ELBO 有效性的破坏（Euler–Maruyama 会导致 log Z 系统性高估）；但训练目标仍是 on-policy 反向 KL + 全轨迹反传，能量调用与梯度更新仍 1:1 锁死。
3. **CMCD（ICLR 2024）解决的是"唯一性/模式覆盖"轴而非"能量调用经济性"轴**：同时学前向+后向动力学，用退火路径 `π_t` 的分数约束恢复唯一解，等价于**无穷多个无穷小 Schrödinger 问题的拼接**；其概念贡献是证明 **EM ⟺ IPF**（完全灵活参数化 + 匹配初始化下二者逐迭代相同），这也正是后来 ASBS 用"matching 回归替换 IPF half-bridge"的理论铺垫。
4. **Adjoint 线解决的瓶颈可以精确命名**：(i) 反传瓶颈——lean adjoint 在零 base drift 下有解析解 `∇g(X_1)`，KL 泛函优化变成逐点回归；(ii) on-policy 耦合瓶颈——Reciprocal 投影 + replay buffer 把能量调用与梯度更新解耦（对比 PIS/DDS 的 1 能量调用 + ~10³ 网络前向/每次更新）；(iii) 先验瓶颈——ASBS 的 Corrector Matching 恢复任意先验的一般 SB，其收敛证明恰恰回到 IPF 两个 half-bridge 投影。
5. **未被 adjoint 线吸收的遗产**：PIS 的 path-integral importance weight / ESS / log Z 界（采样质量诊断），CMCD 的退火路径 pin 中间边缘（结构性防 mode collapse）。反向 KL 的 mode-seeking 在 AS/ASBS 里仍残留（ASBS 的 Ramachandran 图漏低密度模态），这是谱系上的开放缺口。

---

## 1. 精读：Path Integral Sampler（PIS）

### 1.1 元信息

- **论文**：Path Integral Sampler: A Stochastic Control Approach for Sampling
- **作者**：Qinsheng Zhang, Yongxin Chen（Georgia Tech；Chen 是库内 ASBS 的合作者、Guan-Horng Liu 的导师，见 `reports/guan_horng_liu_research_roadmap.md`）
- **venue**：**ICLR 2022 poster**（OpenReview `_uCb2ynRu7Y`，2026-08-14 核验；此前缺口分析中的"ICLR 2022?"确认无误）
- **链接**：https://arxiv.org/abs/2111.15141 ｜ 代码 https://github.com/qsh-zh/pis
- **理论前身**：Tzen & Raginsky (COLT 2019) 的 latent diffusion 生成模型—随机控制对应；Dai Pra (1991)、Pavon (1989) 的 SB 随机控制表述；Thijssen & Kappen (2015) 的 path integral control。

### 1.2 动机

从未归一化密度 `μ̂ = Zμ` 采样时，两大传统框架各有硬伤：VI 需要显式密度模型（normalizing flow 的双射 + 易算 Jacobian 约束限制表达力）；MCMC/SMC 混合慢、有限步行为难分析、易困在局部模态。PIS 的提议：**用一个学出来的 SDE 在有限时间 T 内把粒子从简单初始分布推到目标分布**，控制器是自由架构的神经网络，且可以用 path integral 理论对次优控制器做重要性加权，恢复无偏性。

### 1.3 方法核心

**SOC 形式化**。受控 SDE 为 `dx_t = u_t dt + dw_t`，`x_0 ~ ν`（实践中取 Dirac `δ_0`），代价函数

```
E[ ∫_0^T ½‖u_t‖² dt + Ψ(x_T) ]
```

即"控制能量 + 终端代价"。HJB 方程经对数变换（Hopf–Cole）线性化，由 Feynman–Kac 公式得值函数 `φ_t(x) = E_{Q⁰}[exp(−Ψ(x_T)) | x_t = x]`（`Q⁰` 为无控过程）——这就是 path integral control：最优值函数与最优控制原则上可用**无控轨迹**的重要性采样估计，但高维下方差爆炸，因此需要学习。

**与 SB 的关系（Theorem 1）**。在所有把 `ν` 送到 `μ` 的控制中选路径 KL 距无控过程最小者，即 Schrödinger bridge。当 `ν` 为 Dirac 且终端代价取 `Ψ(x_T) = log(μ⁰(x_T)/μ(x_T))`（`μ⁰` 为无控过程的终端边缘，pinned BM 时即 `N(0, T·I)`），最优控制诱导的路径测度恰为 `Q*(τ) = Q⁰(τ|x_T)μ(x_T)`，终端边缘 = `μ`。注意这里的结构性简化：**Dirac 初态使 SB 的一端边界自动满足，问题退化为单侧 half-bridge，一次反向 KL 最小化即可解**，无需 IPF 迭代——这是 PIS 能"端到端一次训练"的根本原因，也是它被 memoryless 条件锁死的根源（对照 §3 的 ASBS）。

**训练目标**。由 Girsanov 定理，`D_KL(Q^u‖Q⁰) = E[∫½‖u‖²dt]`，于是反向 path KL

```
u* = argmin_u E_{Q^u}[ ∫_0^T ½‖u_t‖² dt + log(μ⁰(x_T)/μ̂(x_T)) ]
```

对未归一化 `μ̂` 只差常数 `log Z`，不影响最优解；该式同时是学习质量的评估度量（= `D_KL(Q^u‖Q*) − log Z`）。

**梯度估计方式（可扩展性瓶颈所在）**。目标是**当前控制诱导的路径测度上的期望**（on-policy），梯度必须穿过整条模拟轨迹：论文用 Neural SDE 的 stochastic adjoint sensitivity（torchsde；反传本身是又一条伴随 SDE），或在内存允许时缓存全图直接 BPTT，并把 running cost 增广为额外状态维度使训练端到端。三个直接后果：

1. **每次参数更新 = 1 次完整前向模拟（K 步网络前向，默认 K=100）+ 1 次等价代价的反向传播**；轨迹用完即弃（on-policy），能量调用与梯度更新 1:1 锁死。
2. **网络规模受限**：反传内存/时间随 K 与网络深度增长（AS 论文在 LJ 实验中明确指出 PIS/DDS 只能用较浅的 EGNN 才能让"反传穿过 SDE"可行）。
3. **数值稳定性问题**：PIS 附录 G.1 自己报告 torchsde 的 adjoint/Reversible Heun 求解器"比不用 adjoint 的朴素 Euler 更不稳、损失更高"，只在内存撑不住时才被迫使用——即当时最好的轨迹反传工具本身就是不可靠组件。

**Gradient-informed 参数化（PIS-Grad）**。`u_t(x) = NN₁(t,x) + NN₂(t)·∇log μ̂(x)`，即把目标分数直接注入 drift（可视为被网络调制的 Langevin 动力学）。多模态目标上显著优于纯网络的 PIS-NN——但代价是**每个模拟步都要调一次能量梯度**（K 步 → 每条轨迹 O(K) 次 `∇E` 调用），LGCP 上还需要对能量梯度做裁剪才能稳定。这是"能量调用次数"维度上经常被忽略的一笔账。

**无偏化与诊断（PIS 的独特遗产）**。对任意次优控制 `u`，path integral 给出重要性权重

```
w^u(τ) = exp( −∫ ½‖u‖²dt − ∫ u′dw − Ψ(x_T) )
```

由此得到：无偏 `Z` 估计 `Z = E[exp(−Ŝ^u(τ))]`（Theorem 4，含 ELBO 下界）；ESS 下界 `1/E[(w^u)²] ≥ 1−ε`（Theorem 3，`ε` 为控制误差）；以及 **W₂ 采样质量界**（Theorem 2/5）：控制误差 `‖u−u*‖² ≤ dε`、步长 `Δt` 时 `W₂(Q^u(x_T), μ) = O(√(Td(Δt+ε)))`。这三件套让 PIS 成为少数"次优也能定量自检"的学习采样器。

### 1.4 实验

- **基准**：mode-separated 9-Gaussian（d=2，方差 0.3 故意压小）、Funnel（d=10）、LGCP（d=1600）上估计 log Z；对比 HMC/NUTS、SMC、AFT、VI-NF（以及附录中的 SVGD）。PISRW-Grad（带 IW）全面最优，如 LGCP 上 A=2.14 vs AFT 3.46 / SMC 436 / HMC ~1300；100 步的 PIS 可与 10⁴ 步的 AFT 相当。
- **Alanine dipeptide**（d=132，OpenMM 真空模拟设定目标能量）：与 VI-NF/SMC/SNF 比 KL，PIS-NN（公平起见不用梯度）总体最低。
- **VAE latent 后验**（d=50，binary MNIST）：估 `log p(x)`，PISRW-Grad 偏差与方差最小。
- **效率**：采样比 SMC/AFT 快约一个数量级（2k 粒子 16.8–34.3 ms）；训练开销"一次训练、到处采样"可摊销，但 LGCP 每 epoch 7–9 分钟（A6000）。

**规模上限的后视镜证据**（引自库内已有材料，非本文实验）：AS 论文（`texts/2504.11713_adjoint_sampling.txt` Table 1）测得 PIS 在 LJ-13 已开始吃力（path-ESS 0.012）、LJ-55（d=165）失败（path-ESS 0.001，能量 W₂ 228.70±131.27），且每次梯度更新耗 1 次能量调用 + 约 10³ 次网络前向。**PIS 的失效不是理论错误，而是"轨迹泛函优化"这一实现方式在能量昂贵、维度更高、网络更大时的经济性崩溃。**

### 1.5 局限

- **每目标重训**，训练开销不可忽略（对比免训练 MCMC）；采样质量完全取决于控制器优化程度。
- **反向 KL zero-forcing**：表达力不足或初始化不当时丢模态（论文 Fig 2 与附录 G.2 的失败模式：T 太小 → 初始 `N(0,TI)` 覆盖不到目标模态 + 最优控制 Lipschitz 常数大 → 训练失败；T 大 + Δt 粗 → 离散化误差劣化样本）。
- **超参敏感**：LGCP 需扫 T∈{1,2,5} 才避开 NaN；能量梯度需裁剪。
- **先验被锁死**：一般参考动力学 `f,g` 需保证 `μ⁰` 闭式可算，论文承认"没有一般方法"，只用 `f=0, g=I`；Dirac 初态 + pinned BM 的组合后来被 ASBS 明确列为"非 memoryless、任意先验"四象限中最受限的一格。
- **重要性权重是补丁不是解药**：高维下 ESS 仍会塌缩（AS 表中 LJ-55 的 path-ESS 0.001 即 IW 已失效的信号）。

---

## 2. 半精读：Denoising Diffusion Samplers（DDS）

### 2.1 元信息

- **论文**：Denoising Diffusion Samplers ｜ Francisco Vargas, Will Grathwohl, Arnaud Doucet（Cambridge / DeepMind）
- **venue**：**ICLR 2023 poster**（OpenReview `8pvnfTAbu1f`、iclr.cc/virtual/2023/poster/10913，2026-08-14 核验）。**注意：任务单中"已核验 ICLR 2024"系记录混淆，需修正**——与 DDS 密切相关且发表于 ICLR 2024 的是 Richter & Berner 的 log-variance 统一框架（arXiv 2307.01198，见 §5.2 节点 3）和 CMCD；DDS 本体是 ICLR 2023。
- **链接**：https://arxiv.org/abs/2302.13834 ｜ 代码 https://github.com/franciscovargas/denoising_diffusion_samplers

### 2.2 核心思想

把 DDPM 的"正向加噪 + 学时间反演"搬到采样问题：正向 OU/VP 扩散 `dx_t = −β_t x_t dt + σ√(2β_t) dB_t` 把目标 `π` 推向 `N(0,σ²I)`，采样器 = 学它的时间反演。**关键障碍：score matching 不可用**（无法从 `π` 采样正向轨迹），于是回到反向 path KL——与 PIS 同一范式，但有两处结构改进：

1. **平稳参考过程**。引入同动力学但初始化于平稳分布 `N(0,σ²I)` 的参考测度 `P^ref`（所有时刻边缘均为 `N(0,σ²I)`），并把网络重参数化为 `f_θ = s_θ + x/σ²`，其含义是值函数梯度 `∇log φ_t`（`φ_t = p_t/p_t^ref`）。KL 化为与 PIS 同构的"控制能量 + 终端代价"：`KL = E[σ²∫β‖f_θ‖²dt + log(N(y_T)/π(y_T))]`。相比 pinned BM，OU 参考的时间反演 drift 不陡峭（PIS 的 pinned 参考在端点附近 drift 奇异），**训练对超参显著更稳**（论文 Fig 1 做了逐超参对照）；顺带修正了 PIS 原文 Funnel 实验对 PIS 有利的 `σ_f` 设定失误。
2. **ELBO 保持的离散化**。命题 2：若参考过程的离散化不保持平稳性（如 Euler–Maruyama），离散 RND 不再给出合法 ELBO，log Z 估计会**系统性高估**（实验确认）。解法：OU 精确积分 + 受控过程的指数型积分器，保证 `p_k^ref` 逐步不变。这个"离散化正确性"问题是 PIS 一笔带过而 DDS 首次讲清楚的。

**训练与梯度**：JAX reparameterization trick 对整条 K 步轨迹反传；论文明确指出**无法像 DDPM 那样对时间下标做 minibatch**（`q^θ_{k|K}` 无闭式，采样任意中间时刻必须模拟到该时刻）——这句话就是"simulation-based 训练"瓶颈的官方表述，与后来 AS 用 reciprocal projection 解析采样 `(X_t, X_1)` 形成精确对照。网络沿用 PIS-Grad 参数化（含每步 `∇log π` 调用），detach 目标分数以稳定训练。

**SB 视角**（论文 §3.4，对谱系图重要）：DDS ≈ 求解边界 `(π, N(0,σ²I))`、OU 参考的 SB，但只近似做了 IPF 的第一步半桥（`p^sb ≈ p³ ≈ p²`，K 大时成立）；而 PIS 对应的 Dirac 终端 SB 中 `p^sb = p²` **精确成立**——两者都是"单个 half-bridge 就够"的特例。这为 §3 的 IPF 叙事埋下伏笔：一旦两端都是非退化分布，单次反向 KL 不再够，必须迭代（IPF）或加约束（CMCD）或换回归（ASBS）。

### 2.3 实验与局限（半精读要点）

- 基准：Funnel、LGCP(d=1600)、Bayesian logistic（Ionosphere d=32 / Sonar d=61）、Brownian motion 时序（d=32）、NICE 流模型目标（d=196，用 Sinkhorn 距离检测 mode collapse）。DDS 比 PIS 稳、与精调 SMC 竞争；NICE 多模态目标上 log Z 估计 −3.20 优于 PIS −3.93 / SMC −4.26 / MCD −6.25。
- 概率流 ODE normalizing-flow 变体与 underdamped 变体在高维**均令人失望**（作者自述）。
- 局限：训练时间不可忽略（简单目标上算总账 SMC 更好）；反向 KL 的 mode dropping 承认存在；**范式级瓶颈与 PIS 完全相同**——on-policy 全轨迹模拟 + 反传，每次更新 1 次能量调用 + O(K) 网络前向（AS Table 1 中 DDS 与 PIS 同列：1 / ~10³），LJ-55 同样失败（能量 W₂ 173.09±18.01）。

---

## 3. 半精读：Controlled Monte Carlo Diffusions（CMCD），重点：与 IPF/EM 的关系

### 3.1 元信息

- **论文**：Transport meets Variational Inference: Controlled Monte Carlo Diffusions ｜ Francisco Vargas, Shreyas Padhy（Cambridge）, Denis Blessing（KIT）, Nikolas Nüsken（KCL）
- **venue**：**ICLR 2024 poster**（OpenReview `PP1rudnxiW`，proceedings.iclr.cc 2024 收录，2026-08-14 核验，与任务单一致）。camera-ready 摘要较 arXiv v1 增加了一句："deriving as well a regularised objective that **bypasses the iterative bottleneck of standard IPF-updates**"（对应附录 E.5 的正则化 IPF 目标）。
- **链接**：https://arxiv.org/abs/2307.01050 ｜ 代码 https://github.com/shreyaspadhy/CMCD

### 3.2 统一框架：path space 上的散度

从 hierarchical VAE 的无限深度极限出发，得到一对前向/后向 SDE：`dY = a_t dt + σ d→W`（`Y_0~μ`）与 `dY = b_t dt + σ d←W`（`Y_T~ν`），及其广义 Girsanov 定理（Prop 2.2，带自由选择的参考过程）。**Framework 1′**：最小化任意 path-space 散度 `D(→P^{μ,a} ‖ ←P^{ν,b})`，为零时前向把 `μ` 运到 `ν`、后向反之。已有方法全是它的特例：固定前向 drift → score-based 生成模型；固定后向为 ergodic 过程 → **DDS/DIS**；取 `b_t = x/t`（Föllmer drift）→ **PIS**；特定参考选择 → action matching。

**核心张力**：若 `a, b` 都完全自由，零点集 = 所有 `(μ,ν)` 耦合，**极不唯一**，训练不稳、结果不可解释。已有方法靠"冻结一端"回避；CMCD 的两个贡献分别对应两种更有原则的解法。

### 3.3 澄清 EM ⟺ IPF（本次任务的重点）

**IPF**：求解 Schrödinger 问题（静态式 (16)/动态式 (17)）的经典途径是交替 KL 投影——奇数步固定 `x`-边缘为 `μ`，偶数步固定 `z`-边缘为 `ν`，从参考 `r(x,z)` 出发收敛到 `π*`。

**EM**：Neal & Hinton 表述下，EM 是对 `L_KL(φ,θ) = D_KL(q^φ(z|x)μ(x) ‖ p^θ(x|z)ν(z))` 的**坐标交替最小化**（先 θ 后 φ）。

**Proposition 3.1（EM ⟺ IPF）**：若 `p^θ, q^φ` 参数化**完全灵活**，且 EM 初始化满足 `q^{φ₀}(z|x) = r(z|x)`，则 EM 迭代与 IPF 迭代**逐步完全一致**：`π^n = q^{φ_{(n−1)/2}}μ`（n 奇），`π^n = p^{θ_{n/2}}ν`（n 偶）。三条推论对理解谱系至关重要：

1. **VI 与 OT 是同一个交替投影的两种语言**：IPF 的 half-bridge = EM 的坐标更新。对经典 VAE（受限参数化）该对应失效；对 SDE 参数化"误差可忽略地"成立。
2. **散度方向在理论上可换**：把 (18a)/(18b) 中的 forward-KL 换成 reverse-KL 不改变最小化子序列——这解释了为什么文献里 IPF 的两个 half-bridge 可以各自用最方便的方向实现（也是 DSB/DSBM 系与 VI 系能互引的原因）。实践上 forward-KL 偏 moment-matching、reverse-KL 偏 mode-seeking，交替反而可能是折中。
3. **共同病灶**：EM 与 IPF 都是**串行求解一列只能近似解的子问题**——慢，且每次迭代的近似误差沿迭代**累积**（引 Vargas et al. 2021、Fernandes et al. 2021 "Shooting Schrödinger's cat"对 IPF 误差累积的分析）。这是 CMCD 提出"端到端、同时更新两端"的直接动机；camera-ready 补充的正则化目标（附录 E.5）就是绕开 IPF 迭代瓶颈的另一实现。

**对库内叙事的意义**：ASBS 的收敛证明（AM 与 Corrector Matching 交替 = IPF 两个 half-bridge 投影，见 `reports/2506.22565_adjoint_schrodinger_bridge_sampler.md`）正是站在这一"交替投影"传统上；CMCD 的 Prop 3.1 提供了把这一传统同时读作 VI（EM）与 OT（IPF）的字典。**谱系上应把 CMCD 记为"EM/IPF 等价性的澄清者 + 端到端替代方案的首批探索者"，把 ASBS 记为"把 IPF half-bridge 从模拟+反传的 KL 最小化替换成可扩展 matching 回归的完成者"。**

### 3.4 CMCD 采样器本体

固定退火曲线 `π_t`（如几何插值，要求每个 `t` 的未归一化密度与分数可算——AIS 的标准设定），学习控制 `∇φ_t`：

```
dY_t = (σ²∇log π_t(Y_t) + ∇φ_t(Y_t)) dt + σ√2 d→W_t,   Y_0 ~ π_0
```

后向 drift 由 Nelson 关系**约束**为 `b = −σ²∇log π_t + ∇φ_t`、终端 `ν = π_T`，于是 Framework 1′ 的目标 `L^CMCD(φ) = D(→P ‖ ←P)` 恢复**唯一最优解**（Prop 3.2），最优时前向边缘逐点等于 `π_t`。理解锚点：

- `φ=0` 退化为 Unadjusted Langevin Annealing（ULA）；只在分子取 `φ=0` 得 MCD——**CMCD = 受控版退火 Langevin**。
- **Prop 3.4**：CMCD = 把 `[0,T]` 切成 N 段、每段解一个以 `∇log π_t` 为先验 drift 的 Schrödinger 问题再拼接，`N→∞` 的极限——即 Bernton et al. 2019 Sequential SB 采样器的无穷小极限（"jointly solving an infinite number of Schrödinger problems"）。
- **统计物理底座**：受控 Crooks 恒等式与 Jarzynski 等式的推广；控制项通过交互项 `C^φ_T` 压制功涨落，最优时 log Z 估计**零方差**。
- 散度可选：`D_KL` 版（式 24，需模拟 + 反传）与 log-variance 版（off-policy，可 detach 掉对模拟的求导；与 Richter & Berner 2024 同期呼应）。

**实验**（半精读要点）：复刻 Geffner & Domke 2023 的 6 个基准（对比 ULA/MCD/UHA/LDVI），K∈{8,…,256} 全档 ELBO 领先，尤其低 K；funnel/GMM 的 log Z 对比 PIS/DDS/SMC 更稳。**注意实验规模仍是 d≲10³ 的贝叶斯/合成目标**，没有分子级或摊销设定。

### 3.5 CMCD 的谱系位置与局限

- CMCD 解决的是**唯一性与模式覆盖**：退火路径把中间边缘 pin 在 `π_t` 上，是结构性防 mode collapse 的设计（对照 PIS/DDS 只约束两端）。
- CMCD **没有**解决能量调用经济性：base drift 每步都要 `∇log π_t`（O(K) 能量梯度/轨迹），KL 版还要全轨迹反传；它也依赖人工指定的退火曲线（作者列为 future work）。
- ASBS 论文中 CMCD 以"SCLD 的退化情形"出现在基线体系里（SCLD 不切 subtrajectory 时退化为 CMCD，见 `texts/2506.22565_*.txt`），说明 adjoint 线视其为需要超越的代表性 VI 采样器之一。

---

## 4. venue 复核汇总（检索日期 2026-08-14）

| 论文 | arXiv | 正式 venue | 证据 | 核验结论 |
|---|---|---|---|---|
| PIS | 2111.15141 | **ICLR 2022 poster** | OpenReview `_uCb2ynRu7Y`（页面标注 "ICLR 2022 Poster"）；官方 repo bibtex | 确认，缺口分析中"ICLR 2022?"去掉问号 |
| DDS | 2302.13834 | **ICLR 2023 poster** | OpenReview `8pvnfTAbu1f`（"ICLR 2023 poster"）；iclr.cc virtual/2023/poster/10913；官方 repo bibtex | 确认；**任务单中"已核验 ICLR 2024"为记录混淆，应更正为 ICLR 2023**（ICLR 2024 的是其后续 log-variance 框架与 CMCD） |
| CMCD | 2307.01050 | **ICLR 2024 poster** | OpenReview `PP1rudnxiW`；proceedings.iclr.cc 2024 论文页；KCL Pure 收录页；官方 repo bibtex | 确认，与 R09 既有核验一致 |
| （谱系节点）DIS | 2211.01364 | **TMLR 2024** | OpenReview `oYIjw37pTP`；tmlr.infinite-conf.org 论文页 | 新核验（此前库内无记录） |
| （谱系节点）log-variance 统一框架 | 2307.01198 | **ICLR 2024 poster** | OpenReview `h4pNROsO06`；proceedings.iclr.cc 2024 | 新核验（即 AS 表中 "LogVariance"/LV-PIS 的正式出处；R09 曾以 Richter & Berner 2024 指称） |

Adjoint Matching（2405.13731，ICLR 2025 Spotlight）与 AS（ICML 2025）/ASBS（NeurIPS 2025）的 venue 沿用 R09 于 2026-08-14 的既有核验，本次未重复检索。

---

## 5. SOC 采样谱系定位图：PIS/DDS/CMCD → AS/ASBS

### 5.1 四维对照表

约定：K = SDE 离散步数（典型 10²–10³）；"能量调用"按**每条轨迹每次梯度更新**计，`E` 为能量评估、`∇E` 为能量梯度；"轨迹反传" = 梯度是否必须穿过 SDE 模拟器（BPTT 或 adjoint SDE）。

| 方法（venue） | 训练目标 | 需要模拟完整轨迹并反传？ | 能量调用次数 | 可扩展性（实证边界） |
|---|---|---|---|---|
| **PIS**（ICLR 2022） | 反向 path KL：`E_{Q^u}[∫½‖u‖² + Ψ(x_T)]`，on-policy，Dirac 初态 + pinned BM 参考 | **是**（Neural SDE stochastic adjoint 或缓存全图；附录自报 adjoint 求解器不稳） | 1×`E`（终端）；PIS-Grad 另加 K×`∇E`（drift 内） | d≲10³ 合成/贝叶斯任务 SOTA（2022 年）；LJ-13 勉强、LJ-55 失败（path-ESS 0.001）；网络深度受反传限制 |
| **DDS**（ICLR 2023） | 同构反向 KL，但平稳 OU 参考 + ELBO 保持积分器；`f_θ ≈ ∇log φ_t` | **是**（JAX reparam 全轨迹；明确无法对时间下标 minibatch） | 同 PIS（1×`E` + K×`∇E`，PIS-Grad 网络） | 比 PIS 稳（drift 不陡峭）；NICE d=196 多模态优于精调 SMC；LJ-55 同样失败 |
| **DIS / LV**（TMLR 2024 / ICLR 2024，谱系节点） | 同框架换散度：log-variance divergence，off-policy，**免 SDE 反传、免目标梯度** | 需模拟完整轨迹算 RND，但**不反传**（detach 参考测度） | 1×`E`（RND 内） | 免反传≠可扩展：AS 表中 LogVariance 在 DW-4 之外即不可用（方差主导）；说明瓶颈不止反传一项 |
| **CMCD**（ICLR 2024） | 前后向同时学 + 退火路径 `π_t` 分数约束恢复唯一性；KL 版或 LV 版 | KL 版**是**；LV 版免反传但仍需全轨迹 | K×`∇E`（base drift 每步退火分数）+ 端点 `E` | 低步数（K=8–128）下 ELBO 全面领先 ULA/MCD/UHA/LDVI；实验止于 d≲10³ 贝叶斯基准，无分子/摊销设定 |
| **AM**（ICLR 2025 Spotlight；库内缺条目，见并入建议） | 把 memoryless SOC 解写成 **lean adjoint 回归**的自洽固定点（原生场景：diffusion reward 微调） | **否**（回归目标由 lean adjoint 终端条件给出；采样只为收集数据，不求导） | 取决于任务的 reward/能量结构 | 使"SOC 解 = matching 损失"成为可能，是 AS/ASBS 的直接理论供体 |
| **AS**（ICML 2025） | Reciprocal Adjoint Matching：回归 `u(X_t,t) ≈ −σ(t)∇E(X_1)`，`(X_t,X_1)` 由解析后验重采样 + replay buffer | **否**（零 base drift 下 lean adjoint 解析 = `∇g(X_1)`；无任何穿模拟器梯度） | **与更新解耦**：1 次 `∇E` 支持多次更新（对 iDEM 约省 10⁵ 倍能量调用；AS 表：iDEM 512×`E`/更新 vs AS ≪1） | LJ-55 能量 W₂ 30.83（PIS 228.7/DDS 173.1）；首个摊销到 SPICE/GEOM-DRUGS 级构象生成的 on-policy 采样器 |
| **ASBS**（NeurIPS 2025） | AM（控制器）+ Corrector Matching（校正项）交替 = **IPF 两个 half-bridge 的 matching 化**；任意先验一般 SB | **否**（两个都是回归） | 同 AS 量级 + corrector 网络小幅开销 | 全面优于 AS（LJ-55 W₂ 4.00/28.10）；harmonic prior 免 RDKit warm-start 接近 warm-start AS |

（iDEM/NETS 等 simulation-free 竞品属扩充计划第 15 项 E15 的范围，此处仅作对照点引用，不展开。）

### 5.2 演进叙事：四个瓶颈如何被逐个拆除

**节点 0（2019–2021，理论铺垫）**：Tzen & Raginsky 证明 latent diffusion 生成 = 随机控制问题；Dai Pra/Pavon/Föllmer 的 SB–SOC 对应提供了"终端能量代价 ⇒ 目标分布"的机制。此时没有可扩展的数值实现。

**节点 1：PIS（ICLR 2022）——范式确立，瓶颈同时确立**。PIS 证明"参数化控制 drift + 反向 path KL"在真实任务上可行，并配齐 IW/ESS/W₂ 三件套诊断。但其实现把三个耦合写死：梯度穿模拟器（BPTT/adjoint SDE）、on-policy（轨迹一次性消耗）、Dirac+pinned BM（先验锁死）。**"SOC 采样开山"与"SOC 采样瓶颈开山"是同一篇论文。**

**节点 2：DDS（ICLR 2023）——参考过程工程学**。DDS 表明范式内还有免费午餐：换平稳 OU 参考（drift 不陡、训练稳）、修 ELBO 破坏（积分器保平稳性）。但它同时把范式的硬边界说破：无法对时间 minibatch、必须整轨迹模拟反传——**参考过程可以优化，训练目标的"轨迹泛函"本性无法在此范式内绕开**。

**节点 3（旁支）：DIS/log-variance（TMLR 2024 / ICLR 2024）——换散度自救，且证伪了一个流行猜想**。log-variance divergence 免 SDE 反传、免目标梯度、最优点梯度零方差，看似解决了瓶颈 1；但 AS 的对照显示 LogVariance 连 DW-4 之外都难扩展。**这说明真正的瓶颈是"每次更新消耗一条完整新轨迹 + 高维下泛函估计方差"的复合体，单独去掉反传不够。**这一负结果是 adjoint 线价值的最有力反衬。

**节点 4：CMCD（ICLR 2024）——唯一性轴的解，与 IPF/EM 的字典**。CMCD 松开"冻结一端"的约定（前后向都学），用退火路径分数约束恢复唯一性，等价于无穷小 SB 拼接；并证明 EM ⟺ IPF、散度方向理论可换、二者共享"串行近似误差累积"病灶。**CMCD 给出了后来者的两条路标**：(a) 要逃离串行 IPF，就要端到端目标或更好的 half-bridge 求解器；(b) VI 语言（EM/回归/ELBO）与 OT 语言（IPF/half-bridge/SB）可以自由换算。但 CMCD 自身在能量经济性轴上无进展（每步退火分数 + KL 版反传）。

**节点 5：Adjoint Matching（ICLR 2025 Spotlight）——瓶颈 1 的正解**。在 memoryless 设定下，SOC 最优性条件被改写为 lean adjoint 回归的自洽固定点：损失是逐点 L2，回归目标来自 lean adjoint ODE 的终端条件而非穿模拟器的梯度。与 PDDS/TSM 形式相似但期望测度关键不同——**AM 的期望取自当前控制（可行采样、无偏、无 IW），PDDS/TSM 取自最优控制（需要目标样本/SMC/IW）**。

**节点 6：AS(ICML 2025)——瓶颈 1+2 同时拆除**。采样特化（零 base drift）使 lean adjoint 有解析解 `∇g(X_1)`，回归目标只依赖 `(X_t, X_1)`；reciprocal projection 让 `X_t|X_1` 可从基过程后验解析重采样 → **replay buffer 合法化**，能量调用与梯度更新彻底解耦。PIS/DDS 的"1 更新 = 1 轨迹 + 1 能量调用"变成"1 批能量调用 = 任意多次更新"。这是"为什么 adjoint matching 是突破"的完整答案：**不是更好的散度，而是把轨迹泛函优化改写成带解析目标的逐点回归，从而同时消灭反传与 on-policy 消耗。**

**节点 7：ASBS（NeurIPS 2025）——瓶颈 3（先验）拆除，并回到 IPF**。AS 继承了 AM 的 memoryless 条件（Dirac 初态）。ASBS 用 Corrector Matching 补上任意先验的自由度，其收敛证明把"AM + Corrector 交替"解读为 IPF 的两个 half-bridge 投影——**与 CMCD 的 EM⟺IPF 澄清首尾呼应：谱系从 PIS 的"单 half-bridge 特例"出发，绕经 CMCD 的"IPF 全景"，最终由 ASBS 以"matching 化的 IPF"收束**。四象限（非 memoryless / 任意先验 / matching 目标 / 免 IW）由 ASBS 首次全部打勾（ASBS Table 1，库内 `texts/2506.22565_*.txt`）。

**仍开放的轴**：(i) mode-seeking——AS/ASBS 仍是反向 KL/SOC 家族，ASBS 在 alanine dipeptide 上仍漏低密度模态；CMCD 的退火路径 pin 边缘是现成的结构性缓解，尚未与 adjoint 回归结合（"annealed adjoint sampling"是自然的空白点）。(ii) 诊断——PIS 的 IW/ESS/log Z 三件套在 adjoint 线中没有对应物（AS/ASBS 免 IW 训练是优点，但推断期缺质量自检）。(iii) 能量不可微/含噪时 `∇E` 回归目标失效（AS 已知局限）。

### 5.3 对 SB-Render-Lite 的含义（一段话）

SB-Render-Lite 若走"能量式目标"路线（真实感 + 任务一致性能量而非成对/无配对样本），采样器选型的决定性维度就是本报告的第 3、4 列：能量（判别器/VLM/物理一致性模型）每次调用昂贵时，PIS/DDS/CMCD 范式的"1 更新 1 轨迹"不可承受，AS/ASBS 是唯一经济的 on-policy 选项，且 ASBS 允许直接拿 sim 分布当 source prior（呼应库内 ASBS 报告的建议）。反之，若担心渲染风格的多模态覆盖（同一 sim 场景对应多种合法真实观感），CMCD 的退火路径约束值得作为 bridge 中间边缘正则的设计参考。PIS 的 ESS/log Z 诊断可移植为 bridge 质量的训练外指标。

---

## 6. 并入主库建议

1. **新增 3 个正式条目**：本报告的 §1（PIS，精读级）、§2（DDS，半精读级）、§3（CMCD，半精读级）可直接拆分为 `reports/2111.15141_path_integral_sampler.md`、`reports/2302.13834_denoising_diffusion_samplers.md`、`reports/2307.01050_cmcd_transport_meets_vi.md`，并在 `INDEX.md` 新开"Neural Sampler 源头（SOC 采样）"小节，置于"Adjoint Sampler 方法线"之前。
2. **修正一处 venue 记录**：任何引用 DDS 的地方统一为 **ICLR 2023**（本次任务单中"已核验 ICLR 2024"为混淆；R09 的 G8/G9 表记录 ICLR 2023 正确，予以保留）。同时建议把 DIS（TMLR 2024）与 log-variance 框架（ICLR 2024，arXiv 2307.01198）作为导航条目补入，标注本报告的核验日期。
3. **更新两篇 Adjoint 线报告的口径**（不改文件，由维护者定夺）：`2504.11713_adjoint_sampling.md` 的"解决的问题"与"与后续系列的关系"两节可加一句上游指针——AS 针对的"每次更新重新模拟 + 反传"瓶颈的原型即 PIS（含其附录 G 自报的 adjoint 求解器不稳定），而"免反传但仍不可扩展"的 LV 旁支证明了 reciprocal + replay buffer 的必要性；`2506.22565_*.md` 的 IPF 段可引 CMCD Prop 3.1（EM⟺IPF）作为其交替投影证明的概念前史。
4. **谱系图归档**：§5 的四维对照表 + 演进叙事建议并入 `sb_adjoint_extended_synthesis.md` 作为"上游谱系"一章，或与 E15（iDEM/NETS simulation-free 竞品线）的对照表合并成一页"采样器全景图"——两报告的分工边界：本报告覆盖 SOC/on-policy 支，E15 覆盖 simulation-free/off-policy 支，合并时以"能量调用经济性 × 无偏性"二维坐标呈现。
5. **候选研究缺口登记**（供 idea 池）：(a) annealed adjoint sampling——把 CMCD 的退火边缘约束与 AS 的 reciprocal 回归结合，针对多模态能量；(b) 给 adjoint 线补 PIS 式推断期诊断（IW/ESS 的 matching 版类似物）；(c) SB-Render-Lite 实验设计中把"能量调用预算"作为与样本效率并列的报告指标（沿用 AS Table 1 的 `# E(·) evals per gradient update` 口径）。

---

*报告：E14（文献扩充研究员）｜日期：2026-08-14｜精读 1 篇（PIS）+ 半精读 2 篇（DDS、CMCD）+ 谱系节点 venue 核验 2 篇（DIS、LV 框架）｜全文获取方式：arXiv abs HTML（三篇均成功）｜遵守约束：未修改任何现有文件，仅新建本文件。*
