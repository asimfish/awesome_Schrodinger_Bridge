# E04：Flow Matching 精读 + minibatch coupling 设计笔记（unpaired sim↔real）

> 扩充研究员：E04 ｜ 日期：2026-08-14
> 选题来源：内部审查 R09 缺口 G2 / 选题 T4——「Flow Matching + minibatch coupling 设计」。
> 范围：精读 Flow Matching（arXiv 2210.02747）方法主体；提炼 OT-CFM（arXiv 2302.00482）与 Multisample Flow Matching（arXiv 2304.14772）关于 minibatch coupling 选择的结论；落点是 `SB-Render-Lite` 的 unpaired sim↔real 图像 coupling 设计。
> 全文获取方式：三篇均于 2026-08-14 通过 arXiv HTML 全文（abs 页渲染版）读取方法与实验章节，另以 PMLR/OpenReview/ICLR 官方页复核 venue；无仅凭 abstract 撰写的部分。
> 与库内 25 篇精读不重复；SB Flow、GSBM、I²SB、EgoBridge、Guided OT Co-Training 等仅作引用衔接。

## 选题定位

库内三大支柱（OT-IL、SB 生成、Adjoint 采样）都默认读者已掌握 flow matching 语言，但库中此前没有任何 FM 一手精读：`sb_adjoint_extended_synthesis.md` 第一阶段实验设计里写了「deterministic 对照：OT-ODE / flow matching」，却无可引用的报告；更关键的是，**unpaired sim-real 数据在 bridge/flow 训练中如何构造配对（coupling）**是 SB-Render-Lite 的工程核心，而 minibatch coupling 的系统结论正在 OT-CFM 与 Multisample FM 这两篇里。本报告补上这两块地基。

## TL;DR

1. **FM/CFM 是「条件回归解 marginal 场」的定义性论文**：把不可算的 marginal 向量场回归拆成逐样本条件向量场回归，二者梯度严格相等（Theorem 2）；它把 denoising score matching 的技巧从 score 推广到任意 Gaussian 概率路径的向量场，且在扩散路径上比 score matching 更稳、conditional OT 直线路径上训练/采样效率再上一档。局限：源分布必须是 Gaussian——不能直接做 sim→real。
2. **coupling 选择的净结论**：independent coupling（I-CFM/RF/SI 式）保证边缘正确但路径弯、目标方差在收敛点不归零；minibatch OT coupling 使直线度、目标方差、传输代价三者随 coupling 批量 `k→∞` 全部趋优（MFM Theorem 4.2），实验上换来 30–60% 的 NFE 节省与更快收敛，而最终 FID/NLL 不退化；**边缘分布对任意 coupling 都严格保持**（MFM Lemma 4.1），这是它相对「静态 barycentric 拟合 batch OT」的本质优势。
3. **对 unpaired sim↔real 的操作建议**：在 frozen encoder latent（而非像素）上算 cost；cost 用「视觉特征 + 任务/动作特征」的加权平方欧氏；coupling 批量 64–256 即可（低维 NPE 在 k≈64 饱和、ImageNet 级 per-GPU 50–256 且跨 GPU 无收益）；entropic 版本的 `ε` 是 OT-CFM（ε→0）与 I-CFM（ε→∞）之间的插值旋钮，`ε=2σ²` 时恰对应 SB——这条「ε 谱」把 FM coupling 设计与库内 SB Flow/GSBM 无缝衔接。

---

# 一、精读：Flow Matching for Generative Modeling

## 基本信息

- 论文：Flow Matching for Generative Modeling
- 方法名：FM / CFM（Conditional Flow Matching）；OT 路径版本常记 FM-OT / CondOT
- 作者：Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le（Meta AI FAIR + Weizmann）
- 会议：**ICLR 2023，notable-top-25%（Spotlight 档）**。venue 于 2026-08-14 核验：OpenReview `PqvMRDCJT9t` 标注 "ICLR 2023 notable top 25%"，iclr.cc 官方虚拟页 `iclr.cc/virtual/2023/poster/11309` 收录（虚拟站对 notable 论文统一显示 poster 页）。
- 链接：https://arxiv.org/abs/2210.02747
- 归类：simulation-free CNF 训练；conditional matching 范式源头；Gaussian 概率路径族；conditional OT 直线路径。

## 一句话总结

FM 提出用固定条件概率路径的向量场做回归来免模拟地训练 CNF：conditional FM 目标与不可算的 marginal FM 目标梯度相同，从而绕开扩散过程构造、直接设计概率路径——其中 conditional OT 直线路径在 ImageNet 上同时改进似然、FID、训练速度与采样 NFE。

## 解决的问题（动机）

CNF（Chen et al. 2018）理论上能表达任意概率路径，但训练不可扩展：最大似然要沿 ODE 前后向模拟；已有 simulation-free 尝试要么含高维不可算积分（Moser Flow），要么 minibatch 梯度有偏（Ben-Hamu et al. 2022）。扩散模型靠 denoising score matching 免模拟训练而可扩展，但概率路径被限制在简单扩散过程可解的族内——训练时间长、采样要靠专门快速采样器，且前向过程在有限时间内到不了真正的噪声先验（边界只能近似）。FM 的目标：保留回归式训练的可扩展性，同时把「可用的概率路径」从扩散族解放为任意（Gaussian 条件）路径。

## 方法核心

### 1. FM 目标与条件构造

给定目标概率路径 `p_t` 及生成它的向量场 `u_t`，FM 目标是 `L_FM(θ) = E_{t,p_t(x)} ||v_t(x;θ) − u_t(x)||²`。它不可直接用——`p_t, u_t` 未知。FM 的第一个关键构造：以数据点 `x_1` 为条件定义逐样本路径 `p_t(x|x_1)`（`t=0` 为噪声先验、`t=1` 集中在 `x_1`），marginal 路径 `p_t(x) = ∫ p_t(x|x_1) q(x_1) dx_1`；对应的 marginal 向量场为条件向量场的后验加权平均：

`u_t(x) = ∫ u_t(x|x_1) · p_t(x|x_1) q(x_1) / p_t(x) dx_1`

**Theorem 1**：若 `u_t(·|x_1)` 生成 `p_t(·|x_1)`，则上式的 `u_t` 生成 `p_t`（用连续性方程逐条验证）。这一步把未知的 marginal 场拆成了可显式写出的逐样本场。

### 2. CFM 目标：与 FM 梯度相等

marginal 场里的积分仍不可算，于是定义 Conditional Flow Matching：

`L_CFM(θ) = E_{t, q(x_1), p_t(x|x_1)} ||v_t(x;θ) − u_t(x|x_1)||²`

**Theorem 2**：在 `p_t(x)>0` 的条件下，`L_CFM` 与 `L_FM` 只差与 θ 无关的常数，故 `∇_θ L_FM = ∇_θ L_CFM`。训练只需能从 `p_t(x|x_1)` 采样并算 `u_t(x|x_1)`——完全 simulation-free、无偏。注意最优解 `v_t = u_t(x)` 是条件场的后验均值，而**不是**任何单条条件场；这正是后文「收敛点残余方差」讨论的出发点。

### 3. Gaussian 路径族与 canonical 向量场

取 `p_t(x|x_1) = N(x | μ_t(x_1), σ_t(x_1)² I)`。生成同一路径的向量场不唯一（可加无散度分量），FM 选择对应仿射流 `ψ_t(x) = σ_t(x_1) x + μ_t(x_1)` 的最简场。**Theorem 3** 给出闭式：

`u_t(x|x_1) = (σ'_t(x_1)/σ_t(x_1)) (x − μ_t(x_1)) + μ'_t(x_1)`

代入 CFM 后目标变为 `E ||v_t(ψ_t(x_0)) − d/dt ψ_t(x_0)||²`（`x_0∼N(0,I)`）——即后来所有 bridge/flow matching 工作沿用的「插值点处回归插值速度」形式。

两个实例：

- **扩散路径（VE/VP）**：把已有扩散过程的 `μ_t, σ_t` 代入 Theorem 3，恢复的条件场与 probability flow ODE（Song et al. 2020b, eq.13）一致。用 FM 损失训练扩散路径 = 一个比 score matching 更稳健的替代（见实验）。
- **conditional OT 路径**：`μ_t = t·x_1`，`σ_t = 1−(1−σ_min)t`，即均值与方差线性插值。其条件流恰是两个 Gaussian 间的 OT displacement map（McCann 1997, Example 1.7）：粒子直线恒速运动。条件场可写成 `g(t)h(x|x_1)` 的形式——回归目标的方向在时间上恒定，比扩散路径末端爆炸的 score 更易拟合。**作者明确提醒：conditional 路径最优不意味着 marginal 路径是 OT 解**——这句话正是 OT-CFM/MFM 两篇的出发点。

### 4. 与 score matching 的关系（本节为选题要求重点）

- **继承**：CFM 的「条件目标替代 marginal 目标」直接类比 denoising score matching（Vincent 2011）——DSM 用条件 score `∇ log p_t(x|x_1)` 回归实现对 marginal score 的无偏学习；CFM 把同一杠杆从 score 推广到**任意 Gaussian 路径的向量场**。
- **超越**：(a) score matching 只覆盖扩散过程诱导的路径，CFM 直接设计 `μ_t, σ_t`，扩散路径成为特例；(b) 扩散路径在有限时间到不了噪声先验，FM 路径边界可精确设定；(c) 参数化对象从 score 换成向量场后，扩散路径上的同一目标（FM w/ Diffusion）在实验中比 SM 更稳定（CIFAR-10 上 SM 的 FID 19.94 vs FM-Diff 8.06，且 SM 的采样 NFE 在训练中剧烈波动而 FM 恒定）；(d) 在 `σ→0` 极限下，OT 路径的 CFM 就是后来 rectified flow/stochastic interpolant 的线性插值回归——三者在 2022 年秋独立提出（论文 related work 自己指出 concurrent）。

## 实验与结果

同一 U-Net（Dhariwal & Nichol 架构）下换损失做消融，CIFAR-10 / ImageNet-32/64/128：

- **质量三指标同时占优**：CIFAR-10 上 FM-OT `NLL 2.99 / FID 6.35 / NFE 142`，对照 DDPM `3.12 / 7.48 / 274`、SM `3.16 / 19.94 / 242`；ImageNet-32 `3.53 / 5.02 / 122`；ImageNet-64 `3.31 / 14.45 / 138`；ImageNet-128 FID `20.9`（当时无条件生成中除 IC-GAN 外最好）。
- **训练更快**：ImageNet-64 的 FID 训练曲线全程压住三个基线；ImageNet-128 用 500k 迭代 × batch 1.5k（比 ADM 少 33% 图像吞吐）达到更优 FID。
- **采样更省**：OT 路径模型达到同等 ODE 数值误差只需扩散路径约 60% 的 NFE；低 NFE 区间 FID 显著更好；样本可视化显示 OT 路径「更早开始成形」，扩散路径噪声主导到最后一刻。
- **条件生成**：64→256 超分上 FID 3.4 / IS 200.8，优于 SR3（5.2 / 180.1）。

## 局限性

- **源分布锁死为 Gaussian**：条件构造单边（只以 `x_1` 为条件），`p_0` 必须是可采样、可写密度的先验。**不能直接做 data→data / unpaired sim→real**——这正是 OT-CFM 广义化的核心动机，对本库是最重要的一条。
- conditional OT ≠ marginal OT：独立采样 `(x_0,x_1)` 使条件直线相互交叉，marginal 场弯曲、收敛点目标方差不为零（MFM 量化了这一点）。
- 复现细节不全：OT-CFM 论文指出其 `σ_min`、FID 样本数、数据增广、训练 epoch 等未完整报告且有自相矛盾处，按原超参复现 CIFAR-10 只得 FID ~11.5（vs 报告 6.35），改进超参后反而到 3.66——引用 FM 数字时应注明这一复现分歧。
- 仅无条件/超分图像生成，无 translation、无机器人任务。

## 与库内 SB 系工作的关系

- **matching 语言的源头**：库内 SB Flow（α-DSBM）、GSBM、3MSBM、ASBS 的训练循环里那步「回归 bridge/条件路径的漂移」全部是 CFM 式目标；GSBM 报告（`2310.02233`）中的 explicit matching 更新、SB Flow 的 flow 形式都以 Theorem 1/2 的条件-边缘等价为地基。读懂 FM 的两个定理，SB 系论文的 matching 步骤就没有黑箱。
- **deterministic 对照的精确 spec**：`sb_adjoint_extended_synthesis.md` 第一阶段要求的「OT-ODE / flow matching deterministic 对照」应实现为本文的 FM-OT（Gaussian 源）与下文 I-CFM/OT-CFM（数据源）两档。
- **与 I²SB 的分工**：I²SB 处理 paired 两端（退化 bridge），FM 是单边 Gaussian 源；两者都不解决 unpaired data↔data——那是 SB Flow 与 OT-CFM/SB-CFM 的地盘。
- **有限时间边界性质**：FM 对「扩散前向到不了先验」的批评，同样是库内 SB 系（两端边缘都精确约束）叙事的一部分；写论文 preliminaries 时可从 FM 这句话自然引出 SB。

---

# 二、提炼：OT-CFM——minibatch OT coupling 的标准实现

## 基本信息

- 论文：Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport（arXiv v1 标题为 "Conditional Flow Matching: Simulation-free Dynamic Optimal Transport"，正式版改名）
- 作者：Alexander Tong*, Kilian Fatras*, Nikolay Malkin, Guillaume Huguet, Yanlei Zhang, Jarrid Rector-Brooks, Guy Wolf, Yoshua Bengio（Mila；*equal contribution）
- 发表：**TMLR，2024 年 3 月正式发表**。venue 于 2026-08-14 核验：正式 PDF 页眉 "Published in Transactions on Machine Learning Research (03/2024)"；OpenReview `HgDwiZrpVq`；爱丁堡大学机构库记录 "Transactions on Machine Learning Research, pp. 1-34, 2024"。引用时**勿再写 arXiv 2023**，并注意标题已更换。
- 链接：https://arxiv.org/abs/2302.00482 ｜ 代码：https://github.com/atong01/conditional-flow-matching （torchcfm 包，社区标准实现）
- 归类：广义 CFM；minibatch OT coupling；simulation-free dynamic OT / SB probability flow。

## 框架：把 FM 的条件变量广义化

条件 `z` 任意化后取 `z=(x_0,x_1)`、条件路径为两点间 Gaussian 流 `p_t(x|z)=N(tx_1+(1−t)x_0, σ²)`、`u_t(x|z)=x_1−x_0`。此时**源分布不再需要 Gaussian、不需要密度可算**——generative modeling 与 data→data translation 统一进一个目标。三个层级：

| 变体 | coupling `q(x_0,x_1)` | 条件路径方差 | 逼近对象 |
|---|---|---|---|
| I-CFM | `q_0 × q_1`（independent） | `σ²`（常数） | 任意源的生成模型（≈ rectified flow / stochastic interpolant 的推广） |
| OT-CFM | 2-Wasserstein OT plan `π`（minibatch 近似） | `σ²` | `σ→0` 时逼近 **dynamic OT**（Prop 3.4） |
| SB-CFM | entropic OT plan `π_{2σ²}` | Brownian bridge `σ²t(1−t)` | **SB 的 probability flow**（Prop 3.5，参考过程为 σ-Brownian motion） |

关键的「ε 谱」：SB-CFM 的熵正则 `ε→0` 退化为 OT-CFM，`ε→∞` 退化为 I-CFM——**independent 与 minibatch OT 不是二选一，而是同一坐标轴的两端，SB 恰好落在 `ε=2σ²` 处**。这是本报告 coupling 设计笔记的理论主轴。

minibatch 实现：每个训练步对 batch 内 `k` 对样本解一次精确 OT（POT 库），从 batch 级 plan 重采样配对。OT 批量与优化批量可以不同（论文为简便取相同）。

## 对 coupling 结论有效的实验证据

- **直线度/最优性**：2D 四组 transport 任务上，normalized path energy（NPE，路径能量对 `W₂²` 的相对偏差）OT-CFM 为 `0.018–0.087`，I-CFM 为 `0.222–2.738`——**差 1–2 个数量级**；`moons→8gaussians` 是 I-CFM 的重灾区（NPE 2.738）。学出的 map 拟合精度（test W₂²）也一致更好。
- **方差**：§C.1/附录 D.1 表明 `σ→0` 时 OT-CFM/SB-CFM 的目标残余方差 `E||u_t(x|z) − u_t(x)||² → 0`（independent coupling 做不到）；实测目标方差显著低于 CFM/FM（Fig D.4/D.8），并对应更快的验证集收敛（Fig 2 左）。
- **coupling 批量消融（Fig D.2）**：NPE 随 OT 批量增大而快速下降，**在 k≈64 后即 plateau**——不到 10k 数据集的 0.5%；难的分布对（moons→8gaussians）需要更大 k。
- **CIFAR-10 生成**：充分训练后 OT-CFM FID `3.577`（adaptive）vs I-CFM `3.659` vs 复现版 FM-OT `3.655`——**终点质量基本打平**；差距体现在训练前期与低 NFE 推理（100 步 Euler：4.443 vs 4.461；NFE 133.94 vs 146.42）。OT 求解开销 <1% 训练时间。
- **unpaired translation（对本库最相关）**：CelebA 40 属性、128 维 VAE latent 上学 unpaired 属性翻译，OT-CFM 的 MMD `2.81±2.62`（σ=0.1）显著优于 I-CFM `4.85±5.09`；σ 增大 MMD 降低但配对语义变差（σ>1 开始退化）。作者同时承认像素空间的 GAN 翻译仍占优——**FM/SB 系在像素级 unpaired 翻译上尚未全面超越 GAN**，与库内 SB Flow 报告口径一致。
- **SB 逼近**：SB-CFM 对 ground-truth SB 的平均 W₂ 误差在四组 2D 任务上全部优于 DSB（如 `N→8gaussians`：0.454 vs 1.440），且 CPU 训练快约 5 倍。**数值警告：σ 太小时 Sinkhorn 会数值不稳，SB-CFM 性能反而崩（Fig D.3）**。
- **单细胞插值**：CITE/EB/Multi 三个数据集 leave-one-timepoint-out，OT-CFM 全部第一（EMD 0.790–0.937），优于 DSB、I-CFM、SB-CFM 与 TrajectoryNet。
- **明示局限**：CFM 需要闭式条件流，难以在 marginal 场上加先验正则；**minibatch OT 在高维会引入对真 OT 的偏差**，作者建议后续用 neural OT 替换（与库内 G13 的 NOT 选题呼应）。

---

# 三、提炼：Multisample Flow Matching——coupling 选择的系统研究

## 基本信息

- 论文：Multisample Flow Matching: Straightening Flows with Minibatch Couplings
- 作者：Aram-Alexandre Pooladian, Heli Ben-Hamu, Carles Domingo-Enrich, Brandon Amos, Yaron Lipman, Ricky T. Q. Chen（NYU + Weizmann + Meta AI；与 OT-CFM 互为 concurrent work，两篇都在正文声明）
- 发表：**ICML 2023**。venue 于 2026-08-14 核验：PMLR 202:28100–28127（`pooladian23a`），Proceedings of the 40th ICML。
- 链接：https://arxiv.org/abs/2304.14772
- 归类：Joint CFM；minibatch coupling 族（BatchOT/BatchEOT/Stable/Heuristic）；直线度与方差的理论刻画。

## 框架与理论：coupling 影响什么、为什么合法

**Joint CFM**：`L_JCFM = E_{t, q(x_0,x_1)} ||v_t(x_t;θ) − u_t(x_t|x_1)||²`，`q(x_0,x_1)` 是任意满足两端边缘约束的联合分布。多样本构造：各采 `k` 个源/目标样本 → 按样本构造双随机矩阵 `π(i,j)` → 从离散联合分布重采样配对。**Lemma 4.1：不论 `π` 怎么选（只要双随机），隐式 coupling 的边缘严格等于 `q_0, q_1`**。这一条是与「静态拟合 batch OT」路线的分水岭：直接把神经网络回归到 batch OT 配对（barycentric projection）在有限批量下**不保边缘**（§5.1 详述），而动态 ODE 参数化天然保边缘。

coupling 的三个作用被分别量化：

1. **梯度方差**（Lemma 3.2）：固定 `(x,t)` 的梯度总方差被 `||∇_θ v_t||² × L_JCFM` 上界控制——JCFM 的**最优值**就是收敛点残余方差的 proxy。independent coupling 因路径交叉使该值恒不为零。
2. **直线度**（式 18/19）：`S = E[ ||u_t(ψ_t(x_0))||² − ||ψ_1(x_0)−x_0||² ] ≥ 0`，为零当且仅当轨迹是直线——直线 ODE 少步即可精确积分。
3. **传输代价**（式 20）：学到的 map `ψ_1` 的平方位移期望。

**Theorem 4.2（BatchOT，k→∞）**：(i) JCFM 最优值 → 0（方差消失）；(ii) 直线度 S → 0；(iii) 传输代价 → `W₂²`。**Theorem D.8**：传输代价上界随 `k` 单调不增——有限 `k` 也有弱保证。2D checkerboard 上的 JCFM 终值直接给出各 coupling 的方差排序：CondOT `10.72` ≫ Stable `1.60` ≈ Heuristic `1.56` > BatchEOT `0.57` > BatchOT `0.24`。

**coupling 族与计算复杂度**（batch 大小 `k`）：

| coupling | 算法 | 复杂度 | 性质 |
|---|---|---|---|
| CondOT（=independent/uniform） | 无 | `O(1)` | FM 默认；基线 |
| BatchOT | Hungarian/network simplex（POT） | `O(k³)` | 每 batch 精确 OT 置换 |
| BatchEOT | Sinkhorn | `Õ(k²/ε)` | ε 两端分别退化为 BatchOT / independent |
| Stable | Gale-Shapley 稳定匹配 | `O(k² log k)` | 只用相对排名；稳定性是 OT 的必要条件 |
| Heuristic | 修改版 GS（局部循环单调） | `O(k² log k)` | 代价感知的便宜近似 |

## 对 coupling 结论有效的实验证据

- **低 NFE 效率（主结果）**：face-blurred ImageNet-32/64。达到 FID=10（IN32）：CondOT 需 20 NFE，BatchOT/Stable 只需 14；达到 FID=20（IN64）：CondOT 29，Stable 11、BatchOT 12——**采样成本省 30–60%，训练时间只多 0.8%（IN32）/ 4%（IN64）**。极低 NFE 下对扩散基线是碾压级（IN32 Euler 8 步：BatchOT FID 15.64 vs DDPM 232.97）。
- **终点质量不退化**：adaptive 求解器下 BatchOT IN32 FID `4.68` vs CondOT `5.04`，NLL 双双 `3.58`；IN64 FID `12.37` vs `13.93`。**收益集中在效率与一致性，不指望 FID 大幅提升**。
- **高维方差降幅收窄**：`Var(u_t)` IN32 从 594（CondOT）降到 507（BatchOT，-15%）；IN64 从 1880 到 1733（-8%）。理解为 minibatch OT 在高维只能部分解开路径交叉——**k 固定时维度越高，coupling 越接近 independent**（这是把 coupling 从像素空间挪到 latent 的核心论据）。
- **样本一致性**：定义 Consistency(m)=低步数样本与高精度样本的 Inception 特征 MSE。BatchOT 在所有 m 全面更优（IN32 m=8：0.052 vs 0.079）——同一噪声种子下，低 NFE 生成的**内容**更接近高 NFE 结果。对「迁移后图像喂下游策略」这类对内容漂移敏感的用途，这个指标比 FID 更相关。
- **Stable coupling 的性价比**：IN64 上 Stable 的 FID（11.82）反而略优于 BatchOT（12.37），复杂度低一档——便宜 coupling 未必差。
- **coupling 批量实践**：ImageNet 实验只在**每 GPU 内**做 coupling（IN32 每 GPU 256、IN64 每 GPU 50）；作者试过跨 GPU 用 effective batch（1024/800）算 coupling，**没有收益且质量略降**。加大 coupling 批量的边际收益很快消失（与 OT-CFM 的 k≈64 饱和互证）；App B.6 显示更大 `k` 主要改善收敛稳定性。
- **未知 cost / oracle coupling 实验（对任务 cost 设计重要）**：只给「黑箱 oracle 的 batch 最优配对」（cost 未知，含 `L2²`、`L1`、cosine、线性变换 `L2²` 四种），Joint CFM 学出的动态 map 的代价**不高于 oracle 配对本身**（Theorem D.8 的实验支持）且 KL 极小（0.004–0.19）；静态 barycentric 拟合代价更低但 KL 高达 150–530——边缘完全崩坏。**结论：任意（甚至未知）cost 的 batch coupling 都能被 Joint CFM 消化，且边缘永远安全**——这是在 coupling 里塞任务感知 cost 的理论许可证。

---

# 四、coupling 设计笔记：unpaired sim↔real 图像的配对构造

## 4.1 三篇合并后的净结论（直线度 / 方差 / 质量）

1. **直线度**：independent coupling 的 marginal 路径必然弯曲（条件直线在空间中交叉平均）；minibatch OT coupling 使直线度随 `k→∞` 趋零（MFM Thm 4.2(ii)），有限 `k` 下 NPE 已降 1–2 个数量级（OT-CFM 2D）。实践含义：**closed-loop 机器人推理预算（10–50 Hz，等价 NFE 1–8）下，coupling 选择直接决定可行性**；这也是少步部署（G10 的 LBM/CDBM 蒸馏）之前最便宜的第一步。
2. **方差**：coupling 决定 CFM 目标在收敛点的残余方差（MFM Lemma 3.2 + OT-CFM §C.1），OT coupling 在 `σ→0` 时使其归零 → 训练收敛更快。但**高维图像上降幅只有 8–15%**——不改表示空间、只换 coupling，收益有限。
3. **生成质量**：终点 FID/NLL 与 independent 基本持平（两篇一致），低 NFE 质量与内容一致性显著更好；**unpaired translation 任务比无条件生成更受益**（CelebA latent 上 MMD 差近一倍）——因为 translation 的「map 语义」直接由 coupling 引导。
4. **边缘安全性**：任何双随机 coupling 都严格保两端边缘（MFM Lemma 4.1）；静态拟合 batch OT 配对则不保。**给 SB-Render-Lite 的直接含义：在 coupling cost 上做任何任务感知设计，都不会破坏「迁移后数据整体分布 = 真实域分布」这一底线**，设计自由度很大。

## 4.2 对 SB-Render-Lite 的具体设计建议

设 sim 边缘 `q_sim`（仿真 RGB 或其 latent）、real 边缘 `q_real`（真机 RGB 或其 latent），unpaired。

**A. 在哪个空间算 coupling——latent，不是像素。**
minibatch OT 的高维偏差（OT-CFM 结论）+ 方差降幅在 ImageNet 像素上收窄至 8%（MFM Table 6）+ 像素 L2 不是语义度量，三条证据都指向：先用 frozen encoder（VAE latent / DINO / CLIP）把图像降到 10²–10³ 维再解 OT。OT-CFM 自己的 unpaired 翻译实验就在 128 维 VAE latent 做。transport 本身若也在 latent 空间训练（与 G10 的 LBM 路线合流），coupling 与 transport 共用表示最干净。

**B. cost 度量——「视觉 + 任务」加权，理论上放心加。**
- 基线：latent 平方欧氏 `||f(x_sim) − f(x_real)||²`（对齐 W₂ 理论与 Brenier 结构）。
- 升级：`c(x_sim, x_real) = ||f_vis(x_sim) − f_vis(x_real)||² + λ_geo ||f_geo(·)||² + λ_task ||f_task(·)||²`，其中 `f_geo` 可用 depth/keypoint 特征、`f_task` 用任务阶段或（若有）state/action 特征——即把库内 EgoBridge 的 joint feature-action cost 思想搬进 coupling。MFM 的未知 cost 实验（含 cosine 与线性变换代价）证明**任意 cost 的 batch coupling 都能被 Joint CFM 消化且边缘不坏**；唯一代价是 cost 偏离平方欧氏越远，「直线插值路径 + 该 coupling」组合的动力学意义越弱（直线路径隐含欧氏几何）——建议 λ 从小到大消融。
- cosine cost 在 latent 空间是被 MFM 验证过的合法选项（对尺度不敏感，适合跨域特征幅值不一致的场景）。

**C. coupling 批量 `k`——64–256 起步，不必贪大。**
- 证据：OT-CFM 低维 NPE 在 `k≈64` 饱和（<0.5% 数据集）；MFM ImageNet 用每 GPU 50–256，且**跨 GPU 扩大 coupling 批量无收益甚至略差**；App B.6 表明大 `k` 主要帮收敛稳定性。
- 建议：`k=128` 默认（latent 空间 `O(k³)` 的精确 OT 在此规模开销 <1–4% 训练时间，POT 即可）；难配对的分布（sim 与 real 视角/布局差异大）适当加大到 256–512；每步重新采样、重新解 OT，配对不缓存。
- 若追求更便宜：Stable coupling（`O(k² log k)`）是 MFM 实测不输 BatchOT 的替代，值得作为消融档位。
- 若用 Sinkhorn（BatchEOT）：`ε` 别贪小——OT-CFM 实测 σ（等价 ε）太小时 Sinkhorn 数值崩溃。

**D. 确定性 vs 随机——用「ε 谱」组织消融，而不是各做各的。**
同一套代码沿 `ε` 扫出三档：`ε→0`（OT-CFM，deterministic OT 对照）、`ε=2σ²` + Brownian bridge 路径（SB-CFM，= σ-参考过程 SB 的 probability flow）、`ε→∞`（I-CFM，independent 基线）。这恰好落进 `sb_adjoint_extended_synthesis.md` 第一阶段的消融矩阵：I²SB/SB Flow baseline + OT-ODE deterministic ablation 现在有了精确到超参的定义。σ/ε 同时是「realism 多样性 vs 配对确定性」的旋钮：sim→real 渲染修正预期接近确定性 map（一张 sim 图只该有少数合理 real 对应），**建议 σ 取小、ε 取小端，把多模态留给光照/纹理等真不确定的维度**。

**E. 结构化分桶 coupling——防语义错配。**
batch 内全局 OT 可能把 sim 抓取帧配到 real 放置帧（视觉相似但任务阶段不同）。建议按 task phase / 场景 / 相机视角先分桶，桶内做 BatchOT——等价于在 cost 里加桶指示的无穷惩罚，仍在 Lemma 4.1 的保边缘框架内。库内 Guided OT Co-Training 的引导思想与 G12 备选 COT Policy 的「conditional 任务中 naive coupling 失效」结论都支持这一条。时序数据（视频/轨迹）另需 episode 级 coupling 与帧级 coupling 分层设计，衔接 3MSBM 的多边缘路线，本报告不展开。

**F. 评估口径——按三篇的指标体系补齐，不只 FID。**
- 直线度/效率：NPE（OT-CFM 式）或直线度 S（MFM 式 18）、FID-vs-NFE 曲线（重点 NFE≤8）。
- 方差：JCFM 终值（= 收敛点残余方差上界，MFM 用它排序 coupling）。
- 内容一致性：Consistency(m)（MFM 式 21，Inception/DINO 特征 MSE）——对「迁移图像喂策略」比 FID 更相关。
- 最终裁决：real-domain policy success + keypoint/depth/inverse-dynamics 保持（库内主口径不变）。

**G. 失败模式清单。**
(i) 高维直接算像素 OT → coupling 退化为近似 independent，白付 `O(k³)`；(ii) Sinkhorn ε 过小 → 数值不稳；(iii) 全局 OT 跨任务阶段错配 → 语义漂移进 transport；(iv) 静态蒸馏 batch 配对（barycentric）→ 边缘崩坏（MFM Table 4 的 KL 150–530 教训）；(v) 把 coupling 批量与优化批量强行绑死 → 二者本可解耦（OT-CFM 明示）。

## 4.3 与 GSBM / SB Flow 的方法衔接

把三层方法放进一个谱系（成本递增、逼近对象递强）：

| 层级 | coupling 如何得到 | 逼近对象 | 训练成本 | 库内对应 |
|---|---|---|---|---|
| I-CFM / FM-OT | independent（或 Gaussian 源） | 任意保边缘流 | 一次回归 | 本报告 §一、§二 |
| OT-CFM / MFM | **一次性** minibatch OT/EOT，训练中固定规则 | `σ→0`：dynamic OT；`ε=2σ²`：SB flow | 一次回归 + 每步一次 batch OT | 本报告 §二、§三 |
| SB Flow(α-DSBM) / DSBM / GSBM | **迭代更新**：模拟当前 bridge 得新 coupling，再 matching | 真 SB / GSB 不动点 | 多轮「模拟+回归」 | `2409.09347`、`2310.02233` |

三条具体衔接：

1. **SB-CFM 是 SB Flow 的一步近似与 warm start**。SB Flow/IMF 类方法从 independent coupling + Brownian bridge 出发迭代逼近 SB；SB-CFM 用 batch entropic coupling **直接**近似 SB 的静态耦合，一轮回归拿到「接近 SB」的初始 drift。工程路径：先训 SB-CFM（便宜、simulation-free），若指标不够再切 SB Flow 迭代精修——两者共享条件路径与网络，切换成本低。SB-CFM 的偏差来源是 minibatch EOT 对全局 EOT 的近似（batch 有限 + 高维），SB Flow 的迭代恰好在修的就是这个 coupling 的偏差。
2. **GSBM 的 coupling 初始化与 cost 分工**。GSBM（库内 `2310.02233`）交替「给定 coupling 解 conditional SOC 路径 + matching 更新 drift/coupling」。minibatch OT coupling 可作它的第 0 轮 coupling（比 independent 起点好，且 simulation-free）；分工原则：**粗粒度「谁配谁」的语义约束放进 coupling cost（本笔记 §4.2B/E），细粒度「路径途中不许变形」的约束放进 GSBM 的状态代价 `V_t`**（DINO/keypoint/depth 保持）。两处约束互补而不冗余：coupling 管端点配对，`V_t` 管中间路径。
3. **对照组的完整性**。审稿人视角下，SB-Render-Lite 的方法消融现在可以写成一条干净的轴：I-CFM（无 coupling 设计）→ OT-CFM（+minibatch coupling）→ SB-CFM（+熵/随机性）→ SB Flow(α-DSBM)（+迭代精修）→ GSBM（+任务路径代价）。每一步只加一个组件，每个组件都有本报告或库内报告的一手依据。

---

# 五、venue 核查记录（检索日期 2026-08-14）

| 论文 | arXiv | 正式发表 | 证据 |
|---|---|---|---|
| Flow Matching for Generative Modeling | 2210.02747 | **ICLR 2023（notable-top-25% / Spotlight 档）** | OpenReview `PqvMRDCJT9t`（"ICLR 2023 notable top 25%"）；iclr.cc/virtual/2023/poster/11309 |
| Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport（OT-CFM） | 2302.00482 | **TMLR，2024-03**（v1 标题 "Conditional Flow Matching: Simulation-free Dynamic Optimal Transport"，正式版改名） | 正式 PDF 页眉 "Published in Transactions on Machine Learning Research (03/2024)"；OpenReview `HgDwiZrpVq`；U. Edinburgh 机构库记录 TMLR 2024 pp.1-34 |
| Multisample Flow Matching: Straightening Flows with Minibatch Couplings | 2304.14772 | **ICML 2023**（PMLR 202:28100–28127） | proceedings.mlr.press/v202/pooladian23a.html |

与任务书预期核对：FM=ICLR 2023 ✓（补充 notable-top-25% 档位）；OT-CFM 正式版本=TMLR 2024-03 ✓（注意改名）；Multisample FM=ICML 2023 ✓。OpenReview 直接访问遇反爬验证（与 R05 记录一致），FM/OT-CFM 的 OpenReview 信息经搜索引擎摘要 + 官方 proceedings/机构库交叉验证。

---

# 六、并入主库建议

1. **INDEX.md**：新增分组「方法基础：Flow Matching 与 coupling 设计」，收本报告一条（或按库惯例拆为报告条目 + 归入综合入口）。建议放在「重要对照 / SB 方法支撑」之前，因为它是 SB Flow/GSBM 报告的前置阅读。
2. **metadata/papers.tsv**：追加三行——`2210.02747`（year=2023, ICLR notable-top-25%）、`2302.00482`（year=2024, TMLR；注意用正式标题）、`2304.14772`（year=2023, ICML/PMLR v202）。OT-CFM 若库内他处曾以 arXiv v1 旧标题引用，需统一为正式标题。
3. **synthesis.md §4.2（baseline 规格）**：deterministic 对照「OT-ODE / flow matching」可更新为本报告 §4.2D 的三档 ε 谱 spec（I-CFM / OT-CFM / SB-CFM），并引用 §4.2C 的 coupling 批量与 §4.2F 的评估指标。
4. **与其他扩充选题的边界**：求解器谱系图（DSB→DSBM→SB Flow）归 T1（DSBM 精读）；FM↔SB 的理论桥（stochastic interpolants 显式优化恢复 SB）归 T5（RF+SI 精读）；本报告只负责 FM 本体与 coupling 设计，三者引用不重复。COT Policy 的 conditional coupling 失效结论（T18）与本报告 §4.2E 互为印证，建议 T18 完成后交叉引用。
5. **SB-Render-Lite 实验清单可直接落地的三件事**：(a) 用 torchcfm 在 VAE latent 上跑 I-CFM vs OT-CFM vs SB-CFM 三档（`k=128`，latent 平方欧氏 cost）作为第一阶段最便宜的 coupling 消融；(b) 把 JCFM 终值与 Consistency(m) 加进评估脚本；(c) 任务感知 cost（+depth/keypoint 特征项）作为第二阶段消融，先在 coupling 做，再决定是否升级到 GSBM 的 `V_t`。
