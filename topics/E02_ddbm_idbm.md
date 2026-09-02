# E02 扩充报告：DDBM 精读 + IDBM（bridge matching 理论源头）笔记

## 选题定位

本报告来自 2026-08-14 缺口分析：库内已覆盖 I²SB（paired restoration 强基线）与 Schrödinger Bridge Flow（unpaired 方法底座），但缺少两块拼图——(1) **DDBM**（Denoising Diffusion Bridge Models），它是 paired 分布翻译的另一强基线，把 diffusion 全套工程设计（EDM 参数化、噪声调度、高阶采样器）系统性推广到 bridge 上，且在 pixel-space paired translation 上显著优于 I²SB；(2) **IDBM**（Iterated Diffusion Bridge Mixture），它是 I²SB / DDBM / DSBM 这一整条 "bridge matching" 技术路线的理论源头之一，给出了「桥混合 → Markov 化 → 迭代收敛到 SB」的完整定理链。前者精读，后者只做一页核心定理笔记。

## TL;DR

- DDBM（arXiv 2309.16948，ICLR 2024 poster，已核验）把 image-to-image translation 表述为「反转一个两端钉住的 diffusion bridge」：训练时对解析可采样的 Gaussian bridge 中间态做 denoising bridge score matching，采样时解一个带 Doob h 项的反向 SDE / PF-ODE。在 Edges→Handbags 64×64 与 DIODE 256×256 上，同等采样步数下 FID/LPIPS/MSE 全面优于 I²SB、Rectified Flow、Pix2Pix、SDEdit、DDIB；退化到噪声端点时复现 EDM 的无条件生成质量。
- DDBM 的适用前提与 I²SB 相同：**需要 paired (x_0, x_T) 训练数据**。它不解 Schrödinger Bridge 的最优耦合——耦合直接取自数据；DSBM/SB Flow 才是 unpaired 场景解 SB 耦合的方法。
- IDBM（arXiv 2304.00917，JMLR 24(374):1–51, 2023，已核验）证明：任取一个耦合 C，把参考过程的 bridge 按 C 混合，再用一个 drift 为条件期望的单一 diffusion 匹配其边缘分布（Markov 化），**每一次迭代都是 Γ→Υ 的合法 transport**，且 KL 单调、迭代在法律意义下收敛到 SB 解。这就是 bridge matching / IMF 路线的理论地基；DSBM-IMF 与之等价（并行独立工作），I²SB 与 Rectified Flow 分别是其 Brownian 参考特例与 σ→0 极限。
- 对 SB-Render-Lite 的直接结论：若有 paired sim/real 帧，paired 翻译基线应设 **I²SB + DDBM(VP)** 双基线（DDBM 质量更高、faithfulness 指标更好，但对参数化/latent 空间更敏感）；若在 latent 空间做，DDBM 自报的 latent 负结果提示优先 I²SB / Rectified Flow。

---

# Part A · DDBM 精读报告

## A1 基本信息

- 论文：Denoising Diffusion Bridge Models
- 方法名：DDBM
- 作者：Linqi Zhou, Aaron Lou, Samar Khanna, Stefano Ermon（Stanford）
- 会议：ICLR 2024（poster，主会；OpenReview id `FKksTayvGo`，任务侧已核验，本次检索 2026-08-14 再次确认）
- 链接：https://arxiv.org/abs/2309.16948 ｜ 全文精读用 https://ar5iv.labs.arxiv.org/html/2309.16948
- 代码：https://github.com/alexzhou907/DDBM（官方，含 Edges2Handbags 与 DIODE 的 VP 预训练权重，HuggingFace `alexzhou907/DDBM`）
- 归类：paired distribution translation；diffusion bridge；Doob h-transform；EDM 式工程设计推广。

## A2 一句话总结

DDBM 用 Doob h-transform 把 diffusion 过程钉在成对端点 `(x_0, x_T) ~ q_data(x, y)` 之间，训练网络匹配解析 Gaussian bridge 的条件 score，从而以「与 EDM 几乎同构」的训练/采样管线实现任意两个成对分布之间的翻译；diffusion 无条件生成与 OT-Flow-Matching/Rectified Flow 分别是它的两个特例。

## A3 动机与问题设定

标准 diffusion 假设先验是白噪声，做 image translation 只能靠 conditioning（Palette 式）或改采样（SDEdit 式），理论上不 principled 且单向。另一侧，直接建模分布间 transport 的方法各有短板：flow matching 类 ODE 方法在翻译任务上验证不足、生成质量不及 diffusion；经典 SB（DSB/IPF）与 bridge matching 的迭代式解法（DSBM、IDBM）需要多轮昂贵迭代。

DDBM 的设定是：端点服从**未知联合分布** `(x_0, x_T) = (x, y) ~ q_data(x, y)`，训练集给出成对样本，目标是学会从 `q_data(x | y)` 采样——即给定源域样本 y（如 sketch / normal map / 仿真帧），生成对应目标域样本 x。注意这是「给定耦合、反转桥」的问题，**不是**「求最优耦合」的 SB 问题；这一点是它与 DSBM/SB Flow 分野的根源。

## A4 方法核心

### A4.1 钉住端点的桥与两个解析量

对参考 SDE `dx_t = f(x_t,t)dt + g(t)dw_t`，Doob h-transform 给出到达指定终点 y 的条件过程：`dx_t = f dt + g² h(x_t,t,y,T) dt + g dw_t`，其中 `h = ∇_{x_t} log p(x_T=y | x_t)` 是参考过程转移核的对数梯度。对 VE/VP 这类 Gaussian 转移核，h 解析（论文 Table 1 给出 VE：`h = (x_T − x_t)/(σ_T² − σ_t²)`；VP 为 SNR 加权版本）。

训练用的中间态分布取「两端都钉住」的 `q(x_t | x_0, x_T) = N(μ̂_t, σ̂_t² I)`：

- `μ̂_t = (SNR_T/SNR_t)(α_t/α_T) x_T + α_t x_0 (1 − SNR_T/SNR_t)`
- `σ̂_t² = σ_t² (1 − SNR_T/SNR_t)`

即均值是（缩放后）端点的线性插值、方差在两端收缩为 0——采样 x_t 无需模拟 SDE（simulation-free），与 DDPM 训练同构。

### A4.2 反向 SDE / 概率流 ODE（Theorem 1）

条件边缘 `q(x_t | x_T)` 的演化有反向 SDE：

`dx_t = [f − g²( s(x_t,t,y,T) − h(x_t,t,y,T) )] dt + g dŵ_t, x_T = y`

以及对应 PF-ODE（把 s 前系数换成 ½）。其中 `s = ∇_{x_t} log q(x_t | x_T)` 是**唯一需要学习的量**，h 解析已知。直觉：h 是把过程拉向端点 y 的「导航项」，s 是数据耦合诱导的「修正项」；这与把 classifier guidance 里的 guidance 项内置到过程定义里同构，论文据此引入 guidance 强度 w 调节 h 的权重（generalized time-reversal）。

### A4.3 Denoising Bridge Score Matching（Theorem 2）

核心训练定理：设 `(x_0,x_T) ~ q_data`、`x_t ~ q(x_t|x_0,x_T)`、任意非零时间分布 p(t) 与权重 w(t)，则最小化

`E[ w(t) ‖ s_θ(x_t, x_T, t) − ∇_{x_t} log q(x_t | x_0, x_T) ‖² ]`

的解满足 `s_θ = ∇_{x_t} log q(x_t | x_T)`。即：**对解析 Gaussian bridge 的条件 score 做回归，就能学到以端点 y 为条件的真实 bridge score**。证明与 denoising score matching 完全平行（L2 回归的最优解是条件期望，附录 A.2 五行推完）。注意网络显式接收 x_T 作条件（实现上 input-level concatenation），这与 I²SB「条件只通过采样起点隐式进入」不同。

### A4.4 广义 EDM 参数化与 hybrid 采样器

- **pred-x 参数化**：沿 EDM 的 `D_θ = c_skip x_t + c_out F_θ(c_in x_t, c_noise)`，作者从「输入/目标单位方差 + 最小化 c_out」第一性原理重推 c_in/c_out/c_skip/w(t)，新增两个刻画端点分布的超参 `σ_T`（x_T 方差）与 `σ_0T`（端点协方差）。当 `x_T = x_0 + Tε` 时严格退化为 EDM 系数——所以这是 EDM 预条件的**严格超集**。翻译实验取 `σ_0 = σ_T = 0.5, σ_0T = σ_0²/2`。
- **hybrid sampler**：纯 PF-ODE 从固定数据点 x_T 出发会走「期望路径」，输出发糊；作者在 Heun 高阶 ODE 步之间按比例 s 插入 Euler–Maruyama SDE 步注入随机性（predictor-corrector 思路）。消融显示 s 从 0 到 0.3 左右 FID 大幅下降，ODE-only 输出明显模糊。
- 训练配置（事实）：AdamW lr 1e-4，500K iter，4×A100 40G；翻译任务 N=40 个采样步（附录注明该 hybrid 采样器合计 NFE=118）。

### A4.5 统一视角

- 端点取 `x_T ~ N(α_T x_0, σ_T² I)` 时，中间边缘、反向 SDE/ODE 全部退化为标准 diffusion——无条件生成是特例。
- 取 VE 桥 `σ_t² = c² t`，令 c→0，PF-ODE 的 drift 逐点收敛到直线速度场 `x_1 − x_0`——OT-Flow-Matching 与 Rectified Flow 是 noiseless 极限特例。
- Brownian bridge（BBDM 等离散时间工作所用）只是 VE 桥的特例；DDBM 表明可从任意 VP/VE diffusion 构造连续时间桥，且 VP 桥实证上更强。

## A5 与 I²SB / DSBM 的技术区别（本节含对比性总结，判断为个人梳理，事实均出自各论文）

| 维度 | I²SB（ICML 2023，库内已核验） | DDBM（ICLR 2024） | DSBM（NeurIPS 2023，检索 2026-08-14 核验） |
| --- | --- | --- | --- |
| 问题 | paired restoration，给定退化算子采样对 | paired translation，端点联合分布任意 | **unpaired** 边缘分布间的 SB/熵正则 OT |
| 耦合来源 | 数据对（可合成） | 数据对（q_data(x,y) 直接给出） | 由 IMF 迭代逐步逼近 SB 最优耦合 |
| 学习目标 | DDPM 式 posterior 去噪回归（tractable SB 类） | 条件 bridge score `∇log q(x_t\|x_T)`，网络显式吃 x_T | 每轮 bridge matching 回归 + 耦合更新 |
| 迭代性 | 一次训练 | 一次训练 | 多轮 IMF 迭代 |
| 参数化/采样 | DDPM 工具链 | EDM pred-x 超集 + hybrid Heun/EM 采样器 + guidance w | 各轮内同 bridge matching |
| 参考过程 | scaled Brownian（对称噪声） | 任意 VP/VE diffusion | 一般参考过程 |

关键区分：I²SB 与 DDBM 同属「**给定耦合的一次性 bridge matching**」（IDBM 语言里的第一次迭代 / DBM transport），差别主要在参数化与采样工程；DSBM/SB Flow 属「**求耦合的迭代 SB 解法**」。DDBM 论文明确承认与 Peluchetti 的 forward-time 视角可推出类似框架，自身增量在于 reverse-time 视角带来的 diffusion 设计复用与 FM/RF 统一。

## A6 实验与主要结论

**Pixel-space paired translation**（Edges→Handbags 64×64、DIODE-Outdoor 256×256；全部基线同架构、同采样步数 N=40）：

| 方法 | E→H FID↓ | E→H MSE↓ | DIODE FID↓ | DIODE MSE↓ |
| --- | --- | --- | --- | --- |
| Pix2Pix | 74.8 | 0.209 | 82.4 | 0.133 |
| SDEdit | 26.5 | 0.510 | 31.14 | 0.534 |
| Rectified Flow | 25.3 | 0.088 | 77.18 | 0.157 |
| I²SB | 7.43 | 0.191 | 9.34 | 0.145 |
| DDBM (VE) | 2.93 | **0.013** | 8.51 | **0.0107** |
| DDBM (VP) | **1.83** | 0.0402 | **4.43** | 0.0839 |

主要结论（论文事实）：

1. DDBM 在 9 项指标绝大多数上领先，最接近的对手是 I²SB，但低 NFE 预算下 I²SB 明显掉队；Rectified Flow 在两域低层外观差异大时（normal map→RGB）表现差；DDIB 完全失败。
2. MSE/LPIPS 的大幅优势说明 DDBM 的翻译**忠实度**（贴近 paired ground truth）远超基线——VE 桥 MSE 比 I²SB 低一个数量级。
3. 消融：预条件与 hybrid 采样器缺一不可（E→H VE：都不用 14.02 → 都用 2.93；DIODE VE：126.3 → 8.51）；VE 桥对 guidance w 不敏感，VP 桥强依赖 w=1（作者解释：VP 曲线路径中途破坏信号，更依赖 h 项导航）。
4. 无条件生成：CIFAR-10 FID 2.06（NFE 35）对 EDM 2.04；FFHQ-64 2.44 对 EDM 2.53——「更一般的框架不牺牲生成质量」的主张成立。
5. **诚实负结果**：latent-space 翻译（day→night 256，SD autoencoder 8× 下采样）DDBM FID 27.63，输给 Rectified Flow 12.38 与 I²SB 15.56（IS/MSE 第二）。作者推测 pred-x 参数化是为 pixel 空间调的，latent 结构更难学。

个人推断（非论文声明）：主文说各方法「same number of function evaluations (N=40)」，但附录注明 DDBM hybrid 采样器 N=40 对应 NFE=118（每步含 Euler+两次 Heun 评估）；若基线按每步 1–2 次网络调用计，实际算力对齐是按「采样步数」而非严格 NFE。复现对比时应按 NFE 对齐重跑，或至少报告两种口径。

## A7 局限

- 与 I²SB 一样**必须有 paired 数据**：真实 sim2real 中每帧真实图像通常没有对应仿真场景，这仍是主要适用性障碍。
- latent 空间表现弱于 RF/I²SB（论文自报），预条件系数对数据统计（σ_0、σ_T、σ_0T）敏感，换域需重调。
- 采样忠实度好不等于**任务量保持**：MSE 是对 paired GT 的像素距离，不度量深度/接触状态/action 一致性（个人推断，延续库内对 I²SB 的同类批评）。
- hybrid 采样器的随机步比纯 ODE 慢且引入方差；NFE=118 对机器人数据管线偏贵。后续已有两个已核验的加速工作：DBIM（Diffusion Bridge Implicit Models，arXiv 2405.15885，ICLR 2025 poster，最高 25× 提速）与 CDBM（Consistency Diffusion Bridge Models，arXiv 2410.22637，NeurIPS 2024）（检索 2026-08-14）。
- 理论上 DDBM 不解 SB：不宣称耦合最优性，也没有 unpaired 能力；这不是缺陷而是设定差异，但选型时必须分清。

## A8 与库内已有工作的关系

- **I²SB（`reports/2302.05872_i2sb.md`）**：同一「paired 一次性 bridge」家族。I²SB 从 tractable SB 类推导，DDBM 从 reverse-time bridge 推导，两者训练分布同为解析 Gaussian bridge；DDBM 的增量 = EDM 参数化超集 + hybrid 采样器 + 显式 x_T 条件 + FM/RF/diffusion 统一。实验上 DDBM 在 pixel 空间以显著优势胜出、在 latent 空间反被 I²SB 压制——库内「I²SB 是 paired 强基线」的结论应修订为「pixel 空间首选 DDBM(VP)，latent 空间仍用 I²SB/RF」。
- **SB Flow（`reports/2409.09347_schrodinger_bridge_flow_unpaired_translation.md`）**：互补而非竞争。SB Flow 解 unpaired 边缘间的 SB（IMF/bridge matching 路线的连续 flow 推广），DDBM 假设耦合已知。SB-Render-Lite 的「paired 预训练 → unpaired 适配」两段式里，DDBM 替换第一段的 I²SB 位置，SB Flow 负责第二段。
- **GSBM（`reports/2310.02233_generalized_schrodinger_bridge_matching.md`）**：若要把 geometry/action 保持写成路径代价，GSBM 是在 DDBM/DSBM 之上加 state cost 的自然框架；DDBM 的 bridge score matching 目标可视为 GSBM 无 state cost、固定耦合的退化情形（个人推断）。
- **BDGxRL（`reports/2602.23737_bdgxrl_diffusion_schrodinger_bridge.md`）**：其 dynamics gap 桥接同样落在「两端分布 + bridge」框架，DDBM 的 VP 桥与 pred-x 工程细节可直接移植到其视觉部分（个人推断）。

## A9 对 SB-Render-Lite 的直接启发（paired 翻译基线选型）

1. **双 paired 基线**：renderer 可对同一 scene 出 paired sim/real-like 帧时，基线组 = I²SB + DDBM(VP)，同一 ADM 骨干、同一 NFE 口径。预期（推断）DDBM(VP) 在 FID 与 keypoint/depth 保持上占优，因为其 MSE 忠实度高一个量级。
2. **空间选择**：SB-Render-Lite 若走 SD-VAE latent 管线以省算力，注意 DDBM 的 latent 负结果——要么留在 pixel 64–256 分辨率，要么为 latent 重调预条件（σ_0/σ_T/σ_0T 按 latent 统计重估）后再对比。
3. **随机性=可控消融旋钮**：hybrid 采样器的 s（Euler 步占比）与 guidance w 给了「翻译随机性 vs 几何保真」的显式调节维度；建议消融 s ∈ {0, 0.3}、并把 s=0 的 OT-ODE 极限与 SB Flow 的确定性极限对齐比较（呼应库内 I²SB 报告的同款消融建议）。
4. **NFE 预算**：机器人数据集是百万帧级，NFE=118 不现实；若 DDBM 胜出，直接上 DBIM（免训练加速，ICLR 2025）做部署侧采样。
5. **评估纪律不变**：FID/LPIPS/MSE 之外必须加 policy success、inverse dynamics consistency、depth/keypoint preservation——DDBM 的 MSE 优势只是必要条件（推断，沿库内共识）。

---

# Part B · IDBM 核心定理笔记（一页）

## B1 基本信息

- 论文：Diffusion Bridge Mixture Transports, Schrödinger Bridge Problems and Generative Modeling
- 作者：Stefano Peluchetti（Cogent Labs）
- 期刊：JMLR 24(374):1–51, 2023（4/23 投稿、10/23 发表，编辑 Aapo Hyvärinen；任务侧已核验，检索 2026-08-14 于 jmlr.org 再次确认）
- 链接：https://arxiv.org/abs/2304.00917 ｜ https://www.jmlr.org/papers/v24/23-0527.html
- 代码：https://github.com/stepelu/idbm-pytorch
- 归类：dynamic Schrödinger Bridge；bridge matching 理论；IMF 等价工作。

## B2 核心对象与定理陈述

**设定**。目标边缘 `Γ, Υ`，参考 diffusion `R`（drift μ_R、扩散 σ_R），动态 SB 问题 `S* = argmin_{S ∈ P_C(Γ,Υ)} KL(S ‖ R)`。经典分解：`S* = S*_{0,τ} R_{•|0,τ}`，即 SB 解 = 参考过程的 bridge 按最优耦合混合；难点全在求静态耦合 `S*_{0,τ}`。

**Theorem 1（Diffusion mixture matching / Markov 化）**。设一族 diffusion `{P^λ}` 按 Ψ 混合得到过程 Π（一般不是 Markov 的），则取 drift/扩散系数为各分量系数按后验权重 `p_t^λ(x)Ψ(dλ)/π_t(x)` 的条件平均，所得**单一 diffusion 与 Π 在每个时刻边缘分布相同**。（把 Brigo 2002 的有限一维混合结果推广到一般混合。）

**DBM transport**。对任意耦合 `C ∈ P_2(Γ,Υ)`，令 `Π(C) = C R_{•|0,τ}`（参考桥按 C 混合）。对其应用 Theorem 1 得到 diffusion `M(Π(C))`：

`dX_t = [μ_R(X_t,t) + Σ_R(X_t,t) · E_Π[∇_{X_t} log r_{τ|t}(X_τ | X_t) | X_t]] dt + σ_R dW_t, X_0 ~ Γ`

它与 Π(C) 逐时刻同边缘，故**对任意 C 都是 Γ→Υ 的合法 transport**。drift 里的条件期望用 L2 回归学习（式 27）：`min_α E_{t,Π} ‖α(X_t,t) − Σ_R ∇log r_{τ|t}(X_τ|X_t)‖²`——这正是后来通称的 **bridge matching** 损失；反向时间版本（BDBM）回归 `∇log r_{t|0}(X_t|X_0)`，与 score-based 生成模型只差「期望取在 Π 还是 R 下」。

**IDBM 迭代（Algorithm 2）**。`C^(0)` 任取（如独立耦合 Γ⊗Υ），循环：`Π^(i) = Π(C^(i−1)) → M^(i) = M(Π^(i)) → C^(i) = M^(i)_{0,τ}`。

**Theorem 2（IDBM 收敛）**。σ_R = I 及温和条件下：

1. `Π^(i) → S*`、`M^(i) → S*`（依法律收敛，i→∞）；
2. KL 单调链：`KL(Π^(i)‖S*) ≥ KL(M^(i)‖S*) ≥ KL(Π^(i+1)‖S*)`；
3. 不动点判据：`KL(Π(C)‖S*) = KL(M(C)‖S*)` 当且仅当 `Π(C) = M(C) = S*`。

**配套事实**（论文）：与 IPF/DSB 相比，IDBM 每轮都保持双边缘正确（IPF 每轮只匹配一个边缘，仅极限处是合法耦合），且 Π^(i) 与 M^(i) 同边缘故**无 simulation-inference mismatch**——这解释了 Gaussian 解析实验与 MNIST↔EMNIST 迁移实验中 σ→0 时 IPF 发散/停滞而 IDBM 稳健的现象（DIPF 在 σ=0.5/0.2 时几乎不动，IDBM 仍收敛，但 σ=0.2 也开始退化；σ=0 即 Rectified Flow 情形两者都学不出合法 transport）。1-d Gaussian 情形迭代映射是相关系数上的压缩映射，且 σ→0 时**第一次迭代**即依法律收敛到 OT 解。

## B3 意义解读（一段）

IDBM 的价值在于把散落的方法钉进一张定理网：**「第一次迭代就是合法 transport」**（Theorem 1 + DBM 构造）为 I²SB、DDBM、Aligned SB 这类"给定耦合、一次训练"的 paired 方法提供了共同的正当性来源——它们都是 DBM transport 在特定参考过程/参数化下的实例（论文第 5 节明确指出 I²SB 等价于 scaled Brownian 参考下的 DBM，Rectified Flow 是 σ→0 极限，FM/CFM 是 RF 的第一次迭代）；而**「迭代收敛到 SB」**（Theorem 2）则是 DSBM-IMF 与后续 SB Flow 这条 unpaired 路线的收敛性依据（DSBM 为并行独立的等价工作，NeurIPS 2023，已核验）。对本库最有操作性的推论是：paired 与 unpaired 方法不是两个流派而是**同一迭代的第 1 步与第 ∞ 步**——SB-Render-Lite 的「I²SB/DDBM 预训练 → SB Flow 适配」两段式，本质上是「用数据耦合初始化 C^(0)，再跑若干 IMF 迭代逼近 SB」，IDBM 的 KL 单调链保证了这个 pipeline 每一步都不会破坏 transport 合法性，而它对 σ（熵正则强度）的敏感性分析也预警了我们：把随机性调得太小去追求确定性 OT 式的几何保持时，训练难度会陡增（σ=0.2 退化、σ=0 失败），这为 SB-Render-Lite 的 realism-vs-geometry 旋钮设定了理论上有依据的下界。

---

## venue 核验记录（检索日期 2026-08-14）

| 论文 | arXiv | venue | 状态 |
| --- | --- | --- | --- |
| DDBM | 2309.16948 | ICLR 2024 poster（主会） | 任务已核验 + OpenReview/proceedings 复核 ✓ |
| IDBM | 2304.00917 | JMLR 24(374):1–51, 2023（期刊） | 任务已核验 + jmlr.org 复核 ✓ |
| DSBM | 2303.16852 | NeurIPS 2023（主会） | 本次 proceedings.neurips.cc 核验 ✓ |
| Rectified Flow | 2209.03003 | ICLR 2023 oral（主会） | 本次 iclr.cc/OpenReview 核验 ✓ |
| EDM | 2206.00364 | NeurIPS 2022 oral（主会，Outstanding Paper） | 本次 proceedings/NVIDIA 页核验 ✓ |
| DBIM | 2405.15885 | ICLR 2025 poster（主会） | 本次 OpenReview 核验 ✓ |
| CDBM | 2410.22637 | NeurIPS 2024（主会） | 本次 thu-ml 官方仓库声明核验（未查 proceedings 页，置信度略低） |
| I²SB | 2302.05872 | ICML 2023（主会） | 库内已核验（R05），直接沿用 |
| SB Flow | 2409.09347 | NeurIPS 2024（主会） | 库内已核验（R05），直接沿用 |

Pix2Pix/SDEdit/DDIB/BBDM 等仅作为 DDBM 实验基线名称出现，本报告不引用其 venue 声明。

## 并入主库建议

1. **新增两份正式报告**：将本文件 Part A 摘出为 `reports/2309.16948_ddbm.md`（沿库内模板），Part B 摘出为 `reports/2304.00917_idbm_theory_note.md`（标注为"理论笔记"而非全文精读）。
2. **INDEX.md 归档位置**：DDBM 放入「SB 图像、科学数据与确定性 OT 应用」组、紧邻 I²SB；IDBM 放入「重要对照 / SB 方法支撑」组、紧邻 SB Flow，并在综合入口注明它是 bridge matching/IMF 的理论源头。
3. **修订库内既有口径**：`synthesis.md` 与 `INDEX.md` 尾注中「I²SB 是 paired restoration 强基线」应更新为「paired 基线双雄 = I²SB + DDBM(VP)；pixel 空间 DDBM 占优、latent 空间 I²SB/RF 占优」；I²SB 报告的「与 unpaired SB 方法的区别」一节可补一句 IDBM/DSBM 的第一迭代解释。
4. **SB-Render-Lite 实验清单落地项**：(a) paired 赛道加 DDBM(VP) 基线（官方权重可直接热启动 Edges2Handbags/DIODE 架构）；(b) 采样侧引入 DBIM 做 NFE 压缩；(c) 消融表加 hybrid 采样器 s∈{0,0.3} 与 guidance w 两行；(d) 若做 latent 管线，先复现 DDBM latent 负结果再决定是否投入重调预条件。
5. **后续扩充候选**（不在本次范围）：DBIM（2405.15885）与 CDBM（2410.22637）作为 DDBM 部署侧加速的精读候选；Peluchetti 的前作 "Non-denoising forward-time diffusions"（DBM 首次提出处，未正式发表）仅需在 IDBM 笔记中保留一句出处说明，无需精读。
