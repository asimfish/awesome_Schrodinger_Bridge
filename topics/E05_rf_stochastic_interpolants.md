# E05 扩充报告：Rectified Flow 与 Stochastic Interpolants（附 FM ↔ RF ↔ SI ↔ SB 理论桥）

- 研究员：E05（文献扩充）
- 日期 / 检索日期：2026-08-14
- 选题来源：内部审查 R09 缺口 G2（FM / rectified flow / stochastic interpolants 基础线），扩充选题 5
- 本报告产出：2 篇逐篇精读（arXiv 2209.03003、arXiv 2303.08797）+ 1 页 "FM ↔ RF ↔ SI ↔ SB" 理论桥笔记
- 约束遵守：未修改任何现有文件；库内已有 25 篇精读均只引用、不重复精读

## 选题定位与 TL;DR

**定位**。主库 48+ 条覆盖了 SB 求解器与 OT 模仿学习两条线，但方法地基（simulation-free 的 flow/interpolant 训练范式）此前只以学习资源链接存在，没有可引用的精读。Rectified Flow（RF）与 Stochastic Interpolants（SI）分别是这条地基的"工程极简端"与"理论统一端"：RF 给出直线化 + reflow + 蒸馏的一步推理路线（对 `SB-Render-Lite` 的部署加速直接相关），SI 给出把 flow、diffusion、bridge、SB 放进同一个插值框架的数学语言（写 preliminaries 必引）。二者与库内 SB Flow（2409.09347）、GSBM（2310.02233）、I²SB（2302.05872）构成完整的方法谱系。

**TL;DR（3 条）**

1. **RF 证明了"直线化"与"最优传输"是两回事**：rectification 把任意耦合变成对**所有凸代价同时不增**的确定性耦合（Pareto 下降），reflow 以 `O(1/K)` 速率把路径拉直到可一步 Euler 模拟；但 `d≥2` 时 straight coupling 不唯一、也不是任何固定代价 `c` 的最优耦合（1 维例外）。想要 OT 需给速度场加梯度场约束（c-rectified flow, arXiv 2209.14577）。
2. **SI 把"边缘路径的设计"与"采样过程的选择"解耦**：任意插值 `x_t = I(t,x_0,x_1) + γ(t)z` 的边缘密度同时满足一个输运方程和一族可调扩散系数 `ε(t)` 的前/后向 Fokker–Planck 方程，速度 `b` 与 score `s` 都由平方目标可学；**SDE 采样的 KL 由 drift 误差直接控制（`≤ ΔL_b/(2ε) + εΔL_s/2`），而 ODE 采样还需 Fisher 散度控制**——这是"同边缘下 SDE 比 ODE 对估计误差更鲁棒"的定理化表述，也是 `SB-Render-Lite` 做 ODE/SDE 消融的理论依据。
3. **SB 是 SI 家族中被变分挑出的那一个成员**：固定插值 `I` 时任何 SI 都已无偏地桥接 `ρ_0,ρ_1`；对 `I` 再做 max–min 优化（Thm 41）即恢复 Schrödinger bridge；`ε→0` 退化为 Benamou–Brenier 动态 OT。RF 的 reflow 与 DSBM 的 IMF 迭代是同构的"耦合迭代"，但前者（γ=0）不动点是非唯一的 straight 耦合，后者（Brownian-bridge 噪声）不动点是唯一的 SB。venue 复核：RF = **ICLR 2023 Oral**；SI = **JMLR 26(209):1–80, 2025 正式发表**（比库内 R09 记录的"arXiv 长文预印本"状态更新）。

---

## 精读 1：Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow

### 基本信息

- 论文：Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow
- 方法名：Rectified Flow（RF）；迭代过程称 reflow / rectification
- 作者：Xingchao Liu*, Chengyue Gong*, Qiang Liu（UT Austin；*同等贡献）
- 会议：**ICLR 2023（Oral）**。venue 复核（检索日期 2026-08-14）：iclr.cc 虚拟会议页 `virtual/2023/oral/12626` 列为 Oral；OpenReview id `gWxpdtQpiYV`
- 链接：https://arxiv.org/abs/2209.03003
- 代码：https://github.com/gnobitab/RectifiedFlow
- 全文获取方式：arXiv abs + arXiv 官方 HTML 全文（本次精读基于全文）
- 归类：simulation-free ODE 训练；直线化与一步生成；unpaired domain transfer / domain adaptation

### 一句话总结

RF 把"生成"与"域迁移"统一为同一个传输映射问题：对任意端点耦合 `(X_0,X_1)`，用最小二乘回归学 ODE 漂移去尽量贴合线性插值方向 `X_1−X_0`；所得流保边缘、对所有凸传输代价不增，且递归 reflow 把轨迹拉直到单步 Euler 即可采样。

### 动机与解决的问题

连续时间模型（diffusion/PF-ODE）当时的两大痛点：(1) 推理需要几十上百次网络调用；(2) 生成建模与域迁移被当成两类问题分开处理（域迁移要额外造 CycleGAN/SDEdit 式方法）。RF 主张：这两件事都是"在 `π_0` 与 `π_1` 之间找传输映射"，而且**学 ODE 根本不需要绕道 SDE/score 理论**——直接指定插值路径、回归其速度即可。`π_0` 取高斯就是生成，取源域数据就是翻译/域适应，算法完全同一。

### 方法核心

**训练目标（rectification）**。给定耦合 `(X_0,X_1)`（无配对数据时取独立乘积耦合），令 `X_t = t X_1 + (1−t) X_0`，解

`min_v ∫_0^1 E[ ‖(X_1−X_0) − v(X_t, t)‖² ] dt`，最优解 `v*(x,t) = E[X_1−X_0 | X_t = x]`。

线性插值 `X_t` 本身是非因果过程（更新需要偷看终点 `X_1`），其路径可以交叉；ODE `dZ_t = v*(Z_t,t)dt` 的解路径不可交叉，于是 RF 在交叉点"重接线"（rewiring）：保持每个时空点的质量通量不变（边缘不变），但把随机耦合**因果化、马尔可夫化、确定化**为 `(Z_0,Z_1) = Rectify((X_0,X_1))`。

**reflow 与直线化**。递归应用 `Z^{k+1} = RectFlow((Z_0^k, Z_1^k))`。定义直线度 `S(Z) = ∫_0^1 E[‖(Z_1−Z_0) − Ż_t‖²]dt` 与交叉量 `V((X_0,X_1)) = ∫_0^1 E[‖X_1−X_0 − E[X_1−X_0|X_t]‖²]dt`，核心恒等式（取 `c=‖·‖²`）：

`E‖X_1−X_0‖² − E‖Z_1−Z_0‖² = S(Z) + V((X_0,X_1))`，

telescoping 得 `min_{k≤K} S(Z^k) ≤ E‖X_1−X_0‖²/K`：**每次 reflow 消耗的传输代价恰好兑换成直线度与非交叉度**。完全直的流满足无粘 Burgers 方程 `∂_t v + (∂_z v)v = 0`，单步 Euler 精确。

**蒸馏与 rectification 的区别**。蒸馏忠实拟合当前耦合 `(Z_0^k, Z_1^k)` 的一步映射（`t=0` 处的目标项）；rectification 产生**新**耦合（代价更低、路径更直）。论文建议 reflow 1–2 次后再蒸馏收尾，reflow 过多会累积 `v` 的估计误差。

**非线性扩展与 PF-ODE/DDIM 的统一（Prop 3.11）**。把线性插值换成任意可微曲线 `X_t`（例如 `X_t = α_t X_1 + β_t X_0`）仍保边缘，但不再保证降代价与直线化。VE/VP/sub-VP 的 PF-ODE 与 DDIM 恰是 `X_t = α_t X_1 + β_t ξ`（`ξ∼N(0,I)`）的特例；其指数型 `α_t`（OU 过程的遗产）造成路径弯曲 + 速度不均匀，是大步长下表现差的原因。结论：**曲线形状与初始分布可以自由解耦选择，缺省应选线性插值**。

### 主要理论结论

1. 边缘保持（Thm 3.3）：任意可微插值的 rectified flow 满足 `Law(Z_t) = Law(X_t)`，∀t。
2. 凸代价 Pareto 不增（Thm 3.5）：`E[c(Z_1−Z_0)] ≤ E[c(X_1−X_0)]` 对**所有**凸 `c` 同时成立（两次 Jensen）。
3. 不动点刻画（Thm 3.6）：straight ⟺ 线性插值路径不交叉（`V=0`）⟺ Rectify 不动点。
4. `O(1/K)` 直线化速率（Thm 3.7）。
5. straight 与 optimal 的关系（Thm 3.8/3.10）：`c`-最优（严格凸 `c`）⇒ straight；1 维时 straight ⟺ 单调耦合 ⟺ 对全体凸代价联合最优且唯一；`d≥2` 时 straight 不动点不唯一、一般非 `c`-最优。把 `v` 约束为梯度场 `v=∇f` 后不动点为二次代价 OT（c-rectified flow, arXiv 2209.14577）。

### 实验与结果

- **CIFAR-10 生成（DDPM++ 同架构对比）**：1-RF 全 RK45 求解 FID 2.58 / recall 0.57，同架构 ODE 类中最优（VP ODE 3.93）；且 NFE 更少（127 vs 140）。reflow 会牺牲全模拟指标（2-RF 3.36 / 3-RF 3.96）但大幅改善小步数区间。
- **一步生成（N=1）**：2-RF + 蒸馏 FID **4.85**（当时 U-Net 类一步扩散/流模型最优，此前最好为 TDPM 8.91）；3-RF + 蒸馏 recall 0.51，超过 StyleGAN2+ADA 的 0.49。未蒸馏的 VP ODE 一步 FID 451（完全崩坏），说明直线化（而非蒸馏本身）是关键前提。
- **高分辨率生成**：256² 的 LSUN Bedroom/Church、CelebA-HQ、AFHQ Cat 定性结果良好。
- **无配对图像翻译（512²，AFHQ/MetFace/CelebA-HQ 两两）**：同一算法直接以 `π_0`=源域、`π_1`=目标域训练；无对抗、无 cycle 一致性正则（ODE 可逆自动给出双向映射）。为保主体身份，用域分类器隐层特征 `h` 的雅可比加权损失 `E‖∇h(X_t)^T (X_1−X_0 − v)‖²`（saliency 重加权，明确改变了任务：风格迁移而非严格采样 `π_1`）。
- **域适应**：在预训练模型倒数第二层 latent 上学 RF 把测试域搬到训练域，OfficeHome 69.2±0.5（超 CORAL 68.7）、DomainNet 41.4±0.1（持平 CORAL 41.5），DomainBed 协议。

### 局限性

- reflow 的每一轮都在**自生成数据**上训练：`v` 学不准时 `Z_1` 的分布已偏离 `π_1`，偏差随轮数复利累积（SI §5.3 对此给出精确批评：无偏 rectification 应保持映射不变，见精读 2）。
- straight ≠ OT（d≥2）：reflow 收敛到哪个 straight 耦合依赖初始化与估计误差，无 DSBM 那样的唯一不动点；对"耦合语义质量"敏感的 sim2real 任务（避免物体/类别错配）无原生保证。
- 纯 ODE 无 score：不能像 SI/SB 那样用 `ε>0` 的 SDE 换取对 drift 误差的鲁棒性，likelihood 控制弱（见 SI Lemma 21）。
- 域适应实验只在 latent 空间、分类 benchmark 上验证；图像翻译加权损失牺牲了分布保真的可检验性（无 FID 报告）。

### 与库内工作的关系

- **GSBM（2310.02233，已精读）**：GSBM 论文表 1 把 RF 作为其条件优化问题在 `V_t ≡ 0`、噪声 `σ→0` 下的特例（DSBM 为 `σ>0`），本报告与该口径互相印证。
- **SB Flow（2409.09347，已精读）**：α-DSBM 的 IMF 迭代与 reflow 结构同构（拟合桥 → 更新耦合 → 重训），差别仅在条件桥是否带 Brownian-bridge 噪声；见理论桥笔记第 4 条。
- **对 `SB-Render-Lite`**：(a) RF 是 unpaired sim→real 翻译中"最便宜的可跑基线"（单网络、无对抗、可双向）；(b) reflow+蒸馏是把 bridge 类模型压到 1–2 NFE 的部署路线（对应 R09 缺口 G10 部署加速）；(c) latent 域适应实验直接支持"在 policy encoder 特征上做流迁移"的轻量方案（可与 EgoBridge 的 latent OT 对齐做对照）；(d) `∇h` 加权损失是"任务相关保持项"的最简形式，可视作 GSBM 状态代价 `V_t` 的一阶近似消融。

---

## 精读 2：Stochastic Interpolants: A Unifying Framework for Flows and Diffusions

### 基本信息

- 论文：Stochastic Interpolants: A Unifying Framework for Flows and Diffusions
- 方法名：Stochastic Interpolants（SI）
- 作者：Michael S. Albergo*, Nicholas M. Boffi*, Eric Vanden-Eijnden（NYU；字母序、同等贡献）
- 发表：**JMLR 26(209):1–80, 2025**（Submitted 12/23; Revised 9/25; Published 9/25；编辑 Maxim Raginsky）。venue 复核（检索日期 2026-08-14）：jmlr.org `papers/v26/23-1605.html` 确认；arXiv 最新版已同步 JMLR 版式。**注意：库内 R09/学习资源导航记录的"arXiv 2023 长文预印本"状态已过期**。前身短文 Albergo & Vanden-Eijnden, Building Normalizing Flows with Stochastic Interpolants（arXiv 2209.15571）为 ICLR 2023，两篇应双列引用（R07 已指出学习资源表的标题-链接错配）
- 链接：https://arxiv.org/abs/2303.08797
- 全文获取方式：arXiv abs + arXiv 官方 HTML 全文（本次精读基于全文，80 页刊出版）
- 归类：flow/diffusion 统一框架；两侧任意分布的有限时间精确桥接；likelihood 理论；SB 变分刻画

### 一句话总结

SI 用一个带潜变量的插值过程 `x_t = I(t,x_0,x_1) + γ(t)z` 把"设计边缘密度路径"与"选择采样动力学"彻底解耦：同一条密度路径可用 ODE 或任意噪声强度的 SDE 无偏采样，全部漂移由平方回归可学，并在此框架内统一了 score diffusion、denoising、rectified flow，且证明对插值函数再优化即得 Schrödinger bridge。

### 动机与解决的问题

SBDM 的三个结构性缺陷：必须以高斯为一端、OU 过程只在无穷时间才到达先验（有限截断引入偏差）、路径设计与采样机制耦合在 SDE 推导里难以拆解调优。SI 要给出：两端任意、有限时间**精确**到达、可自由加噪的统一框架，并回答一个悬而未决的问题——**同样边缘下，确定性与随机性生成模型的差距到底在哪**。

### 方法核心

**two-sided 插值（Def 1）**。`x_t = I(t,x_0,x_1) + γ(t)z`，其中 `I(0)=x_0, I(1)=x_1`，`γ(0)=γ(1)=0, γ>0` on (0,1)，`z∼N(0,Id)` 独立于端点对 `(x_0,x_1)∼ν`（ν 可以是独立乘积，也可以带耦合）。范例：`x_t=(1−t)x_0+t x_1+√(2t(1−t)) z`。潜变量 `γz` 的作用：(i) 保证密度 `ρ(t)` 与漂移的空间正则性；(ii) 抹平中间时刻的伪模态（两端多模态直接线性插值会在中间叠出虚假 mode，γ>0 平滑之）；(iii) 让 score 可学。

**漂移–score 分解**。`ρ(t)` 满足输运方程 `∂_t ρ + ∇·(bρ) = 0`，其中

`b(t,x) = E[∂_t I + γ̇ z | x_t=x] = v(t,x) − γ(t)γ̇(t) s(t,x)`，
`v(t,x) = E[∂_t I | x_t=x]`（插值速度），`s(t,x) = ∇log ρ(t,x) = −γ(t)^{−1} E[z | x_t=x]`（score）。

`b、v、s` 各自都是简单平方目标的唯一最小元；实践推荐学 **denoiser** `η_z(t,x)=E[z|x_t=x]`（目标无 `γ^{−1}` 因子，端点数值稳定），再由 `s=−η_z/γ` 还原 score。线性插值 `x_t=α x_0+β x_1+γ z` 时进一步分解为三个条件期望 `η_0,η_1,η_z`，且受恒等约束 `αη_0+βη_1+γη_z = x`（学两个可推第三个）。

**一条密度路径、一族采样器（Cor 10/18）**。对任意 `ε(t)≥0`，`ρ(t)` 同时满足前向 FPE（漂移 `b_F = b + εs`）与后向 FPE（漂移 `b_B = b − εs`）。因此同一组 `(b̂, ŝ)` 可组装出：概率流 ODE（`ε=0`，可逆、可算 likelihood）、前向 SDE、后向 SDE，边缘全同。**边缘路径的设计（选 I、γ）与采样噪声水平（选 ε）完全正交**——这是对 SBDM"把两件事焊死在 OU 推导里"的根本改进；SBDM 本身经时间重参数 `t=e^{−τ}` 化为 one-sided 线性插值 `α=√(1−t²), β=t` 的特例，且顺带消除了有限截断偏差与 `t=0` 奇异性（§5.1）。

**one-sided / mirror / diffusive 实例化**。one-sided（§3.2）：`x_t = α(t)z + J(t,x_1)`，高斯端与潜变量合并，覆盖 FM/SBDM 的生成设定；mirror（§3.3）：`ρ_0=ρ_1`，配 SDE 得"围绕数据的重采样/增广"；diffusive（§3.1）：`I + √(2a)B_t`（Brownian bridge）与 `γ=√(2a t(1−t))` 的 SI 单时刻分布相同，从而 **SI 覆盖 stochastic bridge 类方法但绕开 Doob h-transform**；推论：存在从单点 `x_0` 出发采样 `ρ_1` 的非奇异条件 SDE（Thm 31，ODE 原则上做不到）。

**likelihood 控制（§2.4，本文最重要的定理组）**。两条输运方程之间：`KL(ρ(1)‖ρ̂(1)) = ∫∫ (∇log ρ̂−∇log ρ)·(b̂−b) ρ`——**drift 误差小不保证 KL 小**（还需 Fisher 散度控制）；两条 FPE 之间：扩散项贡献负的 Fisher 项，得 `KL ≤ (1/4ε)∫∫ |b̂_F−b_F|² ρ`。汇总为（Thm 23）：

`KL(ρ_1 ‖ ρ̂(1)) ≤ (1/2ε)·ΔL_b + (ε/2)·ΔL_s`（ΔL 为目标函数与最优值之差），

最优噪声 `ε* = √(ΔL_b/ΔL_s)`：**score 学得比 drift 好就该多加噪，反之少加**。§2.5 另给出 SDE 模型的 likelihood/交叉熵估计式（ODE 变量变换公式的随机对应物）。

**SB 变分刻画（§3.4 Thm 41）**。SB 的流体力学形式 `min ∫E|u|²ρ s.t. ∂_tρ+∇·(uρ)=εΔρ, ρ(0)=ρ_0, ρ(1)=ρ_1` 可由 SI 上的 max–min 解出：

`max_I min_u ∫_0^1 E[ ½|u(t,x̂_t)|² − (∂_t I + (γ̇ − εγ^{−1})z)·u(t,x̂_t) ] dt`，

内层最优 `u` 即该插值的前向漂移 `b_F`；外层对 `I` 最大化后，`x_t` 的密度恰为 SB 解，`u=∇λ` 满足 SB 的 Euler–Lagrange 方程组。`ε→0` 时形式上退化为（Benamou–Brenier）OT。关键定位：**固定 I 已是无偏生成模型，优化 I 只是"锦上添花"地挑出最优传输那一个**；数值实现留作未来工作（Assumption 39 需要 `ρ(t)` 可逆映到高斯，较强）。

**对 rectification 的再审视（§5.3 Thm 47）**。若速度 `b` 已学准、流映射 `X_{1}` 可得，则用 `x_t^rec = α(t)x_0 + β(t)X_1(x_0)` 重训得到的新概率流解恰为 `X_t^rec(x)=α(t)x+β(t)X_1(x)`：**路径被拉直成一步可算，但端点映射（耦合）不变**——无偏的直线化不改变生成模型本身。RF 论文中 reflow 改变耦合、降低传输代价的现象，来自初始插值路径的交叉（非确定性耦合）以及实际的估计误差；且若 `b` 未学准，reflow 桥接的是 `ρ_0` 与 `X_1♯ρ_0 ≠ ρ_1`，偏差逐轮累积。Remark 48/49 重申：直线解是 OT 的必要非充分条件；对 `b^rec` 施加梯度场结构并迭代才收敛到 Brenier 映射。

### 主要结论（含数值）

- 2D checkerboard：同一组 `(b̂,ŝ)` 下 SDE 采样普遍优于 ODE；`γ=√(t(1−t))` 时 ODE 差距最小——潜变量选得好能显著缩小确定性/随机性差距。
- 128 维 5-mode 高斯混合（有解析真值）：四种参数化组合中**学 `(b, η_z)` + 调 `ε>0` 最优**，且最优 `ε` 严格非零；`ε` 过小高估 mode 密度、过大高估尾部。
- Oxford Flowers 128²：one-sided 与 mirror 插值均可扩展到图像；mirror + SDE（ε=10）实现"在数据近旁重采样"。定性验证为主，未报告 FID（规模化由后续 SiT, arXiv 2401.08740, ECCV 2024 完成）。

### 局限性

- 图像实验无定量 benchmark，"框架论文"属性明确；工程配方（架构、噪声调度细节）需到 SiT/后续工作找。
- SB 恢复是存在性/变分刻画：max–min 的数值算法与 Assumption 39（可逆高斯化映射）都未落地；实际解 SB 仍需 DSBM/IMF 类算法（库内 SB Flow）。
- score 目标在端点的 `γ^{−1}` 奇异性需 denoiser 参数化 + `t∈[t_0, 1−t_0]` 截断处理；`ε` 大时 SDE 积分步数开销上升。
- 潜变量 `γz` 与扩散 `ε` 的最优联合设计只有 2D/GMM 级别的经验结论，高维视觉域无系统消融。

### 与库内工作的关系

- **为 R09-G2 点名的"FM↔SB 理论桥"提供正主**：SI Thm 41 就是"对 interpolant 显式优化可恢复 SB"的原始出处；`SB-Render-Lite` 论文 preliminaries 引用链建议为 FM(2210.02747) + SI(本篇, JMLR 2025) + 短文(2209.15571, ICLR 2023)。
- **SB Flow（2409.09347）/ I²SB（2302.05872）**：SI 的 diffusive interpolant 等价性说明 Brownian-bridge 桥接类方法（含 I²SB 的 paired 桥、DSBM 的 reciprocal 投影）都能在 SI 语言里写成 `γ=√(2εt(1−t))` 的特例；IMF 的"reciprocal class"投影即固定这一 γ 形状、只更新耦合。
- **3MSBM（2506.10168）**：其相关工作已把 SI 当作统一插值框架引用，本精读补齐正式出处与定理编号。
- **对 `SB-Render-Lite` 的直接价值**：(a) `ε*=√(ΔL_b/ΔL_s)` 给了 ODE/SDE 消融一个**可测的调参准则**（分别监控 drift/score 的验证损失差）；(b) 真实域传感噪声大、光照多模态时，γ>0 + SDE 的鲁棒性定理支持优先选随机采样器；(c) mirror interpolant 是"真实数据近旁增广"的现成工具，可做 real2real 一致性正则；(d) denoiser 参数化 + 端点截断是所有 bridge 训练共享的工程要点。

---

## 理论桥笔记（一页）：FM ↔ RF ↔ SI ↔ SB

**统一骨架**。四者都在学"边缘密度路径 `ρ(t)`（`ρ(0)=ρ_0, ρ(1)=ρ_1`）+ 生成该路径的动力学"，且都用 simulation-free 的条件回归：先指定逐样本条件路径，再回归其条件速度的后验均值。差异只在三个设计轴：**① 条件路径形状（线性/非线性、γ 噪声）；② 端点耦合 ν（独立 / minibatch-OT / 迭代更新）；③ 采样动力学（ODE ε=0 / SDE ε>0）**。

**各方法的规范形式**

- **FM**（Lipman et al., arXiv 2210.02747, ICLR 2023）：高斯条件路径 `p_t(x|x_1)=N(μ_t(x_1), σ_t²Id)`（"OT path"：`μ_t=t x_1, σ_t=1−(1−σ_min)t`），回归 `min_v E‖v(x_t,t) − u_t(x_t|x_1)‖²`；边缘速度 `u_t(x)=E[u_t(x_t|x_1)|x_t=x]`。原始形式一端为高斯；广义 CFM（OT-CFM, arXiv 2302.00482, TMLR 2024）把条件路径挂到任意耦合 `q(x_0,x_1)` 上。
- **RF**（arXiv 2209.03003, ICLR 2023 Oral）：`x_t=t x_1+(1−t)x_0`，`v*(x,t)=E[x_1−x_0|x_t=x]`；reflow 迭代 `(Z_0^{k+1},Z_1^{k+1}) = Rectify((Z_0^k,Z_1^k))`。
- **SI**（arXiv 2303.08797, JMLR 2025）：`x_t=I(t,x_0,x_1)+γ(t)z`；`b=v−γγ̇s`，`s=−γ^{−1}E[z|x_t]`；采样器族 `dX = (b±εs)dt + √(2ε)dW`，∀ε≥0 边缘同。
- **SB(ε)**：动态形式 `min_{u,ρ} ∫_0^1∫ |u|²ρ dx dt s.t. ∂_tρ+∇·(uρ)=εΔρ`，两端固定；静态等价形式为 entropic OT：`min_{π∈Π(ρ_0,ρ_1)} E_π[½|x_1−x_0|²] + 2ε·KL(π‖ρ_0⊗ρ_1)`（参考测度为生成元 εΔ 的布朗运动；Léonard 2014 口径）。其解的边缘路径 = 在 entropic-OT 最优耦合 `π*_ε` 上混合 Brownian bridge：`ρ^SB(t)=∫ N(x; (1−t)x_0+t x_1, 2εt(1−t)Id) dπ*_ε`。

**关系（特例 / 极限 / 等价）**

1. **RF = SI 的 γ≡0、线性 I 特例**；代价是失去 score、失去 SDE 采样与 KL-by-drift 控制（SI Lemma 21/22）。
2. **FM(独立耦合, σ_min→0) = 1-RF(高斯端) = one-sided 线性 SI**：三种写法给出同一个边缘速度回归；FM 的 `σ_min>0` 相当于终端噪声卷积（轻微偏置目标分布）。广义 CFM（任意耦合 + 线性路径）逐字等于 RF 的第一次 rectification。
3. **SBDM/PF-ODE/DDIM = 非线性单侧特例**：RF Prop 3.11（`x_t=α_t x_1+β_t ξ`）与 SI §5.1（`t=e^{−τ}` 时间重参数）从两个方向证明同一件事；SI 版本还消除了有限时间截断偏差。
4. **SB 与迭代法**：DSBM/IMF（含库内 SB Flow 的 α-DSBM）交替做 reciprocal 投影（固定 Brownian-bridge 条件桥，即 SI 取 `γ=√(2εt(1−t))`）与 Markov 投影（更新耦合），**唯一不动点 = SB(ε)**；RF 的 reflow 是其 γ=0 类比，不动点 = straight 耦合的集合（`d≥2` 非唯一、一般非 OT）。等价口径：**SB = "被熵正则挑出的唯一 reflow 不动点"**。
5. **SI ⊃ SB（变分）**：固定 `I` 的 SI 都是 `ρ_0↔ρ_1` 的无偏桥；对 `I` 做 max–min（SI Thm 41，内层解为 `b_F`）恢复 SB(ε)。
6. **entropic 极限链**：`SB(ε) --ε→0--> 动态 OT（Benamou–Brenier）`；同时 `γ=√(2εt(1−t)) --ε→0--> γ≡0`（SI→RF 的插值退化）、`π*_ε → π*_OT`。反向 `ε→∞` 时 `π*_ε → ρ_0⊗ρ_1`（独立耦合）；实践中 IMF/DSBM 正是以"独立耦合 + Brownian bridge"（即 reciprocal 类的第 0 步）初始化，再逐轮把耦合从独立端推向 `π*_ε`。
7. **直线化 ≠ 最优化**：straight 是 c-最优的必要条件（RF Thm 3.8；SI Remark 48）；补上梯度场约束 `v=∇f` 并迭代才到二次代价 OT（c-RF, arXiv 2209.14577；= Brenier 极分解）。且无偏 rectification 不改变端点映射，只把路径变直（SI Thm 47）——"加速"与"改耦合"必须分开归因。

**sim2real 数据情形 → 方法选型**

| 数据情形 | 首选 | 理由（对应上面第几条） |
|---|---|---|
| 有配对 (sim, real) 渲染对 | 条件 FM/RF；I²SB | 耦合已知，无需熵正则挑耦合（2） |
| 无配对、域差小、要快 | RF / γ=0 SI + minibatch-OT 耦合 | iteration-0 即可用；OT-CFM 改善配对（2,6） |
| 无配对、域差大、怕语义错配 | DSBM/SB Flow（必要时 GSBM 加 `V_t`） | 唯一 SB 不动点给出有原则的耦合（4） |
| real 端噪声大 / 多模态（光照、传感器） | SI γ>0 + SDE，按 `ε*=√(ΔL_b/ΔL_s)` 调噪 | KL-by-drift 鲁棒性只在 ε>0 成立（1,5） |
| 机器人闭环、1–2 NFE 预算 | reflow+蒸馏（RF）或无偏直线化（SI Thm 47） | 直线化独立于耦合选择，可后置（7） |
| 只有 reward/能量、无 real 样本 | 库内 adjoint 线（AS/ASBS） | 超出本笔记范围，见 `sb_adjoint_extended_synthesis.md` |

---

## venue 复核记录（检索日期 2026-08-14）

| 论文 | 复核结论 | 证据 |
|---|---|---|
| Rectified Flow (2209.03003) | ICLR 2023 **Oral** | iclr.cc `virtual/2023/oral/12626`；OpenReview `gWxpdtQpiYV`；代码 gnobitab/RectifiedFlow |
| Stochastic Interpolants (2303.08797) | **JMLR 26(209):1–80, 2025**（Submitted 12/23; Published 9/25；编辑 Maxim Raginsky） | jmlr.org `papers/v26/23-1605.html` 及 PDF 首页；arXiv 最新版含 JMLR 版式头 |
| （引文核对）c-Rectified Flow | arXiv:2209.14577（Qiang Liu, 2022, preprint） | arxiv.org/abs/2209.14577 |
| （引文核对）SI 前身短文 | Building Normalizing Flows with Stochastic Interpolants, arXiv 2209.15571, ICLR 2023 | 与 R07 复核结论一致（OpenReview `li7qeBbCR1t`） |

## 并入主库建议

1. **INDEX 归类**：建议在 `reports/INDEX.md` 新开小节"方法地基：Flow / Interpolant 训练范式"，收录本报告两篇精读；与既有"重要对照 / SB 方法支撑"小节（SB Flow）互链。
2. **venue 信息更新**（由维护者执行，本报告未改动任何现有文件）：(a) R09/学习资源导航中 SI 的状态应从"arXiv 2023 预印本"更新为 **JMLR 26(209), 2025**；(b) 学习资源表 §4.4 按 R07 建议双列 SI 短文（ICLR 2023）与框架长文（JMLR 2025）。
3. **synthesis 挂接点**：`sb_adjoint_extended_synthesis.md` 第一阶段实验设计中的"OT-ODE / flow matching deterministic 对照"现在有了可引用的精读支撑；建议把本报告理论桥第 4、6 条（reflow vs IMF、ε 极限链）纳入其方法谱系图；GSBM 报告的"表 1 特例"口径与本报告互引。
4. **实验落地转化**（供 `SB-Render-Lite` 计划表选用）：(a) 增加 RF 基线（unpaired sim→real，1-RF → reflow → 蒸馏三档 NFE 消融）；(b) ODE vs SDE 消融按 `ε* = √(ΔL_b/ΔL_s)` 设定噪声档位并报告 drift/score 验证损失差；(c) 把"直线化加速"与"耦合改进"分开归因（SI Thm 47 口径），避免把 reflow 的收益错记到耦合质量上；(d) mirror interpolant 作为 real 域数据增广的低成本对照组。
5. **后续扩充衔接**：本报告未精读的 FM 原文（2210.02747）与 minibatch 耦合两篇（2302.00482、2304.14772）属选题 4（E04）范围；SiT（2401.08740, ECCV 2024）可作为 SI 规模化的导航条目补入。
