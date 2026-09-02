# E20：SB 逆问题 × Trajectory Inference 横断综述

> 扩充研究员：E20｜日期：2026-08-14｜文献检索/venue 核验日期：2026-08-14
> 产出：1 篇精读（CDSB）+ 3 条收录（DPS / DDRM / SDA）+ 1 节 trajectory inference 综述（WOT / TrajectoryNet / MIOFlow / CellOT）+ held-out marginal 评测协议摘录

## 选题定位

主库已有的 SB 逆问题样例（`../reports/2308.12351_sb_unfold.md`，SBUnfold）覆盖了"paired bridge 做科学测量反演"这一种形态。本报告补上两块缺口：

1. **条件模拟的 SB 理论形态**：CDSB 把"给定观测 y 采样后验 p(x|y)"写成扩展空间上的 Schrödinger Bridge，是 SB-Render-Lite"给定仿真观测生成真实观测"这一条件翻译需求的最直接理论对应——它回答了"条件 bridge 应该怎么构造、和无条件 DSB 差在哪、什么时候比 zero-shot guidance 好"。
2. **Trajectory inference 家族及其评测纪律**：WOT/TrajectoryNet/MIOFlow/CellOT 处理"只有若干时间截面的边缘分布、无逐点对应"的推断问题，与 sim2real 的"只有仿真域与真实域两个（或多个）边缘、无配对"结构同型。该社区沉淀出的 **held-out marginal（留出边缘）协议**是 SB/OT 类方法验证泛化的成熟做法，值得直接移植为 SB-Render-Lite 的验证协议。

## TL;DR

- **CDSB（UAI 2022）**把条件模拟写成 (x,y) 扩展空间上的 SB：y 分量沿扩散保持不动（delta 转移核），x 分量从 p(x|y) 扩散到 p_ref；用 amortized 目标绕开"IPF 需要从后验采样"的死结，只要求能从联合分布 p_data(x)g(y|x) 采样。首次把 score 类方法用于 state-space model 的 optimal filtering。对 SB-Render-Lite 最有价值的两个构件：**条件参考测度 p_ref(x|y)**（从"y 的粗翻译"而非噪声出发，N=5–50 步即可）与 **forward-backward 采样**（误差对消）。
- **逆问题家族的分工线**：DPS/DDRM 是"无条件先验 + 推理时注入退化模型"的 zero-shot 路线（DDRM 限线性可 SVD、DPS 泛化到非线性但靠 Jensen gap 不可控的近似）；SDA 把该路线推广到时序状态轨迹；CDSB/I²SB/SBUnfold 是"训练时学条件 bridge"路线。**判据是：forward operator 是否已知可微、是否有成对样本、推理预算、y^obs 是否会偏离训练分布**。sim2real 渲染差距没有解析 forward operator，但仿真天然提供成对/可采样的联合分布，因此落在"训条件 bridge"一侧。
- **held-out marginal 协议**（WOT 首创、TrajectoryNet/MIOFlow 标准化）：扣掉一个中间时间点的全部数据，训练后预测该时刻分布，用 EMD/W1 + MMD 对照真实留出分布，配 prev/next/random/McCann-interpolant 四类廉价 baseline。移植到 sim2real：把"仿真→真实"展成多档中间边缘（渲染保真度/DR 强度/混合比例），做 leave-one-domain-out；注意**边缘匹配≠条件正确**（SBUnfold 教训），须与配对指标和下游 policy 成功率并用。

---

## 一、精读：CDSB — Conditional Simulation Using Diffusion Schrödinger Bridges

### 元信息

- 标题：Conditional Simulation Using Diffusion Schrödinger Bridges
- 作者：Yuyang Shi, Valentin De Bortoli, George Deligiannidis, Arnaud Doucet（Oxford Statistics / ENS Paris）
- arXiv：[2202.13460](https://arxiv.org/abs/2202.13460)；venue：**UAI 2022, PMLR v180**（任务给定已核验；本次未重复查证）
- 代码：https://github.com/vdeborto/cdsb
- 全文获取方式：arXiv HTML（ar5iv 镜像同版），2026-08-14 抓取，正文+附录完整可读
- 谱系：De Bortoli et al. 2021 的 DSB（NeurIPS 2021）的条件化；一作 Shi 后续与 De Bortoli 合作 DSBM（bridge matching 路线，主库 `2409.09347` SB Flow 的前驱之一）

### 动机

条件 SGM（CSGM，如 SR3 式做法）从纯噪声出发采样 p(x|y^obs)，必须把前向加噪扩散跑到接近参考分布，生成链条长；而 y 本身已携带大量关于 x 的信息，从噪声出发是浪费。无条件情形下 DSB 已证明：把生成建模写成 SB 问题（在 KL 意义下最接近前向加噪过程、且两端边缘被钉死的有限时间过程），可以在**有限、短**的时间内精确衔接 data 与 reference。CDSB 要把这套"缩短生成链"的理论框架搬到条件模拟/逆问题上。

关键障碍：naive 的条件 SB 应该以 π_0 = p(x|y^obs) 为端点约束，但 DSB 的 IPF 迭代每一轮都要**从 π_0 采样**——p(x|y^obs) 恰恰是我们想求的东西，此路不通。

### 方法核心

**1）Amortized 条件 SB（CSB）。** 不解单个 y^obs 的 SB，而是解在 Y ~ p_obs 上平均的核族 π^c = (π^c_y)：

min E_{Y~p_obs}[ KL(π^c_Y || p_Y) ]，s.t. π^c_0 ⊗ p_obs = p_join，π^c_N ⊗ p_obs = p_jref

其中 p_join(x,y) = p_data(x)g(y|x)，p_jref(x,y) = p_ref(x)p_obs(y)。约束保证 p_obs-几乎处处 π^c_{y,0} = p(x|y)、π^c_{y,N} = p_ref。

**2）等价为扩展空间上的标准 SB（Proposition 1）。** 在 (x_{0:N}, y_{0:N}) 上定义参考过程：y 分量在 t=0 从 p_obs 采样后**沿 delta 核保持恒定**，x 分量按原加噪核扩散。则上述 amortized 问题的解就是该扩展 SB 的解（π̄* = π^c,* ⊗ p̄_obs）。这样 SB 的存在唯一性理论直接适用，且两个端点分布 p_join、p_jref 都**可采样**——这正是 naive 形式缺失的性质。

**3）CDSB = 条件版 DSB/IPF。** IPF 半桥交替（钉 t=N 端 → 钉 t=0 端），每半步用广义 score matching 学习带 y 输入的高斯转移均值网络 B^y_θ(k+1, x)、F^y_φ(k, x)（Proposition 2 给出迭代的显式表示：y 在全链上几乎必然恒定，所以网络只需把 y 当额外条件输入）。**第一轮 CDSB 迭代恰好等于 CSGM**，后续迭代可理解为对 CSGM 的逐步精化——这给了"CSGM 是 CDSB 特例"的干净解释，也解释了为什么 N 小时 CDSB 增益最大（CSGM 依赖 p_N ≈ p_ref，短链下该假设破产，而 SB 不需要它）。

**与无条件 DSB 的区别（一句话版）**：DSB 解 p_data ↔ p_ref 的桥，IPF 两端都可采样；CDSB 把"后验端不可采样"问题转化为扩展空间上"联合分布端可采样"的桥，网络额外吃 y，训练数据只需联合样本 (X,Y)——**不需要 likelihood g(y|x) 的解析式，也不需要先验 p_data 的显式形式**（对 inpainting 这类隐式先验场景很关键，与 ABC 的对比也在此：ABC 需要显式 prior）。

**4）三个增强构件（对我们最有用的部分）：**

- **CDSB-C：条件参考测度 p_ref(x|y)**。把 p_jref 换成 p_ref(x|y)p_obs(y)：SR 任务直接用上采样后的 y 加方差膨胀高斯 N(x; y, ρσ²_{x|y}Id)；inpainting 用一个小网络输出初始化均值；滤波用 EnKF 的近似后验；甚至用预训练 SRFlow 的输出分布当 p_ref(x|y)。**桥不再是"后验 ↔ 噪声"而是"后验 ↔ 后验的粗近似"**，扩散只需修正残差。
- **条件前向过程 + 收敛速率（Proposition 3）**：IPF 第 n 轮迭代的 0 端边缘与真后验的期望 KL ≤ (2/n)·E[KL(π^c,*_Y | p_Y)]——初始前向过程越接近 CSB 解，收敛越快；据此建议用以 p_ref(·|y) 为不变分布的 Langevin/OU 前向核。
- **CDSB-FB：forward-backward 采样**。测试时先把联合样本 (X,Y) 沿前向半桥推到 X̂_N，再沿 backward 半桥在 y^obs 条件下拉回 X̂_0。前后两个半桥互为近似逆，**系统性误差部分对消**（继承 Spantini et al. 2022 确定性 transport 的技巧）。

### 实验

- **2D 合成（vs MGAN）**：三个非线性非高斯 p(x|y)，CDSB 用约 1/6 参数量取得更贴合真后验的直方图；多轮 IPF 明显修正单轮（=CSGM）的偏差；FB 采样进一步提升。
- **BOD 贝叶斯推断**：高偏度高峰度后验，CDSB/CDSB-FB/CDSB-C 的前四阶矩全面比 MGAN 与 inverse transport 更接近 6×10⁶ 步 MCMC 的金标准。
- **图像逆问题（短链是卖点）**：MNIST 4×SR 与 14×14 inpainting、CelebA 4×SR（含 σ=0.1 高斯噪声）与 32×32 inpainting。N=5–50 步。代表性数字：CelebA inpainting N=50，CSGM 25.29/0.878/7.18（PSNR/SSIM/FID）→ CDSB-C 28.06/0.914/**1.14**；N=20 时 CDSB-C 的 FID 2.28 已接近 N=50 水平，短链优势显著。
- **预训练模型精化**：用 SRFlow（τ=0.8）当 p_ref(x|y)，在 CelebA 160×160 8×SR 上仅 N=10 步把 FID 从 30.92 拉到 15.00（PSNR 略降）。证明 CDSB 可当"任意现成翻译器的 SB 后处理器"。
- **Lorenz-63 optimal filtering（首个 score 类滤波应用）**：序列式地在每个时刻解一个小 CSB（用解析基函数回归代替神经网络、5 轮 IPF）。短链 N=20 时 **CSGM/CSGM-C 直接发散**，CDSB-C RMSE 0.178±0.007（M=2000），显著优于 EnKF 的 0.354；长链 N=100 时 CSGM 才追回来。这是"短链下 SB 结构必要性"的最硬证据。

### 局限

- **N 的下界未知**：CDSB 数值近似 IPF 迭代产生的扩散，最小可用 N 取决于迭代漂移的陡峭程度，实践中无法先验知道。
- **y^obs 的典型性问题**：y 只在采样阶段进入，训练只见过 p_obs 下的 y；若 y^obs 非典型（协变量漂移），后验近似不可靠。作者指出 ABC 社区"在 y^obs 邻域采合成观测"的补救思路尚未搬过来。**这正是 sim2real 的软肋映射：真实观测相对仿真观测边缘就是"非典型 y"**。
- 滤波版本每个时刻要重解一个 SB（未 amortize）；条件 multi-marginal SB 也只是被点名为未来工作（可与主库 3MSBM `2506.10168` 衔接）。
- 图像实验只到 CelebA 160×160，未上大分辨率/真实退化。

### 对 SB-Render-Lite 的直接启示

1. **条件参考测度是免费午餐**：sim→real 翻译不必从噪声出发，可以用"仿真观测本身/廉价风格迁移器/低成本 colorization"当 p_ref(x|real 条件的粗近似)，CDSB-C 式短链修正。这与库内 I²SB（`2302.05872`）从退化图像出发一脉相承，但 CDSB 允许 p_ref 是**任意可采样分布**而非解析 bridge 端点，工程自由度更大。
2. **FB 采样可直接抄**：SB-Render-Lite 推理时可用 sim 图像沿 forward 半桥推到中间态再条件回拉，缓解 bridge 学不准的系统误差。
3. **CSGM=第一轮 IPF** 提供了干净的消融轴：训练预算允许时多跑 IPF 轮数，报告"轮数 vs 下游收益"曲线即可量化 SB 结构带来的净增益。
4. y^obs 非典型性局限提示：真实域观测应尽量参与 p_obs 的构造（哪怕无标签），否则条件 bridge 在真实输入上退化——这支持主库 synthesis 里"unpaired 真实数据也要进训练循环"的立场。

---

## 二、横断综述（a）：SB / 扩散逆问题家族

本节把"用生成先验解逆问题"的方法谱系整理为两条路线 + 一条时序扩展，并给出 SB-Render-Lite 视角的选型判据。三条收录条目均已 web 核验 venue（DBLP，检索 2026-08-14）。

### 谱系图

```
                     逆问题 x → y = A(x) + n，求 p(x|y)
                                   │
          ┌────────────────────────┴───────────────────────┐
   zero-shot guidance 路线                          训练条件模型路线
   （无条件先验 + 推理时注入 A）                     （训练时见 (x,y) 联合样本）
          │                                                │
   ┌──────┴──────┐                                  ┌──────┴──────┐
 线性、已知 SVD   一般（非线性）                    从噪声出发       从退化/近似出发
   DDRM          DPS                               CSGM(SR3 式)    I²SB（解析 paired 桥）
   （谱空间变分）  （Tweedie 后验均值近似）                          CDSB（IPF 条件桥）
          │                                                        SBUnfold（I²SB 用于 unfolding）
   时序状态轨迹扩展：SDA（分段 score + 推理期观测引导）
```

### 收录条目 1：DPS — Diffusion Posterior Sampling for General Noisy Inverse Problems

- arXiv：[2209.14687](https://arxiv.org/abs/2209.14687)；venue：**ICLR 2023**（DBLP `conf/iclr/ChungKMKY23`，OpenReview `OnD9zGAGT0k`；核验 2026-08-14）
- 作者：Hyungjin Chung, Jeongsol Kim, Michael T. McCann, Marc L. Klasky, Jong Chul Ye（KAIST / LANL）
- 代码：https://github.com/DPS2022/diffusion-posterior-sampling
- **方法定位**：纯 zero-shot。反向 SDE 的漂移里补 ∇log p_t(y|x_t)，用 Tweedie 公式的后验均值 x̂_0 = E[x_0|x_t]（由无条件 score 一步算出）做近似 p(y|x_t) ≈ p(y|x̂_0)，梯度经自动微分穿过 A。近似误差被量化为 Jensen gap（Theorem 1：上界随测量噪声 σ 增大而收缩，解释了它在含噪场景反而稳）。
- **与投影法/DDRM 的差异**：不做测量子空间硬投影（噪声会被 ill-posedness 放大）、不需要 A 的 SVD，因此能上**非线性** forward operator（Fourier 相位恢复、非均匀去模糊）与 Poisson 噪声；代价是每步要对 A 反传、生成链长（论文用 1000 步 DDPM 链），且 Jensen gap 在 forward operator 高度非线性/后验多峰时无保证。
- **对 sim2real 的意义**：真实相机退化若可写成可微 operator（噪点/畸变/色偏的参数化模型），DPS 是零训练成本 baseline；但"渲染→真实"整体差距没有解析 A，DPS 用不上——这是它与条件 bridge 路线分工的关键。

### 收录条目 2：DDRM — Denoising Diffusion Restoration Models

- arXiv：[2201.11793](https://arxiv.org/abs/2201.11793)；venue：**NeurIPS 2022**（DBLP `conf/nips/KawarEES22`；核验 2026-08-14）
- 作者：Bahjat Kawar, Michael Elad, Stefano Ermon, Jiaming Song（Technion / Stanford / NVIDIA）
- 项目：https://ddrm-ml.github.io/
- **方法定位**：线性逆问题 y = Hx + z（H、σ_y 已知）的 zero-shot 变分采样器。对 H 做 SVD，在**谱空间**逐坐标定义条件变分分布：奇异值为 0 的坐标走无条件生成；奇异值非零的坐标按"测量噪声换算到谱空间的水平 σ_y/s_i 与扩散噪声 σ_t 的大小关系"分两档混合 y 与模型预测（式 4–5, 7–8），从而把测量噪声与扩散噪声"对齐"。Theorem 3.2 证明预训练无条件 DDPM 的最优解同时是 DDRM ELBO 的最优解——**无需任何再训练**。
- **效率与边界**：20 NFEs 即可工作（比 SNIPS 类快 ≥50×），ImageNet 上重建/感知质量双优；但硬边界是"H 已知 + SVD 可算"——非线性退化、盲退化、以及 sim2real 这种"整域外观差距"都出界（这一点 DPS 的引言里也点名批评过：DDRM 的去模糊只能做可分离高斯核）。
- **对 sim2real 的意义**：适合当"已知线性退化"子问题（下采样、遮挡、色彩投影）的高效解，或作为构造中间监督信号的工具；不适合当主翻译器。

### 收录条目 3：SDA — Score-based Data Assimilation

- arXiv：[2306.10574](https://arxiv.org/abs/2306.10574)；venue：**NeurIPS 2023**（DBLP `conf/nips/RozetL23`；核验 2026-08-14）
- 作者：François Rozet, Gilles Louppe（Liège）
- 代码：https://github.com/francois-rozet/sda
- **方法定位**：把逆问题从单帧推到**状态轨迹**：给定稀疏/含噪观测 y，推断动力系统轨迹后验 p(x_{1:L}|y)。两个关键设计：(i) 利用 Markov 结构，**只在短轨迹段上训练 score 网络**，推理时拼合成任意长轨迹的近似全局 score，非自回归地一次性生成整条轨迹；(ii) 观测模型与训练完全解耦，推理时才用 DPS 式引导注入，因此同一个先验可服务任意观测配置（zero-shot observation scenarios）。
- **与 DPS/DDRM 的差异**：先验从"图像分布"换成"轨迹分布"（物理一致性由分段 score 保证）；引导方式承袭 DPS 但作用在时空联合变量上。它证明了 zero-shot 路线可以扩展到"仿真器太贵、不能在推理环节反复调用"的场景——推理全程不调用物理模型。
- **对 sim2real 的意义**：двойн重身份——它既是逆问题家族成员，又是 **trajectory inference 的近亲**（论文摘要自称 "score-based data assimilation for trajectory inference"）。对机器人 episode 级别的观测序列翻译（video bridge）而言，SDA 的"短段训练 + 长轨迹拼合"是绕开长序列训练开销的现成配方，可与主库 3MSBM（`2506.10168`）的多边缘桥互补。

### 选型判据：何时 zero-shot guidance，何时训条件 bridge

| 维度 | zero-shot（DPS/DDRM/SDA） | 条件 bridge（CDSB/I²SB/SBUnfold） |
| --- | --- | --- |
| forward operator A | 必须已知；DDRM 还要线性可 SVD、DPS 要可微 | 不需要显式 A，只需能从 g(y\|x) 采样（仿真器天然满足） |
| 训练数据 | 只需无条件先验（大规模 x 样本） | 需要联合样本 (x,y)（成对或可仿真配对） |
| 推理开销 | 长反向链（DPS 数百至上千步）；每步可能要对 A 反传 | 短链（CDSB N=5–50；I²SB 类似），amortized |
| 误差控制 | Jensen gap / 投影偏差，非线性强时无保证 | IPF 收敛有理论（Prop 3），但受 y^obs 典型性约束 |
| 任务切换 | 免训练，换 A 即换任务 | 每个任务/域对要重训（或重精调） |
| 典型失效 | 未知/不可参数化退化；多峰后验 | 真实观测偏离训练 p_obs；配对质量差 |

**SB-Render-Lite 结论**：仿真→真实视觉翻译中，"退化模型"是渲染管线与真实世界的整体差距，无解析 A，但 digital twin 可批量产出配对 (sim, pseudo-real) 或至少联合可采样的数据——落在**条件 bridge**一侧；zero-shot 家族的价值是 (i) 为可参数化的子退化（噪声/模糊/色偏）提供免训练 baseline，(ii) SDA 式"训练态先验 + 推理态观测引导"的解耦思想可用于"真实观测稀疏可得时的 episode 级修正"。

---

## 三、横断综述（b）：Trajectory Inference 家族

### 问题设定

给定随时间演化的分布 μ_t 在少数几个时刻 t_0 < t_1 < … < t_{T-1} 的**静态截面采样**（同一实体不能被重复测量，无逐点对应；单细胞测序是破坏性测量的典型），推断：(i) 中间时刻的分布（population 级插值）；(ii) 个体轨迹与分化命运（entity 级推断）。结构上这是"多边缘、无配对"的分布插值问题——与 sim2real"两个（或多个）域边缘、无配对"同构，区别仅在"时间轴"换成"域轴"。

### 方法谱

**WOT / Waddington-OT（静态 unbalanced OT + 增长率）**
- Schiebinger et al., *Optimal-Transport Analysis of Single-Cell Gene Expression Identifies Developmental Trajectories in Reprogramming*, **Cell 176(4):928–943.e22, 2019**（CrossRef 核验 2026-08-14；18 位作者，一作 Schiebinger）
- 家族奠基作。在相邻时间点之间解熵正则 **unbalanced** OT（KL 放松边缘约束以容纳细胞增殖/死亡，增长率从增殖特征估计并迭代修正），把逐对 coupling 沿时间**复合**得到跨多时点的祖先/后代分布（Markov 假设）。分析了跨 18 天、约 39 个采样时刻的 iPSC reprogramming 时程。
- 定位：**静态、逐对、离散**——不产生连续动力学，插值靠 McCann displacement interpolation。它同时是本报告评测协议一节的源头：用"扣掉中间时点、用相邻两时点 OT 插值、与真实数据比距离"验证 OT 假设本身。

**TrajectoryNet（CNF ≈ 动态 OT）**
- Tong, Huang, Wolf, van Dijk, Krishnaswamy, **ICML 2020, PMLR v119:9526–9536**（DBLP 核验 2026-08-14）；arXiv [2002.04461](https://arxiv.org/abs/2002.04461)；代码 https://github.com/KrishnaswamyLab/TrajectoryNet
- 首个把 continuous normalizing flow 与 Benamou–Brenier 动态 OT 连起来的方法：CNF 的最大似然（=各时点边缘 KL 匹配）+ 路径能量惩罚 λ_e∫‖f‖²（Theorem 4.1：λ 足够大时逼近 W₂ 测地流），再叠三个生物先验正则——growth（离散 unbalanced OT 蒸馏出的增长网络）、density（k-NN 铰链损失把路径按在数据流形上）、velocity（RNA-velocity 余弦对齐）。单一光滑向量场跨全部时点，避免了 WOT 逐对拼接在测量时刻的不连续。
- 局限（MIOFlow 论文的三点批评）：必须从高斯源分布出发、确定性流无内生随机性、CNF 的 Jacobian 迹计算 O(k²)。

**MIOFlow（流形上的 W₂ 匹配 Neural ODE）**
- Huguet, Magruder, Tong, Fasina, Kuchroo, Wolf, Krishnaswamy, **NeurIPS 2022**（DBLP 核验 2026-08-14）；arXiv [2206.14928](https://arxiv.org/abs/2206.14928)；代码 https://github.com/KrishnaswamyLab/MIOFlow
- 放弃似然/KL，直接在 Neural ODE 的预测边缘与观测边缘之间算离散 **W₂ 损失**（POT 库），叠能量正则与 density 正则；整个流在 **Geodesic Autoencoder（GAE）**潜空间里跑——GAE 的潜距离被正则到匹配"多尺度 diffusion geodesic distance"（Definition 1，收敛到流形测地距离），于是欧氏 W₂ 等价于测地 ground distance 的 Wasserstein。SDE 化（可学习 σ_t）提供随机性并帮助分叉，训练后 σ→0 收敛回 W₂ 形式。local（相邻时点）+ global（从 t₀ 整条）两阶段训练。
- 实验直接对打 TrajectoryNet 与 **DSB**（De Bortoli et al. 2021）：petal/dyngen 玩具分叉数据上 leave-one-out W₁ 与 MMD 全面占优且训练时间从 60–90 分钟降到百秒级；EB 数据（200 维 PCA）与 AML 治疗响应上验证了 GAE 的增益。值得注意：**DSB 在这两个玩具集上垫底**——朴素 SB 生成框架不加流形/多边缘结构，直接做 trajectory inference 并不占优。

**CellOT（静态神经 OT 映射，扰动响应预测）**
- Bunne, Stark, Gut, del Castillo 等 9 人，*Learning single-cell perturbation responses using neural optimal transport*, **Nature Methods 20(11):1759–1768, 2023**（CrossRef 核验 2026-08-14）
- 把"control 细胞群 → 扰动后细胞群"当成一次 OT：用 ICNN 对偶参数化（Makkuva et al. 风格的凸位势 min-max）学确定性 Monge map，预测单细胞对药物/细胞因子扰动的响应。评测覆盖 i.i.d. 与 out-of-sample（未见过的病人/细胞类型）泛化。
- 定位：**时间轴退化为两点（before/after）**的 trajectory inference 特例，是"跨域映射"最贴近 sim2real 的成员——sim→real 也可视为一次"域扰动"。局限：确定性映射无多样性、每个扰动一张 map（amortization 需后续工作如 CondOT）。

**SB 系与 UDSB（venue 注记）**
- DSB（De Bortoli et al., NeurIPS 2021）与后续 SB 求解器天然适配"两边缘插值"，多边缘扩展见主库 3MSBM（`2506.10168`）。
- **UDSB**：Pariset, Hsieh, Bunne, Krause, De Bortoli, *Unbalanced Diffusion Schrödinger Bridge*（arXiv [2306.09099](https://arxiv.org/abs/2306.09099)）把质量增减引入 SB 动力学以对齐 WOT 的 unbalanced 语义。**状态标注：仅为 ICML 2023 Workshop 论文，非正会**（DBLP 检索 2026-08-14 仅见 CoRR 条目，无 proceedings 记录，与 workshop-only 状态一致）。引用时须带 workshop 标注，不宜作为强基线证据。

### 方法谱一览

| 方法 | 动力学形态 | OT 语义 | 随机性 | 流形结构 | 多时点耦合 |
| --- | --- | --- | --- | --- | --- |
| WOT | 无（离散 coupling） | 静态熵正则 unbalanced OT | coupling 内生 | 无 | 逐对复合 |
| TrajectoryNet | ODE（CNF） | 动态 OT（能量正则近似） | 无（确定性） | density/velocity 正则 | 单一向量场 |
| MIOFlow | SDE（Neural ODE+σ_t） | 边缘 W₂ 匹配 + 能量正则 | SDE 扩散项 | GAE 测地潜空间 | 单一向量场 |
| CellOT | 无（静态 map） | 神经 Monge map（ICNN 对偶） | 无 | 无 | 两点 |
| DSB/UDSB(workshop) | SDE（bridge） | 熵正则 OT / unbalanced 扩展 | 桥内生 | 无 | 两点（3MSBM 扩展多点） |

### held-out marginal 评测协议（摘录与移植）

**协议定义**（该家族的黄金标准，源自 WOT，TrajectoryNet/MIOFlow 将其标准化）：

1. **留出**：从 T 个时间截面中扣掉一个**中间**时点 t_i 的全部数据（不能扣端点）；
2. **训练**：在其余 T−1 个截面上训练模型；
3. **预测**：模型在 t_i 处生成分布 μ̂_{t_i}（连续方法直接积分到 t_i；离散 OT 用相邻两截面 coupling 的 McCann interpolant）；
4. **度量**：μ̂_{t_i} 与真实留出样本之间的分布距离；
5. **遍历**：对每个中间时点轮流留出，报告逐点与均值。

**各论文的具体实例**：
- TrajectoryNet：指标 **EMD（W₁）**；人工数据因有真轨迹另报 per-trajectory **MSE**；baseline 四件套——**prev**（直接拿上一时点数据当预测）、**next**、**rand**（随机时点）、**OT**（WOT 式 McCann interpolant）。mouse cortex 4 时点 + EB 5 时点，5 维 PCA，多种子报均值±标准差。
- MIOFlow：指标 **W₁ + MMD（Gaussian 核与 mean/线性核两种）**；baseline 为 prev/next 距离的平均；petal/dyngen 留 t=2，EB 留 t=2 与 t=3（200 维 PCA），同时报训练时间。
- WOT：留出时点、用相邻截面 OT 插值对比真实数据，以论证 OT 假设在 reprogramming 时程上成立（协议的原始形态）。
- SDA 提供了同一思想的时序版：观测（=部分边缘信息）只在推理期进入，评测在未观测的状态维度/时刻上做——即"held-out 的不是时点而是观测通道"。

**协议为什么好**：(i) 无需任何逐点真值（destructive measurement 下唯一可行的泛化检验）；(ii) 惩罚"记住训练边缘"的过拟合——插值必须来自模型对动力学/传输结构的内化；(iii) prev/next baseline 极其廉价却极具杀伤力——模型若赢不了"直接复制相邻截面"，说明没学到任何传输结构（MIOFlow 的 EB 表里 baseline 在 t=3 的 W₁ 竟比部分模型好，就是这种诚实性的体现）。

**移植到 sim2real 翻译验证（SB-Render-Lite 可执行方案）**：

1. **构造域轴上的中间边缘**：把"仿真→真实"离散成 K 档 marginal，可选的轴包括——渲染保真度阶梯（光栅化→路径追踪→照片级）、domain randomization 强度、真实/仿真数据混合比、或相机 ISP 参数扫描。每档只需**无配对**的观测集合。
2. **Leave-one-domain-out**：扣掉中间某档（如"中保真渲染"），训练 bridge/翻译器后预测该档分布，报告 W₁/Sinkhorn 距离 + MMD（特征空间：DINO/CLIP embedding，替代原文的 PCA 空间）+ FID。
3. **Baseline 四件套照搬**：prev（低一档数据直接充当预测）、next（高一档）、rand、静态 OT 插值（特征空间 McCann interpolant）。另加 **identity**（不翻译的 sim 图）——sim2real 特有的"复制输入"陷阱。
4. **分层指标**：分布级（上述）之外，凡有配对处（digital twin 重渲染可造）补条件级指标（LPIPS/PSNR per-pair，对应 TrajectoryNet 人工数据的 per-trajectory MSE）；最终以 real-domain policy success 为主指标（与主库 INDEX 的既定原则一致）。
5. **时序版**：对 episode/video bridge，改为留出时间片段的边缘（SDA/3MSBM 语义），检验桥在未见时刻的物理/时序一致性。
6. **必须搭配的告诫**：边缘匹配≠条件正确。SBUnfold（`../reports/2308.12351_sb_unfold.md`）已指出 bridge 匹配边缘不保证条件关系正确；协议须与 pairwise correlation / migration-matrix 式检查（sim 内容属性在翻译前后的保持率）并用，防止"分布对了、语义漂了"。

---

## 四、并入主库建议

1. **新增分区建议**：在 `reports/INDEX.md` 的"SB 图像、科学数据与确定性 OT 应用"与"Generalized / Multi-Marginal"之间增设两个分区——"条件 SB 与扩散逆问题"（收 CDSB 精读 + DPS/DDRM/SDA 条目）与"Trajectory Inference 与 held-out 评测"（收本报告综述节）。CDSB 与库内 I²SB、SBUnfold、SB Flow 构成"条件翻译四象限"（paired-解析桥 / paired-IPF 桥 / unpaired-流匹配 / zero-shot-guidance），建议在 synthesis 里补一张对照表。
2. **papers.tsv 候选行**（若正式入库精读）：`2202.13460  Conditional Simulation Using Diffusion Schrödinger Bridges  2022  conditional_sb_inverse  https://arxiv.org/abs/2202.13460`；DPS/DDRM/SDA 可先以"收录条目"身份挂在本报告下，不必逐篇精读。
3. **实验层面最值得转化的三件事**：(i) CDSB-C 的条件参考测度——用廉价翻译器输出当桥的起点分布，短链修正，直接嵌入 SB-Render-Lite 现有训练循环；(ii) held-out marginal 协议——按上节六点方案落地为标准验证脚本，作为所有 bridge 变体的统一泛化检验；(iii) CDSB-FB 采样——零训练成本的推理期改进，适合 A/B。
4. **风险提示**：UDSB 引用必须保持 workshop 标注；WOT/CellOT 为生物期刊论文，方法描述基于其正文与广泛共识，未做逐段精读（本报告未获取二者全文，元信息经 CrossRef 核验）——若后续要深挖 unbalanced 语义（对应 sim/real 数据量严重不对称的场景），建议对 WOT 方法节与 UDSB 做一次专项精读。
5. **与库内既有报告的衔接点**：BDGxRL（`2602.23737`）做 dynamics gap 的 SB 修正，本报告的 SDA 与 trajectory inference 协议可为其提供"轨迹级验证"工具；3MSBM（`2506.10168`）的多边缘桥正是"域轴多档 marginal"方案的天然求解器，二者拼起来就是一个可预注册的实验设计。

## 附：本报告文献清单与核验状态（检索日期 2026-08-14）

| # | 文献 | venue | 核验方式 | 全文获取 |
| --- | --- | --- | --- | --- |
| 1 | CDSB (2202.13460) | UAI 2022, PMLR v180 | 任务给定（已核验） | arXiv HTML 全文精读 |
| 2 | DPS (2209.14687) | ICLR 2023 | DBLP | arXiv HTML（方法节精读） |
| 3 | DDRM (2201.11793) | NeurIPS 2022 | DBLP | arXiv HTML（方法节精读） |
| 4 | SDA (2306.10574) | NeurIPS 2023 | DBLP | arXiv HTML（摘要+方法概览） |
| 5 | TrajectoryNet (2002.04461) | ICML 2020, PMLR v119:9526–9536 | DBLP | arXiv HTML 全文（含评测协议） |
| 6 | MIOFlow (2206.14928) | NeurIPS 2022 | DBLP | arXiv HTML 全文（含评测协议） |
| 7 | WOT | Cell 176(4):928–943.e22, 2019 | CrossRef (DOI 10.1016/j.cell.2019.01.006) | 未获取全文，方法描述基于正文摘要与领域共识 |
| 8 | CellOT | Nature Methods 20(11):1759–1768, 2023 | CrossRef (DOI 10.1038/s41592-023-01969-x) | 未获取全文，方法描述基于摘要与领域共识 |
| 9 | UDSB (2306.09099) | **ICML 2023 Workshop（非正会）** | 任务给定 + DBLP（仅 CoRR 条目，无 proceedings 记录） | 未精读，仅状态注记 |
