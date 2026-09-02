# E15 · Simulation-free 能量采样竞品：iDEM / NETS / Sendera vs AS / ASBS

> 文献扩充研究员 E15 ｜ 检索与撰写日期 2026-08-14
> 输出范围：1 篇精读（iDEM）+ 3 条收录（NETS、Sendera off-policy、Beyond ELBOs 度量）
> 只读依据：`reports/2504.11713_adjoint_sampling.md`、`reports/2506.22565_adjoint_schrodinger_bridge_sampler.md`、`reports/INDEX.md`

## 选题定位

库内 Adjoint 方法线（AS / ASBS / Functional AS / Discrete AM / Discrete ASBS）以「从未归一化能量 `μ(x) ∝ exp(-E(x)/τ)` 采样、训练时无目标样本」为共同目标，并反复声称「高可扩展性」「能量评估极省」。但库内 25 篇报告**没有**把 AS/ASBS 与同期的 simulation-free / off-policy 神经采样竞品做横向对照——尤其是 Mila 系的 **iDEM**（同样主打 simulation-free + LJ-55）与 Courant 系的 **NETS**（主打无偏 + 高维）。本报告补上这块关键证据：判断 adjoint 线到底是不是「能量采样」这一子问题上的最优选择，还是只是众多可行路线中风格不同的一支。

## TL;DR

- **iDEM（ICML 2024）是 AS 最直接的对标物**：两者都用 replay buffer + simulation-free 内环、都首个规模化到 LJ-55（d=165）、都在 DW-4/LJ-13/LJ-55 上报告 `W2`。差别在训练目标——iDEM 用 **K 步 Monte-Carlo 分数估计**（有偏但一致，bias `O(1/√K)`），AS 用 **Reciprocal Adjoint Matching**（自洽回归 + reciprocal 投影复用能量梯度）。AS 论文自称「iDEM 每次更新的能量评估量约高 `10^5` 数量级」，这是 adjoint 线可扩展性叙事的核心论据，但该数字来自 AS 单方口径、且 iDEM 侧的 `K`（LJ-55 仅 K=100）与 buffer 复用未被对称计入。
- **无偏性维度上 adjoint 线并不占优**：NETS 通过 Jarzynski 等式给出**严格无偏**采样（保留 importance weight 但方差被学习到的 drift 压低），并有可训练后调节的 diffusion 系数直接优化 ESS；而 AS/ASBS 是 SOC / reverse-KL 目标，**mode-seeking**、会漏低密度模态（ASBS 作者自己也建议再叠加 importance sampling）。所以「无偏 + 全模态」这条路 adjoint 线目前是短板而非长板。
- **评测度量必须换口径**：Beyond ELBOs（ICML 2024）大规模实证表明，ELBO / reverse-ESS / reverse-`log Z` **对 mode collapse 不敏感**，高维下 ESS 还会退化成 0/1 二值；`W2`、MMD 虽有 kernel/cost 主观性但跨方法一致性好。库内 adjoint 报告目前几乎只看 `W2` 与 energy-`W2`，**缺 EUBO / 前向指标 / mode-coverage** 证据，这会系统性高估 SOC-sampler 的模态覆盖。

---

## 一、iDEM 精读：Iterated Denoising Energy Matching

### 元信息

| 项 | 内容 |
| --- | --- |
| 标题 | Iterated Denoising Energy Matching for Sampling from Boltzmann Densities |
| 方法名 | iDEM（内含纯 off-policy 变体 pDEM） |
| arXiv | 2402.06121 |
| venue | ICML 2024（已核验，关键词页标 ICML；与 NETS 引用一致） |
| 作者 | Akhound-Sadegh, Rector-Brooks, Bose, Mittal, Lemos, Liu, Sendera, Ravanbakhsh, Gidel, Bengio, Malkin, Tong（Mila / McGill / Oxford / Dreamfold 等） |
| 代码 | github.com/jarridrb/dem |
| 归类 | energy-based sampling；diffusion sampler；simulation-free；SE(3)×Sₙ 等变 |

### 动机

科学场景（分子平衡态、n-body 粒子系统）要从 `μ_target(x) ∝ exp(-E(x))` 采样，且**几乎没有初始样本**。经典 MCMC/AIS/SMC/MD 在高维多模态上混合慢、成本高；已有神经采样器（PIS、DDS、DIS）虽能摊销 MCMC，但训练时都要**模拟完整前向/反向 SDE 并对其反传梯度**，高维下不可承受。核心研究问题：能否只用 `E(x)` 和 `∇E`、在**不模拟 SDE**的前提下学一个覆盖全模态的可扩展采样器。

### 方法核心（内外双环 + K 步 MC 能量估计）

iDEM 是 bi-level 迭代结构，恰好把「在哪里学分数（C2）」和「分数怎么估（C1）」拆开：

- **内环（解决 C1，simulation-free）**：提出 Denoising Energy Matching（DEM）目标。关键恒等式是把带噪边缘 `p_t` 的分数写成对 `p_0` 梯度做高斯卷积的比值，从而得到只依赖能量的 MC 估计
  \[
  \mathcal{S}_K(x_t,t)=\nabla_{x_t}\log\textstyle\sum_i \exp(-E(x_{0|t}^{(i)})),\quad x_{0|t}^{(i)}\sim\mathcal N(x_t,\sigma_t^2),
  \]
  再回归 `L_DEM=‖S_K − s_θ‖²`。这个目标对任意 `(t, x_t)` 都成立，**天然 off-policy**，且不需要对 SDE 积分/反传——这是它与 PIS/DDS 的根本区别。数值上用 LogSumExp 实现（附录消融显示朴素 ratio 估计几乎全 NaN、Jensen 估计有不可消偏差，只有 LogSumExp 可用）。

- **外环（解决 C2）**：用当前 `s_θ` 的反向 SDE 生成一批 `x_0` 存入 replay buffer，为内环提供「信息量高」的起点；外环采样阶段 `θ` 冻结、**不反传 SDE**，故计算便宜。整体可视为 on-policy 与 off-policy 之间的混合。

- **K 步 MC 的 bias–variance（这是与 AS 对比的技术要害）**：
  - 一致性：`S_K` 是一致估计，`E[S_K]→` 真分数当 `K→∞`。
  - 偏差：Prop.1 给出高概率界 `‖S_K − ∇log p_t‖ ≤ c(x_t)·log(1/δ)/√K`，即 **bias `O(1/√K)`**；消融（Q1）实测衰减更快、近 `O(1/K)`。
  - 偏差随时间/低密度区放大：`t→1`（接近 prior）或 `exp(-E)` 期望很小的低密度区，需要更大的 `K` 才压得住偏差——这正是外环要找「信息量高的 `x_0`」的理论动机。
  - 实践取值：GMM `K=500`，DW-4/LJ-13 `K=1000`，**LJ-55 仅 `K=100`**（高维反而调小，靠 buffer 与 clip 稳住），回归目标范数做 clip（20~70）。
  - 结论：**iDEM 目标是有偏的**（论文明说 "the DEM objective is biased"），与 FAB 一样接受「有偏但有效」的工程取舍。

- **对称性**：`S_K` 在把标准正态换成零质心正态后即 `SE(3)×Sₙ` 等变（Prop.2），配 EGNN 骨干；这点与 AS 的等变设计同源。

### 实验与结果

- 基准：40-mode GMM（d=2）、DW-4（d=8）、LJ-13（d=39）、LJ-55（d=165）。
- 基线：PIS、DDS（同为 diffusion，但需模拟轨迹）、FAB（AIS+等变流，前 SOTA）、pDEM（纯 off-policy 消融）。
- 主结果（Tab.2，3 seeds）：iDEM 在 `W2`/TV 上最好；NLL 在 GMM/DW-4/LJ-13 匹配或超过基线。**LJ-55 上 PIS/DDS/pDEM 全部训练发散，只有 iDEM 与 FAB 成功**，iDEM 的 energy 直方图明显更贴真值。代表数字：GMM ESS 0.734、`W2` 7.42；LJ-55 NLL 125.86、ESS 0.106、`W2` 16.13。
- 计算成本（Tab.3，训练小时数）：iDEM 比 FAB 快约 **4×**（高维 LJ-13/LJ-55）、约 1.8×（GMM/DW-4）；FAB 慢在 AIS 瓶颈。纯 off-policy 的 pDEM 最快（LJ-13 仅 1.79h）但**稳定性差**（LJ-55 三跑挂两跑）。
- 复杂度表（Tab.1）：iDEM = MCMC-free + off-policy + 时间 `O(L)` + 内存 `O(d)`；pDEM 时间 `O(1)`；对比 PIS/DDS 内存 `O(Ld)`（因需反传轨迹）。

### 局限

- DEM 目标**有偏**、受 MC 样本方差影响；低密度区/大 `t` 需要更大 `K`，`K` 直接决定单次更新的能量-梯度调用数。
- 无 importance-weight 校正，**不是无偏采样器**（对比 NETS/FAB 的重加权）。
- 外环仍要模拟反向 SDE（虽不反传），故非「完全 simulation-free」；纯 simulation-free 的 pDEM 稳定性不够。
- 基准仍是合成能量 + 粒子系统，未触及视觉/机器人等真实高维控制。

---

## 二、收录条目

### 2.1 NETS：A Non-Equilibrium Transport Sampler（arXiv 2410.02711，ICML 2025，已核验）

- **作者/定位**：Albergo & Vanden-Eijnden（Harvard / Courant-NYU）。把 AIS 用 **Jarzynski 等式**写成时间连续版，并在 annealed Langevin SDE 上**再加一项可学习 drift `b_t`**，把非平衡淬火产生的 unbiasing weight 方差压下去。
- **无偏性（核心卖点）**：通过广义 Jarzynski 等式证明**严格无偏**——importance weight 仍在、但被 transport 显著降方差；最优 `b` 下权重方差为 0。可退化为带 resampling 的 SMC，把 GMM 接受率推到近 100%。
- **训练**：`b_t` 是两类目标的极小点，其一是 **PINN loss（off-policy，不需目标样本，且可控 KL）**；**全部目标都不需对 SDE 反传**（对比 PIS/DDS/SOC 必须反传，对比 CMCD 需参考测度或反传）。可 optimize-then-discretize，训练后自由调 step size 与 diffusivity `ε_t`——**这是一个可直接优化 ESS 的旋钮**。
- **实验**：40-mode GMM（`ESS 0.95–0.98`，高于 iDEM 的 0.734、CMCD-LV 的 0.655）；Neal's Funnel（d=10）、Mixture-of-Student-t（d=50，SOTA）；**高维 GMM 直到 d=200（仅靠 transport 就 60% ESS，AIS 此时≈0%）**；lattice φ⁴ 场论（相变附近，优于 AIS/HMC 重加权）。
- **与 adjoint 线的关键差异**：① NETS 无偏、mass-covering 倾向优于 SOC 的 mode-seeking；② 但 **NETS 未在 LJ-13/LJ-55 等 `SE(3)×Sₙ` 粒子系统上测试**，等变性与分子构象是 AS/ASBS/iDEM 的主场；③ NETS 采样时仍需积分 annealed Langevin SDE（非 simulation-free），且需要能构造时间插值势 `U_t` 并取 `∇U_t、∂_t U_t`。

### 2.2 Sendera et al.：Improved off-policy training of diffusion samplers（arXiv 2402.05098，NeurIPS 2024，已核验）

- **作者/定位**：Sendera, Kim, Mittal, Lemos, Scimeca, Rector-Brooks, Adam, Bengio, Malkin（Mila 等）。一篇**方法学/基准**论文：统一实现并公平对比 diffusion-structured 采样器，重点是连续 GFlowNet 的 off-policy 训练（TB、VarGrad、SubTB）对比 on-policy PIS。
- **主贡献**：① 统一 codebase 复现，**质疑此前若干「鲁棒性/样本效率」声明**（明确指出 FL-SubTB 的对比改了关键实验变量、结果不可复现）；② 发现 partial-trajectory（SubTB）相对 TB **收益甚微且更贵**；③ Langevin parametrization 的归纳偏置在 off-policy 下同样有益；④ 提出**目标空间局部搜索 + replay buffer**（并行 MALA 直接在目标空间探索），有效**防止 mode collapse**、提升样本质量。
- **基准**：25GMM（d=2）、Funnel（d=10）、Manywell（d=32）、**LGCP（d=1600，高维但强结构）**、VAE latent posterior（条件采样）；主指标是 `log Z` 估计误差（reverse/forward）。
- **与 adjoint 线的关系**：代表**off-policy GFlowNet + 显式探索**这条与 SOC/adjoint 正交的路线。它给出的经验教训（局部搜索防 mode collapse、SubTB 收益有限、复现陷阱）对评估 AS/ASBS 的模态覆盖声明有直接借鉴价值。GFlowNet TB/VarGrad 在最优点渐近正确，但推理时不带 importance weight，无偏性不如 NETS 的重加权。

### 2.3 Beyond ELBOs：度量摘录（arXiv 2406.07423，ICML 2024 / PMLR 235:4205-4229，web 复核已确认）

Blessing, Jia, Esslinger, Vargas, Neumann（KIT / Cambridge）。统一 task suite + 多度量大规模评测，并提出量化 mode collapse 的新指标。见下方「四、评测度量应该用什么」。

---

## 三、对照表：iDEM / NETS / Sendera vs AS / ASBS

### 3.1 核心四维（题目指定：能量调用次数 / 无偏性 / 可扩展维度 / 适用能量类型）

| 方法 (venue) | 单次梯度更新的能量调用 | 无偏性 | 已验证可扩展维度 | 适用能量类型 |
| --- | --- | --- | --- | --- |
| **iDEM** (ICML'24) | 内环每次更新需 **K 次 `∇E` 评估**（K=100@LJ-55，1000@LJ-13）；内环 simulation-free，外环模拟反向 SDE 但不反传 | **有偏**但一致，bias `O(1/√K)`；无重加权校正 | LJ-55（d=165），首个能量训练规模化到 LJ-55 | 可微 `E` 且 `∇E` 便宜；`SE(3)×Sₙ` 等变；分子/粒子 |
| **NETS** (ICML'25) | 需沿 annealed Langevin 轨迹取 `∇U_t`（simulation-based 采样）；训练目标**不反传 SDE** | **严格无偏**（Jarzynski + 重加权，权重方差被 drift 压低） | GMM d=200；MoS d=50；lattice φ⁴ | 可微 `U` + 可构造时间插值势 `U_t`（需 `∇U_t,∂_tU_t`）；未测等变粒子系统 |
| **Sendera off-policy** (NeurIPS'24) | 轨迹级（L 步 SDE）+ 偶发并行 MALA 局部搜索；Langevin param 需 `∇E` | GFlowNet TB/VarGrad 最优点渐近正确；推理不带 IS，非严格无偏 | LGCP d=1600（强结构）；Manywell d=32 | 可微能量；Langevin 归纳偏置；通用+条件(VAE) |
| **AS**（库内 ICML'25） | **极低**：RAM + reciprocal 投影 + buffer 复用同一次昂贵 `∇E`；AS 自称比 iDEM 省 ~`10^5` | SOC / reverse-KL，**mode-seeking，会漏模态**；无重加权 | LJ-55（d=165）+ 大规模摊销构象（SPICE/GEOM-DRUGS） | 可微终端能量；分子 FM/量化能量；等变+扭转周期 |
| **ASBS**（库内 NeurIPS'25） | 低（继承 AS）；多一个 corrector 网络的小开销 | SOC / reverse-KL，mode-seeking（作者建议叠加 IS） | MW-5/DW-4/LJ-13/LJ-55 + alanine dipeptide | 可微能量 + **任意 source prior**（Gaussian/harmonic/sim） |

> 「能量调用次数」的诚实注记：AS 的 `10^5` 优势是**其论文单方口径**，机制是 RAM 用解析 reciprocal 后验重采中间态、buffer 让一次 `∇E` 支撑多次更新；iDEM 侧每次内环更新确要 `K` 次 `∇E`，但 iDEM 同样有 buffer 复用、且 LJ-55 只用 `K=100`。两者未在同一「每达到目标精度的总 `∇E` 预算」口径下被第三方对齐测量——这正是主库缺、也是最值得补的一次实测。

### 3.2 补充维度（判断路线取舍用）

| 方法 | 训练是否反传 SDE | simulation-free 程度 | 模态覆盖倾向 | prior 灵活度 | 等变/几何先验 |
| --- | --- | --- | --- | --- | --- |
| iDEM | 否 | 内环全免、外环需前向模拟 | 有偏但 buffer+扩散平滑，覆盖尚可 | 固定 mass-covering prior | 原生 `SE(3)×Sₙ` |
| NETS | 否 | 采样需 SDE 积分（非 free） | mass-covering，无偏、ESS 高 | annealed 路径 `U_t` | 未做等变 |
| Sendera | 可 off-policy（TB 不需） | 需 L 步轨迹 + MALA | 局部搜索显式防 mode collapse | 通用 | 未强调 |
| AS | 否（Adjoint Matching 自洽回归） | 需模拟采样、能量调用省 | **mode-seeking，漏低密度模态** | **固定 Dirac、memoryless（弱）** | 原生等变+周期 |
| ASBS | 否 | 同上 | mode-seeking（建议叠 IS） | **任意 prior（强，含 sim prior）** | 原生等变 |

---

## 四、评测度量应该用什么（Beyond ELBOs 摘录 + 对本库的告警）

Beyond ELBOs 的实证结论（O1–O6、M1）直接决定「怎么比才不骗自己」：

- **ELBO / reverse-ESS / reverse-`log Z` 对 mode collapse 不敏感（O2）**：reverse-KL 的 mode-seeking 会让只覆盖部分模态的模型照样拿到好 ELBO/`log Z`。→ **只报这些等于给 SOC/reverse-KL sampler（含 AS/ASBS）免检模态覆盖。**
- **ESS 在多模态/高维退化为 0/1 二值（O5）**：forward-ESS 高维下常年≈0，reverse/forward ESS 都难反映「collapse 有多严重」；EUBO/ELBO 反而更连续可读。→ 库内把 ESS 当主指标要谨慎，尤其 LJ-55 这类高维。
- **`W2` 与 MMD 虽有 kernel/cost 主观性，但跨方法一致、且与定性结果吻合（O4）**：作为 IPM 是相对可靠的样本质量度量（需目标样本）。→ 库内 adjoint 报告普遍用的 `W2`/energy-`W2` 是合理主指标，可继续用，但**不足以单独证明模态覆盖**。
- **EUBO（前向 KL 上界）适合量化 mode collapse（O3 + Fig.1）**：mass-covering，模型漏掉高密度区就会飙高；但**扩展 EUBO 因 latent 松弛，跨方法类别可比性差**——同类内比较可用，跨类慎用。
- **新指标 Entropic Mode Coverage（EMC）∈[0,1]**：需已知 mode descriptor；`EMC≈1` 表示全模态等概率覆盖。粒子系统可用对称性/簇结构近似构造 descriptor。不均衡模态时改用 expected Jensen-Shannon（EJS）。
- **`log Z` 双向误差（reverse `Δlog Z_r` / forward `Δlog Z_f`）**：Sendera 与 Beyond ELBOs 都用；reverse 侧同样受 mode-seeking 影响，**必须与 forward 侧并看**。

**关于 TVD 的诚实边界**：Total Variation 距离在 Beyond ELBOs 里**不是核心度量**（其主表用 `W2`/MMD/ELBO/EUBO/ESS/EMC/`Δlog Z`）。TVD 主要出现在 iDEM/AS 线，做法是在**能量直方图或原子间距等 1-D 边缘**上算 TV，用来查「能量分布/几何统计对不对」。适用性判断：TVD 适合 1-D 投影的分布保真检查，**不能反映高维联合的模态覆盖**，因此只能作为 `W2`+EUBO/EMC 的补充，不能替代。

**给本库的度量最小集建议**：对任何「能量采样」实验，至少报 `{W2（样本）+ energy-W2 + EUBO 或 EMC + forward/reverse log Z}`；只报 `W2`+reverse-ESS 的结论应视为**未验证模态覆盖**。

---

## 五、对 adjoint 线是否最优的判断（关键证据）

1. **可扩展性**：adjoint 线（AS/ASBS）与 iDEM 在「首个规模化到 LJ-55」这件事上**打平**；AS 的独特优势是把摊销构象生成推到 SPICE/GEOM-DRUGS 这种真正大规模分子集，NETS/Sendera 尚未进入等变分子构象主场。所以「可扩展」这句在**分子构象**语境下 adjoint 线确有独到证据，但在**通用高维多模态**语境下 NETS（d=200 GMM）、Sendera（LGCP d=1600）各有专长，adjoint 并非唯一解。
2. **能量省**：AS 的 `10^5` 叙事**尚未被第三方在统一预算口径下证实**；iDEM 有 buffer 复用 + LJ-55 只用 K=100，差距可能被 AS 的自比口径放大。这是主库最该补的一次对齐实测。
3. **无偏 + 全模态**：这是 adjoint 线**当前的短板**。NETS 严格无偏且 ESS 高，Sendera 的局部搜索显式防 collapse，而 AS/ASBS 是 mode-seeking、作者自承会漏模态、建议叠加 IS。若下游任务对「不漏模态/无偏统计量」敏感，adjoint 线不是安全默认。
4. **prior 灵活度**：ASBS 的「任意 source prior」是 adjoint 线相对 iDEM/NETS 的**真差异化优势**，也是与 sim2real 最相关的一点（可把 sim latent 当 source）。

**结论**：adjoint 线在「等变分子构象 + 任意 prior + 极省能量调用（自称）」这三点上有清晰卖点，但在「无偏性、模态覆盖、通用高维、可复现的能量预算对比」上并不占优、甚至偏弱。称其为「最优」缺乏横向证据；更准确的定位是 **energy-sampling 谱系里「SOC + 高复用 + 结构化 prior」的一支，与 iDEM（simulation-free MC 分数）、NETS（无偏 Jarzynski transport）、GFlowNet（off-policy + 探索）互补而非全面碾压**。

---

## 六、并入主库建议

1. **INDEX 归档**：在「Adjoint Sampler 方法线」下新增一节「能量采样竞品对照（外部基线）」，链接本报告；4 篇均**未**进入库内「已精读 25 篇」清单，iDEM 可升格为正式精读条目，NETS/Sendera/Beyond ELBOs 作为收录基线。
2. **补一次对齐实测（最高优先级）**：在 DW-4 / LJ-13 / LJ-55 上，用**统一「达到目标 `W2` 所需 `∇E` 总调用数」**口径复测 AS vs iDEM，验证/证伪 `10^5` 声明——这是主库「高可扩展性」叙事目前唯一的空洞。
3. **度量口径升级**：把 §四「最小集」写入 `sb_adjoint_extended_synthesis.md` 的实验规范；现有 adjoint 报告若只报 `W2`+reverse-ESS，标注「模态覆盖未验证」。
4. **对 SB-Render-Lite 的转化**：ASBS 的「任意 prior」+ NETS 的「无偏 + 可调 ε_t 优化 ESS」可组合——sim latent 作 source prior、终端为真实感/任务一致性能量、评测同时报 `W2`+EUBO/EMC，避免只优化 mode-seeking 目标而在真实域漏模态。
5. **交叉引用**：本报告与库内 `2504.11713_adjoint_sampling.md`、`2506.22565_adjoint_schrodinger_bridge_sampler.md` 双向链接；NETS 的 Jarzynski 视角可与 `guan_horng_liu_research_roadmap.md` 的 SOC 主线对读。
