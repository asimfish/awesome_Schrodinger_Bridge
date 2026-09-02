# Adjoint Matching：Adjoint 谱系的源头精读（reward fine-tuning 的 memoryless SOC）

> 文献扩充研究员 E06 ｜ 2026-08-14 ｜ 输出文件：`topics/E06_adjoint_matching_origin.md`

## 选题定位

本报告补上库内 Adjoint 方法线唯一缺失的**源头**：`Adjoint Matching`（AM, arXiv 2409.08861）。库内已有其五篇后继的精读——`Adjoint Sampling`(AS)、`ASBS`、`FAS`、`DAM`、`DASBS`——但方法本身没有独立精读，此前综述（`sb_adjoint_extended_synthesis.md`）只把 AM 当作 AS 内部的一个技术带过，导致谱系起点不清。前序审查（R04）特别指出两点须厘清：(1) AM 的原始目的是**对预训练生成模型做 reward fine-tuning**，而非能量采样；(2) `DAM` 是"回到 AM 原始 reward 微调目的"的离散版，`AS/ASBS/DASBS` 则是把 AM 的技术**借用**到能量采样。本报告据此把源头补上，并给出"继承了什么 / 改变了什么"的谱系图。

## TL;DR

- **AM 解决的问题**：如何把 diffusion / Flow Matching 的 reward fine-tuning 做成"可证明收敛到 tilted 分布 `p*(x) ∝ p_base(x)·exp(r(x))`"的方法。它把 fine-tuning 写成**随机最优控制（SOC）**，并指出朴素 KL 正则会引入一个 **initial value function bias**（`V(X_0,0)` 项），使结果偏离 tilted 分布。
- **两个核心贡献**：(A) **memoryless noise schedule** `σ(t)=√(2η_t)`——论文证明它不仅**充分**而且**必要**：只有让 `X_0` 与 `X_1` 独立，`exp(V(X_0,0))` 才能被积分消掉，fine-tune 后还能换任意 schedule（含 `σ=0`）采样；(B) **Adjoint Matching 算法**——把 SOC 转成一个**无 importance weight 的最小二乘回归**，并用 **lean adjoint** 去掉经典 adjoint ODE 中期望为零的项，降低方差、省去对控制的 Jacobian。
- **对 SB-Render-Lite 的判断**：把 real-domain 评分（realness / inverse-dynamics / policy value）作为可微 reward，对"sim→real 翻译/生成模型"做 AM fine-tuning 在原理上**可行且有吸引力**——KL-to-base 锚定天然抑制 reward hacking，`λ` 给出 realness↔diversity 的 principled 权衡。但 AM 微调的是"生成模型"而非 unpaired translation 本身，需要一个 sim-conditioned 生成器 + 可微 real reward，且真实样本稀缺时应转向 ASBS（sim 作 source prior）。

---

## 一、元信息

| 项目 | 内容 |
|---|---|
| 标题 | Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models with Memoryless Stochastic Optimal Control |
| 方法名 | Adjoint Matching（AM）；配套的 Memoryless Flow Matching 生成过程 |
| 作者 | Carles Domingo-Enrich, Michal Drozdzal, Brian Karrer, Ricky T. Q. Chen（FAIR at Meta） |
| arXiv | 2409.08861 |
| 会议 | **ICLR 2025 Spotlight**（用户已核验；`DAM` 的 arXiv 参考文献亦将其标注为 "In ICLR, 2025"，形成独立佐证。检索日期 2026-08-14） |
| 归类 | reward fine-tuning；stochastic optimal control；Flow Matching / diffusion；RLHF-style alignment |
| 全文获取 | arXiv HTML 全文（正文 + 附录 §10–§15 完整推导与实验细节）已通读；无缺失章节 |

---

## 二、动机：reward fine-tuning 的 SOC 形式化与 value function bias

### 2.1 目标：tilted 分布

沿用 LLM RLHF 的目标，AM 想让 fine-tune 后的生成模型采样自 **tilted 分布**（论文式 (1)）：

```text
p*(x) ∝ p_base(x) · exp(r(x))
```

其中 `p_base` 是预训练模型的样本分布，`r(x)` 是（可微的）reward model。论文强调：现有 diffusion fine-tuning（ReFL、DRaFT、DPOK 等）**大多忽略 `p_base`、只盯着 reward**，因而容易钻 reward model 的对抗漏洞（adversarial artifacts）；此前没有一个"简单且可证明生成 tilted 分布"的方案。

### 2.2 统一记号：把 FM / DDIM / DDPM 写成同一个 SDE

论文先把 Flow Matching ODE/SDE、DDIM、DDPM 统一为（式 (10)-(11)）：

```text
dX_t = b(x,t) dt + σ(t) dB_t,  X_0 ~ N(0, I)
b(x,t) = κ_t·x + (σ(t)²/2 + η_t)·s(x,t)
κ_t = α̇_t/α_t,   η_t = β_t·(α̇_t/α_t·β_t − β̇_t)
```

基于 reference flow `X̄_t = β_t X̄_0 + α_t X̄_1`（`α_0=β_1=0, α_1=β_0=1`）。这套统一记号是后面一切推导的地基。

### 2.3 SOC 形式化与 MaxEnt RL 的等价

fine-tuning 写成二次代价、control-affine 的 SOC（式 (12)-(13)）：

```text
min_u  E[ ∫₀¹ (½‖u(X_t,t)‖² + f(X_t,t)) dt + g(X_1) ]
s.t.   dX^u_t = (b + σ·u) dt + σ dB_t,  X^u_0 ~ p_0
```

- 由 Girsanov 定理，控制代价 `E[∫½‖u‖²]` 恰等于 `KL(p^u ‖ p_base)`（条件于同一 `X_0`，式 (18)）→ 这就是 **KL 正则的 MaxEnt RL**（式 (19)）。
- 经典结论：最优控制 `u*(x,t) = −σ(t)ᵀ ∇_x V(x,t)`，其中 value function `V(x,t) = −log E_base[exp(−∫ₜ¹ f − g) | X_t=x]`。设 `f=0, g=−r` 即对应 tilted 目标。

### 2.4 关键病灶：initial value function bias

朴素地把 KL 正则加进去，得到的最优路径分布是（式 (22)-(23)，`f=0, g=−r`）：

```text
p*(X_0, X_1) = p_base(X_0, X_1) · exp( r(X_1) + V(X_0, 0) )
```

问题就在 **`V(X_0,0)` 这一项**：它是初始分布的 value function，把结果从 tilted 分布 (1) 上偏移开。直觉极端情形：当 `σ(t)=0`（noiseless，ODE 采样）时 `X_0` 完全决定 `X_1`，朴素 fine-tuning **根本不起作用**。

- 这与 LLM 不同：LLM 没有"从初始噪声 `X_0` 迭代生成 `X_1`"的动力学，因此不存在对 `X_0` 的依赖，KL 正则可以直接用。
- 已有补救（Uehara et al. 2024b）是**再学一个初始分布**去抵消 bias，但需要额外的辅助生成模型，复杂。AM 的主张是：换一个特定 noise schedule 就能直接消掉，无需辅助模型。

---

## 三、方法核心（一）：memoryless noise schedule 为何必要

### 3.1 定义与充分性

**Definition 1（memoryless）**：生成过程 memoryless ⟺ `X_0` 与 `X_1` 独立，即 `p_base(X_0,X_1)=p_base(X_0)·p_base(X_1)`。

一旦 base 过程 memoryless，把式 (23) 对 `X_0` 积分：

```text
p*(X_1) = ∫ p_base(X_0)·p_base(X_1)·exp(r(X_1)+V(X_0,0)) dX_0
        ∝ p_base(X_1)·exp(r(X_1))          ← 正是 tilted 分布 (1)
```

`exp(V(X_0,0))` 被 `X_0` 的积分吸收成归一化常数，**bias 消失**。这就是 memoryless 的充分性。

**Proposition 1（充要条件）**：在 (10)-(11) 族中，过程 memoryless ⟺ `σ(t)² = 2η_t + χ(t)`，且 `χ` 满足 `∀t∈(0,1], lim_{t'→0⁺} α_{t'}·exp(−∫_{t'}^{t} χ(s)/(2β_s²) ds) = 0`。特例 **`σ(t) = √(2η_t)` 即 memoryless noise schedule**。

- 直觉：memoryless `σ(t)` 在 `t=0` 处**趋于无穷**（靠近噪声端时剧烈 mixing，彻底抹掉 `X_0` 的信息），在 `t=1` 处**趋于 0`**（靠近样本端保持稳定）。于是终点样本 `X_1` 不携带任何 `X_0` 信息。
- Memoryless Flow Matching 恰好对应 diffusion 里的 **DDPM**（即 DDIM 取 memoryless schedule = DDPM）。

### 3.2 必要性（Theorem 1）——这是 AM 区别于"随手加 KL"的分水岭

**Theorem 1**：在 (10)-(11) 族中，若想**允许 fine-tune 后用任意 noise schedule 采样**（尤其是 `σ=0` 的 ODE 采样）且仍生成 tilted 分布 (1)，则 fine-tuning **必须**用 memoryless schedule `σ(t)=√(2η_t)`。这是**唯一**保持 velocity 与 score 关系、从而允许训练后自由换 schedule 的选择。

必要性证明的推导链（附录 §12，供复现/审阅定位）：

1. **§10.2 Prop 4（forward-backward 轨迹同分布）**：构造 forward SDE (63) 与 backward SDE (64)，证明二者在任意 noise schedule 下轨迹分布相等（up to time flip），从而 `(X̄_0, X̄_1)` 的联合分布等于 `(X_1, X_0)` 的联合分布。这是把"generative 过程的 memoryless"翻译成"forward 过程端点独立"的桥梁。
2. **§12.1 Prop 1 证明**：用 Lemma 2（variation of parameters 解 forward SDE）显式写出 `X̄_t` 的解 (150)，取 `t→1⁻` 得 (151)；`χ` 的假设 (25) 恰好让"`X̄_0` 前的系数"与"`ξ`（drift 项）积分"都归零，只剩与 `X̄_0` 独立的随机积分项 (154) ⟹ `X̄_1 ⟂ X̄_0`。反向亦证其必要。
3. **§12.2 Theorem 1 证明（HJB + Hopf–Cole）**：对 `p_base` 与 `p*=p_base·exp(r)/Z` 各写一条 forward SDE 的 Fokker–Planck (161)，用 Hopf–Cole 变换 `𝒱=−log p̄` 转成 HJB (163)-(164)，令差 `V̂=𝒱*−𝒱` 也满足 HJB (166)，再**逆向工程**出一个 SOC 问题 (167)-(168)——其扩散系数**必为 `σ(t)=√(2η_t)`**，且最优控制满足 `s*(x,t)=s(x,t)+u*/√(2η_t)`（式 (170)）。这条链把"要收敛到 tilted 分布"反推成"schedule 必须 memoryless"。

**Table 7 的实验反证**：用 `σ(t)=1`（非 memoryless）做 AM fine-tune，ImageReward 明显上不去（`0.009` vs. memoryless 的 `0.882`），直接验证了 Theorem 1 所言的 value bias。

### 3.3 落到可训练形式

代入 `σ=√(2η_t)`，控制 `u` 用 fine-tune 的向量场表示（式 (26)-(27)）：

```text
Memoryless Flow Matching:
  full drift = 2·v_finetune(x,t) − (α̇_t/α_t)·x
  u(x,t) = √( 2 / (β_t(α̇_t/α_t·β_t − β̇_t)) )·( v_finetune − v_base )
DDIM/DDPM:
  u(x,t) = −√( α̇̄_t / (ᾱ_t(1−ᾱ_t)) )·( ε_finetune − ε_base )
```

即：**微调 `v_finetune`（或 `ε_finetune`），控制就是它与 `v_base` 的差**。训练后把 `v_finetune` 塞回任意 schedule（含 `σ=0`）即可采 tilted 分布。

---

## 四、方法核心（二）：lean adjoint 方程的推导链与 Adjoint Matching

### 4.1 两类既有 SOC 解法及其痛点

- **Adjoint method（§5.1.1）**：直接对 SDE 仿真求导。
  - *Discrete Adjoint*（discretize-then-differentiate）：把数值解全存进计算图再反传，**显存爆炸**，常需 gradient checkpointing。
  - *Continuous Adjoint*（differentiate-then-discretize，Pontryagin）：解 adjoint ODE（式 (30)-(31)）
    ```text
    da/dt = −[ aᵀ·∇_x(b+σu) + ∇_x(f + ½‖u‖²) ],   a(1) = ∇g(X_1)
    ```
    梯度由式 (32) 给出。可扩展但**经验上不稳定**。
- **Importance-weighted matching（§5.1.2，SOCM / cross-entropy）**：回归到 `u*`，landscape 凸、训练稳，但 importance weight `ω` 的方差**随维度指数爆炸**，高维生成模型上不可用。

AM 想同时要：least-squares 的稳定/可解释 + adjoint 的可扩展（无 importance weight）。

### 4.2 从 basic Adjoint Matching 到 lean adjoint

**观察一（去 importance weight）**：不必像 SOCM 那样精心构造 `u*` 的 importance-weighted 估计，直接回归到**当前控制**下的目标场 `−σᵀ∇J(u;·)`。定义 **basic Adjoint Matching**（式 (34)）：

```text
L_basic(u; X) = ½ ∫₀¹ ‖ u(X_t,t) + σ(t)ᵀ·a(t; X, ū) ‖² dt,   ū = stopgrad(u),  X ~ p^ū
```

- **Prop 2**：`L_basic` 对 `θ` 的梯度**恰等于** continuous adjoint 的梯度 (32)；且其唯一临界点是最优控制 `u*`。
- 本质是一个 **consistency loss**：`u*` 是不动点关系 `u(x,t)=−σᵀ∇_x J(u;x,t)` 的唯一解，`a` 充当 `∇J` 的随机估计。

**观察二（去掉期望为零的项 → lean adjoint）**：最小二乘最优解是回归目标的条件期望，故最优处
`u*(x,t)=E_{p*}[−σᵀ a(t;X,u*) | X_t=x]`（式 (35)）。两边乘 Jacobian `∇_x u*` 并重排得（式 (36)）：

```text
E_{p*}[ u*ᵀ·∇_x u* + a(t;X,u*)ᵀ·σ·∇_x u*  | X_t=x ] = 0
```

括号内两项**正是 adjoint ODE (30) 里的部分项**，且在最优处期望为零。于是把它们从 adjoint ODE 中**删掉**，定义 **lean adjoint `ã`**（式 (38)-(39)）：

```text
dã/dt = −( ãᵀ·∇_x b(X_t,t) + ∇_x f(X_t,t) ),   ã(1) = ∇_x g(X_1)
```

与经典 adjoint (30) 相比，**去掉了 `∇_x(σu)`（控制的 Jacobian `∇_x u`）与 `∇_x(½‖u‖²)` 两项**。最终 **Adjoint Matching 目标**（式 (37)）：

```text
L_AdjMatch(u; X) = ½ ∫₀¹ ‖ u(X_t,t) + σ(t)ᵀ·ã(t; X) ‖² dt,   ū = stopgrad(u),  X ~ p^ū
```

**Prop 7**：`E[L_AdjMatch]` 的唯一临界点仍是 `u*`（证明依赖 Prop 2 与积分方程解的唯一性 Prop 8 / Grönwall）。

### 4.3 与经典 adjoint 方法的三点区别（审阅要点）

1. **梯度不再相同**：lean adjoint `ã` 一般**不等于** cost functional 的梯度（式 (29) 仅在 `u=u*` 时成立），因此 AM 的期望梯度**不同于** continuous adjoint。
2. **方差更低 / 收敛更好**：删去的是期望为零的项，即便在最优处也能降低方差。
3. **计算更省**：lean adjoint **不需要控制的 Jacobian `∇_x u`**；实测每步 wall-clock，AM `156s` ≈ Discrete Adjoint `152s` < Continuous Adjoint `204s`（后者要额外反传 `∇_x‖u‖²`）。

配套 **Algorithm 1（FM）/ Algorithm 2（DDIM）**：前向用 memoryless schedule 采 `m` 条轨迹（`X_t, ã_t` 均 stopgrad）→ 反向解 lean adjoint ODE（终端 `ã_1=−∇r(X_1)`）→ 计算 `‖(2/σ)(v_finetune−v_base)+σ·ã‖²` 并更新。工程细节（§15）：`σ(t)=√(2(1−t+h)/(t+h))` 加 offset 避免除零；lean adjoint 幅度随时间反向近似指数增长，故用 loss clipping `LCT=1.6λ²` 和时间步子采样（保底采最后 25% 步）。

---

## 五、实验与结论

- **设置**：text-to-image，Flow Matching base（512×512 latent + U-Net，类 LDM），`r(x)=λ·ImageReward`。`K=40` 步，2×A100。指标分离评估：ClipScore / PickScore（一致性）、HPS v2（对**未见**人类偏好的泛化）、DreamSim Diversity（多样性）。
- **baseline**：DPO、ReFL、DRaFT-K；同一 memoryless SOC 框架内还比了 Continuous / Discrete Adjoint。
- **主结论**：
  1. **memoryless SOC 全面优于既有 baseline**；在同框架内，**Adjoint Matching > Continuous / Discrete Adjoint**（后两者一致性与人类偏好指标更差）。
  2. `λ` 提供 **principled 的 consistency/preference ↔ diversity 权衡**，且 AM 对 `λ` 稳定；而 DRaFT-1 只优化 reward、必须靠 **early stopping** 做权衡，超参极敏感（Figure 3、Figure 5 Pareto front）。
  3. **首次能对 Flow Matching 做有理论保证的 reward fine-tuning**。
  4. Table 7 消融确证 memoryless 的必要性；Table 8 显示在 100/200 采样步下 AM 与 40 步统计等价，且相对 DRaFT-1 的优势随步数增大。
- **DPO 的注意点**：论文在"以 reward model 为起点、on-policy 采偏好对"的 apples-to-apples 设定下比较，DPO 退化到与 base 相当甚至更差——不能与其 off-policy、高质量策展数据的原版设定混淆。

---

## 六、局限性

- **需要可微 reward**：lean adjoint 终端条件是 `∇r(X_1)`；reward 不可微 / 噪声大时不适用（这正是后续 DAM 用统计 adjoint 绕开的痛点）。
- **数值细节敏感**：memoryless `σ(t)` 在 `t=0` 无穷、`t=1` 趋零，需 offset；lean adjoint 幅度指数增长需 clipping 与时间步子采样，`LCT` 常数需调。
- **CFG 非 principled**：fine-tune 后再加 classifier-free guidance 只微调了 conditional 模型，作者承认其采样分布不明确。
- **实验域单一**：只在大规模 text-to-image 上验证，没有视觉之外（机器人 / 控制 / 科学）的实证。
- **mode-seeking 未讨论**：反向 KL / SOC 的 mode-seeking 倾向在本文未作为主问题分析（后续 AS/ASBS 才明确提及漏模态）。

---

## 七、继承关系图：AM → AS → ASBS → FAS / DAM / DASBS

**一句话谱系**：AM 是**源头**，贡献三大件——(a) memoryless SOC 形式化、(b) memoryless noise schedule、(c) lean-adjoint 最小二乘回归。此后**分成两条支流**：一条把 AM 的 fine-tuning 本源**离散化**（→ DAM）；另一条**借用** AM 的 lean-adjoint 技术去做"从未归一化能量采样"（→ AS → ASBS/FAS/DASBS）。R04 的关切正在于此：**AM/DAM 属 reward fine-tuning，AS/ASBS/FAS/DASBS 属能量采样**，谱系不是一条直线。

```text
                         Adjoint Matching (AM, ICLR'25 Spotlight)
                         目的: 预训练生成模型的 reward fine-tuning
                         核心: memoryless SOC + memoryless schedule + lean adjoint
                                   │
              ┌────────────────────┴───────────────────────────┐
              │  支流 A：坚持 fine-tuning 本源                    │  支流 B：借 AM 技术做"能量采样"
              │  (有 base 模型, 目标 = terminal reward + KL)      │  (无 base/无目标样本, 从 energy 采样)
              ▼                                                   ▼
   DAM  (Discrete Adjoint Matching, ICLR'26 Poster)      AS (Adjoint Sampling, ICML'25)
   连续 → 离散 CTMC(如 diffusion LLM);                    固定 Dirac 初态 + memoryless;
   不可微 → 用 Dynkin/统计视角定义 discrete adjoint         Reciprocal Adjoint Matching + replay buffer
                                                                   │
                                              ┌────────────────────┼───────────────────────┐
                                              ▼                    ▼                        ▼
                                 ASBS (NeurIPS'25 Oral)    FAS (ICML'26)          (ASBS 的离散化)
                                 解除 memoryless→任意 prior;  R^d 向量 → Hilbert    DASBS (ICML'26)
                                 加 Corrector Matching;      函数空间整条轨迹;      循环群 + uniform
                                 交替 = IPF 两 half-bridge    stochastic max. princ. additive reference
```

各步"继承了什么 / 改变了什么"：

| 方法 | venue（2026-08-14 核验） | 继承自 AM | 相对上游**改变**了什么 | 属性 |
|---|---|---|---|---|
| **AM** 2409.08861 | ICLR 2025 Spotlight | —（源头） | 提出 memoryless SOC + memoryless schedule + lean adjoint 回归 | reward fine-tuning |
| **AS** 2504.11713 | ICML 2025 | lean adjoint、least-squares、stopgrad | 目的从 fine-tuning→**Boltzmann 能量采样**（无 base、无目标样本）；引入 **RAM + replay buffer** 复用昂贵能量评估；固定 **Dirac 初态**、终端条件由能量梯度给出 | 能量采样 |
| **ASBS** 2506.22565 | NeurIPS 2025 Oral | AM 的 Adjoint Matching + AS 的能量采样框架 | **解除 memoryless**、允许**任意 source prior**；加 **Corrector Matching** 消非 memoryless/任意 prior 的偏差；证明交替 AM+CM = **IPF** 两个 half-bridge → 全局 SB。AS 是其 `(drift,prior)=(0,Dirac)` 特例 | 能量采样（SB） |
| **FAS** 2511.06239 | ICML 2026 | AS 的能量采样 + AM 的 matching 目标 | 样本从 `R^d` 向量 → **Hilbert 函数空间整条轨迹**；用 **stochastic maximum principle** 推函数空间 adjoint；Q-Wiener process、trace-class 协方差、Dirichlet 边界基 | 能量采样（函数空间） |
| **DAM** 2602.07132 | ICLR 2026 Poster | **AM 的原始目的**：terminal reward + KL-to-base | **回到 fine-tuning 本源的离散版**：连续 → **离散 CTMC**（diffusion LLM）；不可微 → 用 **Dynkin 公式 / 统计视角**定义 discrete adjoint（而非控制论）；masked CTMC 简化 | reward fine-tuning（离散） |
| **DASBS** 2602.08243 | ICML 2026（早期 ICLR'26 DeLTa Workshop） | ASBS 的 controller/corrector + SB/IPF | **把 ASBS 能量采样离散化**：离散 CTMC；需**循环群结构 + uniform additive reference**（对应连续 AM 的 additive Gaussian noise）；实验 Ising/Potts | 能量采样（离散 SB） |

**对 R04 关切的直接回应**：
- **"DAM 是回到 AM 原始 reward 微调目的的离散版"** —— 成立。DAM 的目标函数 `min E[g(X_1)] + KL(p^u‖p_base)` 与 AM 完全同构（有 base、terminal reward、KL 锚定），只是状态空间离散、adjoint 改由统计视角构造。它是六篇里**唯一**与 AM 同属 fine-tuning 支线者。
- **"AS/ASBS 是把 AM 技术借用于能量采样"** —— 成立。AS/ASBS/FAS/DASBS 都没有"预训练 base 生成模型 + reward"，而是"只有未归一化能量 + 无目标样本"，借用的是 AM 的 lean-adjoint 最小二乘这一**求解器**，问题本身换成了 sampling / Schrödinger Bridge。
- 因此综述里"把 AM 当作 AS 内部技术"是**降维**了：AM 是独立的方法论源头，其 memoryless 必要性定理（Theorem 1）与 fine-tuning 语义在 AS 中被特例化（Dirac + memoryless），而在 ASBS 中被**主动解除**。

---

## 八、与 SB-Render-Lite 的启发：real reward 微调翻译模型是否可行

**结论：可行且有独特优势，但要用对"它微调的是生成模型"这一前提。**

### 8.1 可行的映射

把 sim→real 视觉迁移的"翻译/生成模型"当作 `p_base`，把 real-domain 评分当作可微 reward `r(x)`，AM 让模型采样自

```text
p*(x) ∝ p_translator_base(x) · exp( λ · r_real(x) )
```

即"在**不偏离 base 翻译分布太远**（KL 锚定）的前提下，提升 real-domain 评分"。候选 `r_real` 可组合：realness / real-fake discriminator、inverse-dynamics 一致性、keypoint/depth 保持、下游 policy value。

### 8.2 AM 相对 DRaFT/ReFL 的三点独特红利

1. **无 value bias 的理论保证**：memoryless schedule 让 fine-tune 收敛到定义好的 tilted 分布，避免"只顾 reward、忘了 base"导致的 render 崩坏。
2. **principled 权衡而非 early stopping**：`λ` 显式控制 realness↔diversity，契合库内一贯要求"保留 sim 多样性同时提真实感"；DRaFT/ReFL 需靠早停调 trade-off，脆弱。
3. **KL-to-base 天然抗 reward hacking**：判别器易被对抗样本欺骗，AM 的 KL 锚定 + 适中 `λ` 能抑制 render 出现 adversarial artifact——这正是本文动机 §2.1 点名的问题。

### 8.3 边界条件与风险（须写进实验设计）

- **它微调"生成模型"，不是 unpaired translation 本身**：要做 sim→real，`p_base` 应是一个 **sim-conditioned 生成器**（如以 sim latent/observation 为条件的 Flow Matching translator），reward 是 real-domain 判别/策略分；不能直接拿 AM 当 I²SB/GSBM 那样的两端 bridge。
- **需要可微 real reward**：real 样本稀缺时先训一个 realness / task energy——**这恰好落入 ASBS 的场景**（用 sim 分布作 source prior、可微能量作 target），因此 SB-Render-Lite 的自然路线是 **AM 式 reward fine-tuning 与 ASBS energy-guided transport 二选一或串联**，取决于"真实样本 vs. 真实能量"哪个更易得。
- **reward hacking 仍需监控**：判别器质量、`λ` 大小、以及最终必须用 **downstream real-domain policy success** 作主指标（与库内 `synthesis` / `extended_synthesis` 的既有判断一致），FID/LPIPS 不足以验证 transport 是否保留策略语义。
- **计算与数值**：memoryless `σ(t)` 需 offset、lean adjoint 需 clipping/子采样；在高分辨率 render 上要预估 `K` 步前向 + `K` 步 lean-adjoint 反向的显存/时延。

---

## 九、并入主库建议

1. **定位登记**：将本文登记为 Adjoint 方法线的**源头/method origin**，而非 AS 的附属技术。建议在 `reports/INDEX.md` 的"Adjoint Sampler 方法线"小节**置于 AS 之前**新增一行（若后续把本报告转正为正式精读，文件名建议 `reports/2409.08861_adjoint_matching.md`，与库内 `arXiv号_slug.md` 命名一致）。
2. **勘误既有综述**：`sb_adjoint_extended_synthesis.md` §2 目前把 AM 隐含在 AS 内部；建议补一句"AM（ICLR 2025 Spotlight）是独立源头，AS 是其 memoryless+Dirac 特例，ASBS 主动解除 memoryless"，并引用本报告的谱系表。**遵循硬性约束，本次未改动该文件**，仅在此提出建议。
3. **谱系图复用**：第七节的谱系图与"继承/改变"表可直接并入 `guan_horng_liu_research_roadmap.md` 或 synthesis，作为六篇 Adjoint 论文的统一导览；其中"reward fine-tuning（AM/DAM）vs. 能量采样（AS/ASBS/FAS/DASBS）"的二分是 R04 要求的核心结论。
4. **venue 一致性**：建议全库把 ASBS 记为 **NeurIPS 2025 Oral**（当前库内记为"NeurIPS 2025"，本次已复核为 Oral）；其余 AS(ICML'25)/FAS(ICML'26)/DAM(ICLR'26 Poster)/DASBS(ICML'26) 与库内记录一致，AM 采用用户已核验的 **ICLR 2025 Spotlight**。所有非既定项均于 **2026-08-14** web 复核。
5. **实验对接**：把第八节 "AM reward fine-tuning vs. ASBS energy-guided transport" 作为 SB-Render-Lite 第二阶段（energy-guided target）的一个**并列候选**写入实验计划，主指标锁定 real-domain policy success，并显式记录 reward 评估次数 / NFE / wall-clock。

---

*检索与核验日期：2026-08-14。全文来源：arXiv:2409.08861（HTML 全文，正文 + 附录 §10–§15 完整通读）。谱系五篇 venue 经 OpenReview / nips.cc / iclr.cc / icml.cc / 作者主页交叉复核。*
