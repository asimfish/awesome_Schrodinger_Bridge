# Adjoint / Generalized / Structured Schrödinger Bridge 扩展文献综述

## 1. 总结结论

新增 11 篇并不是同一类“SB 生成模型”，而是把 OT/SB 方法线扩展到四个不同问题：

| 方法线 | 论文 | 输入条件 | 主要输出 |
|---|---|---|---|
| 高效能量采样 | Adjoint Sampling, ASBS | 目标只有未归一化能量 | Boltzmann / conformer samples |
| 新状态空间 | FAS, DAM, Discrete ASBS | 函数轨迹或离散 token/CTMC | 路径样本、离散 fine-tuned model、离散能量样本 |
| 任务结构化 bridge | DeepGSB, GSBM, 3MSBM | 两端/多端样本 + 路径代价或时间边缘 | 遵守任务约束的 transport path |
| 具体条件生成 | I²SB, SBUnfold, React-OT | 成对条件数据或明确两端结构 | 图像恢复、物理 unfolding、化学 TS |

对当前 sim2real 项目最直接的优先级是：

1. **GSBM**：可把 action/geometry/temporal consistency 写成路径状态代价，是 task-aware `SB-Render-Lite` 的直接方法底座。
2. **I²SB**：有 paired sim-real-like 数据时的强 baseline，证明从 source observation 而非 Gaussian 出发可显著减少 NFE。
3. **SBUnfold**：最接近“simulation 训练 + 少量真实观测”的科学案例，强调小修正、少数据稳健性和条件关系评估。
4. **3MSBM**：适合第二阶段的 sequence/video/dynamics bridge，解决逐帧/相邻 pairwise OT 的时间不连贯。
5. **ASBS**：当真实目标样本少、但可以学习可微真实感/任务能量时，用 sim distribution 作为任意 source prior。

其余论文主要提供理论扩展或特定状态空间能力，不宜作为第一版机器人实验入口。

## 2. Adjoint 系列的正确关系

这一谱系的根节点是 **Adjoint Matching**（AM；Domingo-Enrich et al., arXiv 2409.08861, ICLR 2025 Spotlight）：它把预训练 flow/diffusion 模型的 reward 微调形式化为 memoryless 随机最优控制，证明 memoryless noise schedule 是收敛到 tilted 分布 `p* ∝ p_base·exp(r)` 的充要条件，并给出 lean adjoint 最小二乘回归。此后是一源两支：**DAM 回到 AM 的 reward 微调本源**（同构目标 `min E[g(X₁)] + KL(p^u‖p_base)` 的离散 CTMC 版）；**AS/ASBS 则是借用 AM 的 lean-adjoint 技术做能量采样**（无 base 模型、无目标样本——AS 将 AM 特例化为 Dirac 初态 + memoryless，ASBS 再主动解除 memoryless）。AM 不应被视为 AS 的内部技术，而是独立的方法论源头。

### 2.1 Adjoint Sampling：把昂贵能量评估复用起来

AS 将 `target ∝ exp(-E)` 的采样写成随机最优控制，通过 Reciprocal Adjoint Matching 和 replay buffer，让一次终点能量/梯度支持多次训练更新。它解决的是“没有目标样本、能量调用昂贵”，但固定 Dirac 初态和 memoryless 条件限制了先验。

### 2.2 ASBS：允许任意 source prior

ASBS 把 AS 提升为一般 Schrödinger Bridge，增加 Corrector Matching 消除非 memoryless 偏差，并证明交替过程对应 IPF。它允许使用 Gaussian、harmonic 或 simulation distribution 作为 source prior，因此比 AS 更接近 sim-to-real energy-guided transport。

### 2.3 FAS：样本从向量变成整条函数

FAS 在 Hilbert 空间定义 reference measure、Q-Wiener process 与函数空间 adjoint。它不是简单增加序列长度，而是让一条路径成为单个样本，并以函数基严格满足端点，支持跨时间分辨率采样。

### 2.4 两种离散扩展不能混为一谈

| 对比 | Discrete Adjoint Matching | Discrete ASBS |
|---|---|---|
| 主要任务 | 预训练离散模型的 reward fine-tuning | 从离散未归一化能量采样 |
| 模型 | masked CTMC / diffusion LLM | general discrete CTMC sampler |
| 目标 | terminal reward + KL-to-base | discrete Schrödinger Bridge |
| 关键结构 | statistical discrete adjoint、masked 简化 | 循环群、uniform additive reference、controller/corrector |
| 实验 | LLaDA-8B 数学推理 | Ising / Potts |

DAM 更适合离散 action/skill token 的 reward adaptation；Discrete ASBS 更适合 VQ token、格点或符号状态上的 energy-based sampler。

## 3. Generalized SB：从“最短路径”到“任务正确路径”

标准 SB 只最小化相对 reference process 的路径 KL/动能。DeepGSB 和 GSBM 都加入状态或群体代价，但计算路线不同：

- **DeepGSB** 用 FBSDE 与 DeepRL/TD learning 处理 mean-field interaction，可描述拥堵、群体分布和非光滑障碍；
- **GSBM** 将每对端点的路径求解为 conditional SOC，再用 matching 更新全局 drift，更好地保持两端边缘可行性和训练稳定性。

对于单机器人 sim2real，GSBM 更实用。一个可操作的路径代价可以写成：

```text
V_t =
  λ_realism · real/fake energy
+ λ_geometry · keypoint/depth deviation
+ λ_action · inverse-dynamics inconsistency
+ λ_temporal · phase/flow inconsistency
+ λ_safety · collision/contact violation
```

训练时应同时画两类曲线：

- feasibility：source/target marginal 是否匹配；
- optimality：几何、动作和安全代价是否下降。

只用 FID/LPIPS 无法验证 transport 是否保留了策略语义。

## 4. 多时间点：3MSBM 为什么比逐段 SB 更合理

把相邻 snapshot 分别做 OT/SB，再拼起来，容易在连接点产生速度突变，也无法让远端观测约束当前路径。3MSBM 在位置—速度相空间中最小化加速度，并用解析 multi-marginal conditional bridge 同时穿过多个时间边缘。

对机器人视频/rollout，可将以下对象作为 marginals：

- 起始、接触前、接触中、完成等任务阶段；
- 不同时间的 sim/real latent population；
- 多个关键帧或真实传感统计快照。

第一版不建议直接在 RGB 上复现 3MSBM；应先在 policy latent、object pose 或低维 trajectory 上比较：

1. adjacent pairwise SB；
2. sequence-aware cost 的 GSBM；
3. phase-space multi-marginal bridge。

主指标应包括 downstream success、速度/加速度平滑性和跨长时间的 task-phase consistency。

## 5. 三个应用论文给出的工程判断

### 5.1 I²SB：paired restoration 的强基线

I²SB 的效率来自 paired `(clean, degraded)` 端点使 conditional bridge 可解析。它说明从结构化 source image 起步能将采样从约 100 NFE 降到 2–10 NFE，但不能据此声称解决了 unpaired sim2real。

### 5.2 SBUnfold：simulation-trained、小数据稳健

SBUnfold 直接从 detector-level event 运输到 particle-level event，在少量 pseudo-data 下比 data-trained OmniFold 更稳。它提示 sim2real 实验应专门做 target data budget sweep，并检查物理相关性/迁移矩阵，而非只看 target marginal。

### 5.3 React-OT：不确定性低时，确定性 OT 可能更合适

React-OT 面对给定反应物和产物、近似唯一的 TS，采用 deterministic OT-ODE，推理约 0.4 秒。对确定性的 calibration/restoration，OT-ODE 应是 SB 的必要对照；对 one-to-many、unpaired 或多模态目标，stochastic SB 才更有价值。

## 6. 对 `SB-Render-Lite` 的更新建议

### 第一阶段：配对/无配对视觉 transport

- paired synthetic degradation：I²SB；
- unpaired sim/real marginals：SB Flow / DSBM；
- task-aware path cost：GSBM；
- deterministic 对照：OT-ODE / flow matching。

### 第二阶段：energy-guided target

当 real samples 很少时，训练 realness + action-consistency energy，以 sim latent 为 source prior，比较 AS 与 ASBS；ASBS 应通过 prior 消融验证 harmonic/kinematic/sim prior 的价值。

### 第三阶段：trajectory transport

从低维 latent trajectory 开始，比较 pairwise SB 与 3MSBM；若需要把整条连续轨迹视作函数并跨分辨率生成，再考虑 FAS（注意 FAS 是函数空间的 energy-only adjoint 采样器，监督是轨迹能量泛函而非多时刻样本，适用前提是能定义可微的轨迹级能量）。

### 必须增加的评估

- real data budget：`1% / 5% / 10% / 100%`；
- paired 与真正 unpaired 设置分开报告；
- FID 之外加入 keypoint/depth、inverse dynamics、temporal smoothness；
- 最终以 real-domain policy success 为主指标；
- 报告 NFE、能量评估次数、wall-clock 和显存，避免只比较样本质量。

## 7. 最终判断

这批论文把原先“SB 可做无配对 sim-to-real translation”的想法推进成三个更具体的研究命题：

1. **任务结构化**：用 GSBM 让 bridge 同时保持真实感、几何和动作语义；
2. **数据效率**：用 I²SB/SBUnfold 的结构化 source prior，或 ASBS 的任意 prior，减少真实数据与昂贵评估；
3. **时间一致性**：用 3MSBM（样本驱动的多时间边缘 bridge）从单帧 transport 升级为轨迹级 transport；FAS 属函数空间的 energy-only adjoint 采样器，仅当整条轨迹的目标只能写成能量泛函、而非多时刻样本时才适用。

最值得先验证的组合不是直接堆叠所有方法，而是：

```text
I²SB / SB Flow baseline
        + task-aware GSBM cost
        + OT-ODE deterministic ablation
        → real-domain policy success
```

只有在这条最小路线确认视觉 transport 确实提升策略后，再扩展到 ASBS energy guidance 和 3MSBM trajectory bridge。

## 2026-08-14 扩充轮补充

扩充轮为本综述的 Adjoint 谱系补齐了上下文：E06 精读了根节点 Adjoint Matching（ICLR 2025 Spotlight），确认「AM/DAM 属 reward 微调、AS/ASBS/FAS/DASBS 属能量采样」的一源两支结构；E14 补上 AS 之前的 SOC 采样器源头（PIS/DDS/CMCD），并把 adjoint 线解决的瓶颈精确命名为反传、on-policy 耦合与先验三项。E15 用 iDEM/NETS 等竞品对照指出，AS/ASBS 的 reverse-KL 目标在「无偏 + 全模态」维度并不占优，仅看 W2 的评测口径会系统性高估模态覆盖；E16 补齐 latent bridge 与少步部署链（LBM/CDBM/DBIM/ASBM），直接约束第一阶段方案的 NFE 预算。详见 `topics/`：[E06](../topics/E06_adjoint_matching_origin.md)、[E14](../topics/E14_soc_sampler_origins.md)、[E15](../topics/E15_energy_sampler_competitors.md)、[E16](../topics/E16_latent_bridge_fewstep.md)。
