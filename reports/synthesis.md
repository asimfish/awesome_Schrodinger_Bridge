# 综合文献地图：OT / SB 如何迁移具身跨域数据

## 1. 这批论文回答的核心问题

用户关心的“用最优传输来转移具身中的不同数据集”可以拆成四类问题：

1. **人类经验 -> 机器人策略**：human egocentric video / human hand video / passive video 与 robot teleop data 之间存在 embodiment、视角和动作空间差异。
2. **仿真 -> 真机策略**：simulation demos 便宜且覆盖广，但视觉和动力学与真实环境不一致。
3. **不同状态空间之间的 imitation**：expert 和 learner 的 observation/action space 不同，无法直接计算逐点距离。
4. **无配对分布翻译**：source 和 target 只有 marginal samples，没有 paired demonstrations。

OT / Wasserstein / GW / SB 的共同价值在于：它们不要求一一配对数据，而是通过分布、轨迹或结构匹配建立跨域关系。

## 2. 方法谱系

| 谱系 | 代表论文 | 对应问题 | 关键对象 |
|---|---|---|---|
| OT imitation reward | PWIL, Sinkhorn IL, ROT, TemporalOT | 从示范或视频构造 reward | state-action / trajectory occupancy |
| GW cross-domain IL | GWIL | 不同状态空间 / 不同 embodiment | relational structure |
| Offline observation matching | PW-DICE, ZILOT | observation-only / offline imitation | state occupancy / goal occupancy |
| Human-robot latent alignment | EgoBridge, RHyME | human video/egocentric data -> robot | policy latent / video sequence |
| Sim-real co-training | Guided OT Sim-and-Real Co-Training, Affine Transport | sim demos + few real demos -> real policy | feature-action distribution / transition distribution |
| SB / entropy-regularized OT | Schrödinger Bridge Flow, BDGxRL | unpaired image/transition transport | dynamic path measure |
| Paired / structured image SB | I²SB, SBUnfold | paired degradation / simulation correction | informative source prior -> clean/particle level |
| Task-aware generalized SB | DeepGSB, GSBM | path constraints / mean-field interaction | state cost + boundary marginals |
| Temporal multi-marginal SB | 3MSBM | sparse snapshots | phase-space spline |
| Energy-only adjoint sampler | Adjoint Sampling, ASBS | target known only through energy | Boltzmann / conformer samples |
| Function-space adjoint sampler | FAS | 目标只有能量泛函的整条函数/轨迹（energy-only，非多时刻样本） | function-space path |
| Discrete adjoint | DAM, Discrete ASBS | token reward tuning / discrete energy | CTMC transition rate |

## 3. 最值得关注的三条主线

### 3.1 Joint feature-action OT 是当前最实用路线

`EgoBridge` 和 `Generalizable Domain Adaptation for Sim-and-Real Policy Co-Training` 都把对齐目标放在 joint feature-action distribution 上。这比单纯对齐图像特征更稳，因为 robot policy 真正需要的是 action-relevant representation。

这条线最适合马上转成实验：给 sim/real 或 human/robot 数据都过同一个 encoder，用 OT/UOT 对齐 latent-action pair，再训练共享 policy。注意这是 §4.4 定义的**表示对齐范式**（只加对齐损失、不产出翻译数据），与 SB/GSBM 生成式 transport 不可平替；两者的叠加与归因口径见 §4.4。

### 3.2 Sequence-level OT 比 frame-level matching 更适合视频

`RHyME` 和 `TemporalOT` 都说明跨域视频不能逐帧硬对齐。人类和机器人执行速度、顺序、视角都不同，必须使用 sequence-level distance、DTW、temporal cyclic consistency 或 order-aware OT。

这对下一步 video/world-model 方向很关键：如果做 action-conditioned SB video transport，需要保留任务阶段和 temporal order。

### 3.3 SB 更适合做无配对 translation / dynamics bridge

`Schrödinger Bridge Flow` 提供 unpaired data translation 工具；`BDGxRL` 把 DSB 用在 source/target transition dynamics 对齐。这两篇提示：SB 不一定先做 policy generation，反而更适合做 sim-to-real adapter。

最短路径仍是 `SB-Render-Lite`：sim RGB/latent -> real RGB/latent。第二阶段再做 `(s,a,s')_sim -> (s,a,s')_real-like`。

### 3.4 新增文献把 SB 从“分布匹配”推进到任务结构匹配

`I²SB` 证明 paired source-target 场景可以直接从信息丰富的 source image 启动 bridge，大幅降低采样步数；`SBUnfold` 进一步展示 simulation-trained bridge 在 target data 很少时的稳定性。对真正 unpaired 的 sim-real 数据，这两篇应作为 paired/pretraining 对照，而不是直接宣称解决无配对问题。

`GSBM` 是更关键的升级：它允许在路径动能之外加入可微状态代价。因此可以把 keypoint/depth、inverse dynamics、task phase 与安全约束写入 transport objective。`3MSBM` 的多时间边缘思想则启发我们：视频/rollout 或许不应只做相邻两帧的 pairwise bridge，而可在位置—速度相空间同时满足多个时间边缘（注：该论文面向稀疏 population snapshots，本身未在视频/rollout 上验证，此处为本库的外推）。

`Adjoint Sampling`/`ASBS` 面向另一种设置：target 没有足够样本，只有未归一化能量。ASBS 允许直接用 sim distribution 或物理先验作为 source prior；若 target real samples 充足，则普通 GSBM/SB Flow 仍更直接。

## 4. 对 `SB-Render-Lite` 的实验建议

### 4.1 最小实验问题

给定 unpaired sim robot frames 和 real robot frames，学习 transport：

```text
sim observation latent -> real-style observation latent
```

但 transport cost 不应只看视觉真实性，而要保留：

- 物体身份；
- 几何位置；
- 任务阶段；
- action label 有效性；
- temporal consistency。

### 4.2 Baseline

必须包含：

- sim-only BC；
- domain randomization；
- CycleGAN / CUT / pixel-level translation；
- feature-level MMD / adversarial alignment；
- OT/UOT joint feature-action alignment（表示对齐范式，§4.4 范式 A）；
- SB Flow visual transport（生成式 transport 范式，§4.4 范式 B；与上一行分属不同范式，作为独立基线臂报告，不可互换，叠加与归因按 §4.4 执行）；
- I²SB paired restoration（synthetic-paired 设置的上界参照，与 unpaired 结果分开报告，见 §5）；
- deterministic OT-ODE / flow matching 对照（检验随机性是否必要，与扩展综述 §7 最小路线一致）。

### 4.3 主指标

主指标应是 downstream policy success rate。辅助指标可以是：

- DINO/CLIP feature preservation；
- keypoint / depth / object pose consistency；
- inverse dynamics consistency；
- TemporalOT reward consistency；
- sim-real latent distribution overlap。
- trajectory velocity/acceleration smoothness 与 multi-time marginal error；
- 不同 real-data budget 下的性能—数据效率曲线；
- NFE、energy/model evaluation、wall-clock 与显存。

### 4.4 范式耦合定义：表示对齐 vs 生成式 transport

> 本节响应 R10 审查（2026-08-14）指出的 P1 缺口"对齐 vs 生成范式耦合未定义"。技术依据为 [E18 transport→policy 接口专题](../topics/E18_transport_policy_interface.md) 与 R10 建议 2 的修正方案；E18/R10 无依据处已显式标注。

**两种范式的定义（不可平替）**：

- **范式 A：表示对齐（representation alignment）**——EgoBridge / Guided OT co-training 式。在 policy 训练内部对 latent 或 joint (feature, action) 分布施加 OT/UOT 对齐损失，作用于**表示空间**（encoder 辅助损失）；**不产出任何翻译数据**，输出物是共享表示与其上的策略。
- **范式 B：生成式 transport（generative transport）**——SB Flow / I²SB / GSBM 式。训练一个桥/翻译器，作用于**数据空间**，**产出 real-style 的翻译样本**（观测或轨迹）；输出物是翻译后的数据集（或在线增广分布），再进入 policy 训练。

两者训练目标、输出物、下游用法与评估对象（对齐质量 vs 生成质量）完全不同（R10 建议 2）。库内任何"把 OT 对齐替换为 SB transport"（或反向）的表述都是**换问题**而非换方法，不构成公平对比，也不得作为实验设计依据。

**耦合口径（何时可叠加、如何叠加、如何归因）**：

1. **可叠加的前提是作用位置不同**：范式 B 改数据、范式 A 改表示，互不占用同一自由度。标准叠加顺序（E18 §4.2/§4.4）：范式 B 先产出翻译数据 → 翻译数据按 co-training 接口进入 policy 训练（real 须显著 oversample、per-batch real 采样权重恒 >0，翻译/仿真数据总量比 real 大 1–2 个数量级）→ 范式 A 的对齐损失作为**同一训练阶段的正则项**叠加其上。
2. **叠加时的损失组织受配比约束**（E18 引 Lei et al. 机理分析）：OT/UOT 对齐损失在 balanced 配比区间内有效，极端配比下会把表征拽向主导域、产生负迁移；更稳的组织方式是"对齐 + 域可辨识"双管齐下——域标签/CFG 与 OT 对齐并用。
3. **范式 B 的目标设定修正**（E18 引 Wei et al.）：不要把"翻译到与真实不可分"当作目标函数——完美渲染反而有害（策略需要能辨识域以适配残余动力学差异）；transport 应对齐 action-relevant 结构（物体位姿、接触几何、光照大形），并保留或显式提供域身份。
4. **统一 cost 纪律贯穿两层**（E18：COT Policy × Guided OT 双证据）：无论范式 B 的 minibatch coupling 还是范式 A 的特征对齐，ground cost 必须包含条件/动作/任务状态等决定"语义可交换性"的变量，否则 OT 的几何贪心会制造系统性错配。
5. **消融归因采用 A→B→A+B 递进**（R10 建议 2 修正方案）：(i) 仅对齐（无生成器，最便宜，先做）；(ii) 仅 transport（翻译数据 + co-training，无对齐损失）；(iii) 叠加。三臂之间冻结同一 co-training 配比与 policy 配方（配比敏感性证据见 E18 §4.2），并加入"增广强度匹配对照"排除正则化效应解释（协议见 §4.5 与 E11 的 L4 签名检验）。
6. 更紧的耦合形式——如用 DTW/OT plan 配对初始化 GSBM coupling（R10 建议 2 架构假设 C）——**未在文献验证（本库设计决策）**，仅作为 (iii) 通过后的探索项。

### 4.5 真机评估统计协议

> 本节响应 R10 审查（2026-08-14）指出的 P1 缺口"真机评估统计协议缺失"，口径直接取自 [E11 SimplerEnv 评测协议专题](../topics/E11_simplerenv_eval_protocol.md) 的"四层指标 + 两档协议"方案。**本协议对全库所有实验建议一律生效**：任何文档中的实验建议凡涉及评估，均按本节（及 E11 完整版）执行，不得另立口径。

- **指标框架（四层，E11 §3.1）**：L1 图像/特征代理指标只记录、不先验决策——未通过相关性验证的代理指标不得用于选型（SimplerEnv 实测 validation MSE 与真机成功率可呈负相关）；L2 下游 policy success 为王指标（双栏 ID/OOD，单列 safety 与部分成功）；L3 sim-real 相关性验证——sim 评估器须达 MMRV ≤ 0.10 且 Pearson r ≥ 0.85 才可作筛选工具，代理指标须 |Spearman| ≥ 0.7 且 MMRV ≤ 0.15 才可用于选型；L4 因子化扰动鲁棒性 + 签名检验。
- **试次数（E11 §3.2）**：真机独立比较检测 15pp 效应约需 **170 rollouts/臂**（双侧 α=0.05、power 0.8，两比例 z 检验）；每配置仅 25 rollouts 时 95% Wilson 区间半宽约 ±18pp，成功率差异基本不可判读。因此大规模筛选全部放进已通过 L3 验证的 sim 评估器；**真机采用配对设计**——确定性初始状态网格逐 trial 配对 + McNemar 检验：MVP 档每任务 ≥25 配对网格点（真机终检每臂聚合 ≥75），完整版关键对比每任务每臂 ≥50 配对 rollouts（聚合 ≥100）。
- **种子（E11 §3.1-L2、§3.2）**：每配置训练种子 ≥3（完整版 ≥5）；随机推理头（扩散/SB 采样）另加 ≥3 个推理种子并平均；种子间标准差与均值并报。
- **置信区间与检验（E11 §3.2）**：单任务数字必须带 95% Wilson 区间；主结论以 ≥3–4 个任务的任务平均报告，聚合区间用任务分层 bootstrap；任何"方法 A 优于 B"的声明需在种子平均意义上成立且区间不交叠（或配对检验 p<0.05）；sim 大规模扫描以 Benjamini-Hochberg 控 FDR；效应低于可判阈值时如实报告"真机不可判定"，禁止用点估计差下结论。
- 两档协议（MVP / 完整版）的逐条预算与验收门槛表见 E11 §3.3–3.4，本文不复制。

## 5. 当前判断

如果目标是做一个能立项的 embodied OT/SB 方向，不建议继续泛泛说“OT 可以缩小 domain gap”。更强的表述是：

> 用 action-aware / sequence-aware / dynamics-aware 的 entropy-regularized transport，把跨域具身数据从“不可直接混训”变成“可保留任务结构的共享策略训练数据”。

在这批论文里，`EgoBridge` 和 `Generalizable Domain Adaptation for Sim-and-Real Policy Co-Training` 是 joint feature-action alignment 的直接入口（§4.4 表示对齐范式）；`I²SB` / `Schrödinger Bridge Flow` 分别提供 paired / unpaired 视觉基线（§4.4 生成式 transport 范式）；`GSBM` 是生成式 transport 一侧加入 task-aware path cost 的首选升级——升级对象是 SB Flow/I²SB，而非对齐路线的替代品，两范式的叠加与归因口径见 §4.4；`3MSBM` 和 `BDGxRL` 分别支持 trajectory 与 dynamics bridge。完整的新增 11 篇关系见 [扩展文献综述](./sb_adjoint_extended_synthesis.md)。

## 6. 扩充轮更新导读（2026-08-14）

本轮扩充（E01–E20，详见 `topics/`）中与本文选型直接相关的最重要更新：

- **求解器选型**：DSBM 的 IMF 是 unpaired 场景「边缘保持」的正统基线，库内 SB Flow（α-DSBM）是其在线化；DDBM/ASBM/LightSB 等竞品与 NFE/成本对照见 E01–E03。
- **Adjoint 源头**：Adjoint Matching（Domingo-Enrich et al., ICLR 2025 Spotlight）是 AS/ASBS/DAM 的共同源头，reward 微调与能量采样是一源两支（E06）；更早的 SOC 采样器源头 PIS/DDS/CMCD 见 E14。
- **竞争范式**：DR/GAN 翻译经典基线（E08）、GS 渲染与 real-to-sim 数字孪生系统（E09）、世界模型数据引擎（E12）与零训练扩散翻译基线（E17）是论文对照组必备；iDEM/NETS 等能量采样竞品见 E15。
- **评测协议**：SimplerEnv 的 MMRV/Pearson 排序一致性方法与真机统计功效计算（检测 15pp 效应约需 170 rollouts/臂）应作为本文 §4.3 指标清单的协议底座；FID/LPIPS 等代理指标在通过与下游 success 的相关性验证前不得用于选型（E11）。该协议已回填为本文 §4.5「真机评估统计协议」，对全库实验建议生效。
- **接口配比**：翻译数据喂给 diffusion/flow policy 时，real 数据须显著 oversample 且占比不可为零，且视觉翻译不必也不应追求与真实完全不可分；条件生成中 naive minibatch OT coupling 有偏，条件必须写进 ground cost（E18）。E18 的接口结论已扩展为本文 §4.4「范式耦合定义」。

其余主题（FM/RF/SI 理论桥 E04–E05、语义增广 E13、OT 理论工具 E19、SB 逆问题 E20 等）见 `topics/` 目录。
