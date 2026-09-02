# 中文精读报告索引

本目录收录 awesome_Schrödinger_Bridge 的 25 篇核心论文精读（每篇：基本信息 / 一句话总结 / 方法核心 / 实验与结果 / 局限性 / 与相关方向的关系）与 4 份综合文档。英文 PDF 在 [`../papers/`](../papers/)，保版式中文译本在 [`../papers_zh/`](../papers_zh/)（文件名同名、后缀 `.zh.pdf`），20 份专题笔记在 [`../topics/`](../topics/)。本索引按「与具身跨域迁移项目的相关性」分区；按方法族的谱系见 [synthesis.md §2](./synthesis.md)，按主题的总清单见仓库根 [README](../README.md)。

> 归类轴说明（2026-08-27 依 R10 P2 加注）：本索引按**与 `SB-Render-Lite` 项目的相关性**分区；**方法族谱系**见 [synthesis §2](./synthesis.md)；`metadata/papers.tsv` 的 `category` 列是第三套机读标签。三套轴各有用途，同一论文在三处的归属可能不同，以各自口径为准。

## 综合入口

- [综合文献地图：OT / SB 如何迁移具身跨域数据](./synthesis.md)
- [Adjoint / Generalized / Structured Schrödinger Bridge 扩展文献综述](./sb_adjoint_extended_synthesis.md)
- [Schrödinger Bridge × Optimal Transport × Sim2Real：深度调研、前沿论文与学习资源导航](./deep_research_learning_resources.md)
- [Guan-Horng Liu 研究工作专题：从最优控制到 SB、Adjoint Sampling 与 LLM Post-training](./guan_horng_liu_research_roadmap.md)

## 核心：人类数据 / 机器人数据迁移

- [EgoBridge 专项深读：从 egocentric human data 到 robot policy 的 OT 对齐](./2509.19626_egobridge_deep_dive.md)
- [EgoBridge: Domain Adaptation for Generalizable Imitation from Egocentric Human Data](./2509.19626_egobridge.md)
- [One-Shot Imitation under Mismatched Execution](./2409.06615_rhyme_one_shot_mismatched_execution.md)
- [Robot Policy Learning with Temporal Optimal Transport Reward](./2410.21795_temporal_ot_reward.md)——方法族上属 OT imitation reward 族（synthesis §2 与 papers.tsv 归 `robot_video_reward_ot`，与 PWIL/ROT 同源）；因其 reward 直接面向 robot 视频演示、与本区问题设定强相关而保留在核心区（2026-08-27 依 R10 P2 加注，消解三文档口径分歧）

## 核心：仿真 / 真机迁移

- [Generalizable Domain Adaptation for Sim-and-Real Policy Co-Training](./2509.18631_guided_ot_sim_real_policy_cotraining.md)
- [Affine Transport for Sim-to-Real Domain Adaptation](./2105.11739_affine_transport_sim2real.md)
- [Bridging Dynamics Gaps via Diffusion Schrödinger Bridge for Cross-Domain Reinforcement Learning](./2602.23737_bdgxrl_diffusion_schrodinger_bridge.md)

## 方法基础：跨状态空间 / offline observation / zero-shot imitation

- [Cross-Domain Imitation Learning via Optimal Transport](./2110.03684_gwil_cross_domain_imitation_via_ot.md)
- [Offline Imitation from Observation via Primal Wasserstein State Occupancy Matching](./2311.01331_primal_wasserstein_state_occupancy.md)
- [Zero-Shot Offline Imitation Learning via Optimal Transport](./2410.08751_zero_shot_offline_il_ot.md)
- [Primal Wasserstein Imitation Learning](./2006.04678_primal_wasserstein_imitation_learning.md)
- [Imitation Learning with Sinkhorn Distances](./2008.09167_sinkhorn_imitation_learning.md)
- [Watch and Match: Supercharging Imitation with Regularized Optimal Transport](./2206.15469_rot_watch_and_match.md)——OT imitation reward 族（与 PWIL/Sinkhorn IL 同族），不直接解决 human-to-robot domain gap（2026-08-26 依 R10 P1 自"核心：人类数据 / 机器人数据迁移"区移入，统一 README/INDEX/synthesis 三处口径）

## 重要对照

- [Learn what matters: cross-domain imitation learning with task-relevant embeddings](./2209.12093_task_relevant_embeddings_cross_domain_il.md)

## SB 方法底座：unpaired 翻译

- [Schrödinger Bridge Flow for Unpaired Data Translation](./2409.09347_schrodinger_bridge_flow_unpaired_translation.md)——项目 unpaired 翻译基线主力，README 结论 #8 与 synthesis §3.3 称其为 `SB-Render-Lite` 的直接方法底座（2026-08-14 依 R10-S6 自"重要对照"区移入）

## SB 图像、科学数据与确定性 OT 应用

- [I²SB: Image-to-Image Schrödinger Bridge](./2302.05872_i2sb.md)
- [Improving Generative Model-based Unfolding with Schrödinger Bridges](./2308.12351_sb_unfold.md)
- [React-OT: Optimal Transport for Generating Transition State in Chemical Reactions](./2404.13430_react_ot.md)

## Generalized / Multi-Marginal Schrödinger Bridge

- [Deep Generalized Schrödinger Bridge](./2209.09893_deep_generalized_schrodinger_bridge.md)
- [Generalized Schrödinger Bridge Matching](./2310.02233_generalized_schrodinger_bridge_matching.md)
- [Momentum Multi-Marginal Schrödinger Bridge Matching](./2506.10168_momentum_multi_marginal_sbm.md)

## Adjoint Sampler 方法线

- [Adjoint Sampling: Highly Scalable Diffusion Samplers via Adjoint Matching](./2504.11713_adjoint_sampling.md)
- [Adjoint Schrödinger Bridge Sampler](./2506.22565_adjoint_schrodinger_bridge_sampler.md)
- [Functional Adjoint Sampler: Scalable Sampling on Infinite Dimensional Spaces](./2511.06239_functional_adjoint_sampler.md)
- [Discrete Adjoint Matching](./2602.07132_discrete_adjoint_matching.md)
- [Discrete Adjoint Schrödinger Bridge Sampler](./2602.08243_discrete_adjoint_schrodinger_bridge_sampler.md)

## 扩充专题（2026-08-14）

20 份专题笔记（方法谱系、基线协议、评测方案），存放于 [`../topics/`](../topics/)：

- [E01 DSBM 精读 + SB 求解器谱系](../topics/E01_dsbm_solver_lineage.md)：梳理 IPF/DSB → IMF/DSBM → α-DSBM/SB Flow 谱系，确立 DSBM-IMF 为 unpaired 翻译的"边缘保持"正统基线，熵正则 σ² 是 realism–alignment 旋钮。
- [E02 DDBM 精读 + IDBM 定理笔记](../topics/E02_ddbm_idbm.md)：paired 翻译应设 I²SB + DDBM 双基线；IDBM 给出 bridge matching 路线"桥混合→Markov 化→迭代收敛到 SB"的理论地基。
- [E03 ASBM + LightSB 轻量求解器对照](../topics/E03_asbm_lightsb_costs.md)：NFE/训练成本/适用维度对照表；像素空间低 NFE 主线选 ASBM 型 D-IMF，LightSB/LightSB-M 作 latent 上的分钟级探针与 ε 扫描工具。
- [E04 Flow Matching 精读 + minibatch coupling 设计笔记](../topics/E04_flow_matching_coupling.md)：CFM 条件回归原理与 coupling 选择净结论；entropic ε 谱（ε=2σ² 恰对应 SB）把 FM coupling 设计与 SB Flow/GSBM 衔接。
- [E05 Rectified Flow 与 Stochastic Interpolants](../topics/E05_rf_stochastic_interpolants.md)：补上 simulation-free flow/interpolant 方法地基；RF 提供直线化+蒸馏的一步推理路线，SI 提供统一 flow/diffusion/bridge/SB 的插值框架语言。
- [E06 Adjoint Matching 源头精读](../topics/E06_adjoint_matching_origin.md)：memoryless SOC + lean adjoint 回归的谱系源头；厘清 AM/DAM（reward 微调）与 AS/ASBS/FAS/DASBS（能量采样）两条支流。
- [E07 Diffusion Reward 对齐谱系](../topics/E07_diffusion_reward_alignment.md)：DDPO 精读 + 四路线（直接反传/policy gradient/偏好优化/SOC-adjoint）权衡，给出以 policy success 为黑盒 reward 的三段式对齐路线。
- [E08 DR + GAN 翻译经典基线](../topics/E08_dr_gan_baselines.md)：RCAN/RetinaGAN 精读；感知一致性约束是 GAN sim2real 路线胜负手，附经典基线协议规格与各方法真实数据需求谱系。
- [E09 SplatSim + RialTo（real2sim 竞品上篇）](../topics/E09_splatsim_rialto_real2sim.md)：钉死重建/渲染路线的输入假设与真机协议；photorealistic 重建后仍有残余外观 gap，SB 的差异化主张是分布级、免重建、跨场景摊销。
- [E10 LucidSim + X-Sim 接口精读](../topics/E10_lucidsim_xsim_interface.md)：生成式增广与 real-to-sim-to-real 系统的接口结论——几何硬条件生成、on-policy 数据占大头、部署期 replay 制造 real/sim 成对图像在线校准编码器。
- [E11 SimplerEnv 评测协议 + 评测方案草案](../topics/E11_simplerenv_eval_protocol.md)：响应 R10-T2 缺口；sim 评估以保策略排序为目标（MMRV+Pearson），落地"四层指标 + 两档协议"评测方案，真机功效计算约 170 rollouts/臂。
- [E12 世界模型数据引擎](../topics/E12_worldmodel_data_engine.md)：DreamGen 精读 + UniSim 半精读；"代理指标须与下游收益建立相关性"的评估逻辑与"按离 policy 距离分层"的指标排序可直接迁移。
- [E13 扩散语义增广基线](../topics/E13_diffusion_semantic_aug.md)：ROSIE 精读；inpainting 只动 mask 内语义、不动全图外观统计，与 SB transport 构成 domain gap 分量分解对照，附 P-oracle/P-blind 公平协议。
- [E14 SOC 采样器源头](../topics/E14_soc_sampler_origins.md)：PIS/DDS/CMCD 谱系定位；把 Adjoint 线解决的三大瓶颈（全轨迹反传、on-policy 耦合、先验限制）精确命名。
- [E15 能量采样竞品横评](../topics/E15_energy_sampler_competitors.md)：iDEM/NETS/Sendera vs AS/ASBS；adjoint 线在"无偏 + 全模态"维度是短板而非长板，评测须补 EUBO/前向指标/mode-coverage 口径。
- [E16 Latent Bridge 与少步部署](../topics/E16_latent_bridge_fewstep.md)：LBM 的 1 NFE latent bridge 配方与 CDBM"先训 bridge 再压缩"工具链，给出 paired/unpaired 两种数据情形的部署裁决。
- [E17 Zero-shot 翻译基线](../topics/E17_zeroshot_translation_baselines.md)：DDIB 精读 + SDEdit 收录；两段 SB 拼接≠跨域 OT、exact cycle consistency≠对齐，语义漂移是对机器人数据最危险的失败模式。
- [E18 transport→policy 接口与 co-training 配比](../topics/E18_transport_policy_interface.md)：COT Policy 的条件 coupling 负结果、Diffusion Policy 的分布保真特性、oversample real 的定量配比；"完美渲染反而有害"修正翻译目标设定。
- [E19 OT 理论工具箱](../topics/E19_ot_theory_toolbox.md)：UOT/GW/FGW/UGW 准确口径与选参诊断；类别失衡时 balanced coupling 数学上必然错配，coupling 质量必须用独立于视觉质量的指标验证。
- [E20 SB 逆问题 × Trajectory Inference 横断综述](../topics/E20_sb_inverse_trajectory.md)：CDSB 的条件参考测度与 forward-backward 采样、逆问题家族分工判据、held-out marginal 评测协议向 sim2real 的移植。


## 对当前 `SB-Render-Lite` 的直接启发

最值得直接转化为实验设计的是 `EgoBridge`、`Generalizable Domain Adaptation for Sim-and-Real Policy Co-Training` 与 `GSBM`：前两者在 policy latent / feature-action joint distribution 上对齐，后者允许把 action/geometry consistency 直接写成路径状态代价。`I²SB` 是 paired restoration 强基线，`SB Flow` 是 unpaired 基线，`3MSBM` 则适合后续 trajectory/video bridge。所有视觉指标都应服从 downstream real-domain policy success。
