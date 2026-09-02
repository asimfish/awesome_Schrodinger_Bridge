# 中文译本视觉 QA 报告

生成时间：2026-09-02。翻译引擎：[SuperTranslate](https://github.com/asimfish/super_translate)（DeepSeek 后端，保版式，`--preserve-graphics-text`）；QA 由引擎 `inspect` 子命令逐页比对原文/译文产出（页数一致性、图像/公式丢失、文字重叠、字号漂移、列表字号不一致等）。

判定口径：**通过** = 0 个 error 级 issue；**通过（有备注）** = 全部 error 属于局部排版类（字号缩放/列表字号）或 ≤12 词的公式碎片保留英文（如 `is, T = T_aff[X, Y].`），译文内容完整、可读，问题位置已列出供读者知悉；**需人工复核** = 存在成段（>12 词）未译或表格结构错位；参考文献与图内文字按设计保留英文。

| arXiv | 论文 | 页数（原/译） | issues | errors | 状态 | error 位置 |
|---|---|---|---|---|---|---|
| 2006.04678 | [Primal Wasserstein Imitation Learning](./2006.04678_primal_wasserstein_imitation_learning.zh.pdf) | 19/19 | 0 | 0 | ✅ 通过 | — |
| 2008.09167 | [Imitation Learning with Sinkhorn Distances](./2008.09167_sinkhorn_imitation_learning.zh.pdf) | 15/15 | 3 | 3 | ✅ 通过（有备注） | p3 untranslated_block; p3 untranslated_block; p7 font_size_drift |
| 2105.11739 | [Affine Transport for Sim-to-Real Domain Adaptation](./2105.11739_affine_transport_sim2real.zh.pdf) | 7/7 | 3 | 3 | ✅ 通过（有备注） | p3 untranslated_block; p3 untranslated_block; p4 untranslated_block |
| 2110.03684 | [Cross-Domain Imitation Learning via Optimal Transport](./2110.03684_gwil_cross_domain_imitation_via_ot.zh.pdf) | 15/15 | 1 | 0 | ✅ 通过 | — |
| 2206.15469 | [Watch and Match: Supercharging Imitation with Regularized Optimal Transport](./2206.15469_rot_watch_and_match.zh.pdf) | 12/12 | 0 | 0 | ✅ 通过 | — |
| 2209.09893 | [Deep Generalized Schrödinger Bridge](./2209.09893_deep_generalized_schrodinger_bridge.zh.pdf) | 27/27 | 4 | 4 | ✅ 通过（有备注） | p5 font_size_drift; p5 font_size_drift; p21 untranslated_block; p21 untranslated_block |
| 2209.12093 | [Learn what matters: cross-domain imitation learning with task-relevant embeddings](./2209.12093_task_relevant_embeddings_cross_domain_il.zh.pdf) | 18/18 | 0 | 0 | ✅ 通过 | — |
| 2302.05872 | [I²SB: Image-to-Image Schrödinger Bridge](./2302.05872_i2sb.zh.pdf) | 21/21 | 0 | 0 | ✅ 通过 | — |
| 2308.12351 | [Improving Generative Model-based Unfolding with Schrödinger Bridges](./2308.12351_sb_unfold.zh.pdf) | 11/11 | 0 | 0 | ✅ 通过 | — |
| 2310.02233 | [Generalized Schrödinger Bridge Matching](./2310.02233_generalized_schrodinger_bridge_matching.zh.pdf) | 26/26 | 7 | 7 | ⚠️ 需人工复核 | p3 untranslated_block; p4 font_size_drift; p7 font_size_drift; p7 font_size_drift; p16 font_size_drift; p17 table_structure_mismatch; p24 untranslated_block |
| 2311.01331 | [Offline Imitation from Observation via Primal Wasserstein State Occupancy Matching](./2311.01331_primal_wasserstein_state_occupancy.zh.pdf) | 25/25 | 6 | 5 | ✅ 通过（有备注） | p2 font_size_drift; p2 font_size_drift; p2 font_size_drift; p18 font_size_drift; p18 font_size_drift |
| 2404.13430 | [React-OT: Optimal Transport for Generating Transition State in Chemical Reactions](./2404.13430_react_ot.zh.pdf) | 32/32 | 1 | 1 | ⚠️ 需人工复核 | p5 untranslated_block |
| 2409.06615 | [One-Shot Imitation under Mismatched Execution](./2409.06615_rhyme_one_shot_mismatched_execution.zh.pdf) | 8/8 | 0 | 0 | ✅ 通过 | — |
| 2409.09347 | [Schrödinger Bridge Flow for Unpaired Data Translation](./2409.09347_schrodinger_bridge_flow_unpaired_translation.zh.pdf) | 56/56 | 13 | 13 | ✅ 通过（有备注） | p8 untranslated_block; p19 untranslated_block; p22 untranslated_block; p23 untranslated_block; p31 untranslated_block; p32 untranslated_block; p35 untranslated_block; p37 untranslated_block; p37 untranslated_block; p37 untranslated_block; p37 untranslated_block; p38 untranslated_block; p38 untranslated_block |
| 2410.08751 | [Zero-Shot Offline Imitation Learning via Optimal Transport](./2410.08751_zero_shot_offline_il_ot.zh.pdf) | 37/37 | 2 | 2 | ✅ 通过（有备注） | p5 font_size_drift; p5 font_size_drift |
| 2410.21795 | [Robot Policy Learning with Temporal Optimal Transport Reward](./2410.21795_temporal_ot_reward.zh.pdf) | 19/19 | 2 | 2 | ✅ 通过（有备注） | p2 font_size_drift; p2 font_size_drift |
| 2504.11713 | [Adjoint Sampling: Highly Scalable Diffusion Samplers via Adjoint Matching](./2504.11713_adjoint_sampling.zh.pdf) | 37/37 | 5 | 5 | ⚠️ 需人工复核 | p5 font_size_drift; p6 font_size_drift; p19 untranslated_block; p26 font_size_drift; p27 untranslated_block |
| 2506.10168 | [Momentum Multi-Marginal Schrödinger Bridge Matching](./2506.10168_momentum_multi_marginal_sbm.zh.pdf) | 31/31 | 3 | 3 | ✅ 通过（有备注） | p4 untranslated_block; p16 untranslated_block; p24 untranslated_block |
| 2506.22565 | [Adjoint Schrödinger Bridge Sampler](./2506.22565_adjoint_schrodinger_bridge_sampler.zh.pdf) | 30/30 | 3 | 2 | ✅ 通过（有备注） | p17 untranslated_block; p22 untranslated_block |
| 2509.18631 | [Generalizable Domain Adaptation for Sim-and-Real Policy Co-Training](./2509.18631_guided_ot_sim_real_policy_cotraining.zh.pdf) | 29/29 | 5 | 5 | ⚠️ 需人工复核 | p4 untranslated_block; p5 font_size_drift; p5 font_size_drift; p5 font_size_drift; p5 font_size_drift |
| 2509.19626 | [EgoBridge: Domain Adaptation for Generalizable Imitation from Egocentric Human Data](./2509.19626_egobridge.zh.pdf) | 23/23 | 1 | 1 | ✅ 通过（有备注） | p13 font_size_drift |
| 2511.06239 | [Functional Adjoint Sampler: Scalable Sampling on Infinite Dimensional Spaces](./2511.06239_functional_adjoint_sampler.zh.pdf) | 33/33 | 12 | 12 | ⚠️ 需人工复核 | p4 untranslated_block; p4 untranslated_block; p17 font_size_drift; p17 font_size_drift; p17 font_size_drift; p17 font_size_drift; p17 untranslated_block; p18 untranslated_block; p20 untranslated_block; p21 untranslated_block; p23 untranslated_block; p28 untranslated_block |
| 2602.07132 | [Discrete Adjoint Matching](./2602.07132_discrete_adjoint_matching.zh.pdf) | 36/36 | 7 | 7 | ⚠️ 需人工复核 | p1 untranslated_block; p6 untranslated_block; p6 untranslated_block; p9 untranslated_block; p15 untranslated_block; p16 untranslated_block; p16 untranslated_block |
| 2602.08243 | [Discrete Adjoint Schrödinger Bridge Sampler](./2602.08243_discrete_adjoint_schrodinger_bridge_sampler.zh.pdf) | 30/30 | 2 | 2 | ✅ 通过（有备注） | p3 list_font_inconsistent; p6 font_size_drift |
| 2602.23737 | [Bridging Dynamics Gaps via Diffusion Schrödinger Bridge for Cross-Domain Reinforcement Learning](./2602.23737_bdgxrl_diffusion_schrodinger_bridge.zh.pdf) | 12/12 | 1 | 1 | ✅ 通过（有备注） | p12 font_size_drift |

**汇总**：通过 7 · 通过（有备注）12 · 需复核 6 · 共 25 篇。

复现：`bash scripts/translate_batch.sh 3` 生成译本并自动 QA；`bash scripts/translate_retry.sh` 对含 error 的译文用缓存重做；`python3 scripts/qa_table.py` 重建本表。
