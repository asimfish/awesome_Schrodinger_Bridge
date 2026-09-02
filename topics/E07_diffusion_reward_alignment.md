# E07 扩充报告：Diffusion Reward 对齐谱系 —— DDPO 精读 + 四路线综述

## 选题定位

本库已有 reward 微调的 SOC-adjoint 一条线的完整精读（Adjoint Sampling、ASBS、FAS、DAM、Discrete ASBS，见 `../reports/`），但"用 reward 信号微调 diffusion/flow 生成模型"这一话题还有三条主流路线未入库：policy gradient 类、直接反传类、偏好优化类。本报告补全这三条路线，使"reward 微调"在库内形成完整版图，并回答 SB-Render-Lite 的核心选型问题：**用 policy success 信号微调 sim→real 视觉翻译器时，应该走哪条路线**。

- 精读 1 篇：DDPO（arXiv 2305.13301，policy gradient 路线的奠基工作；选它而非 DRaFT 的理由见 A.2）。
- 收录条目 3 条：DPOK（NeurIPS 2023）、DRaFT（ICLR 2024）、Diffusion-DPO（CVPR 2024）。
- 四路线谱系综述 + 对照表 + SB-Render-Lite 选型决策。
- 全文来源：arXiv abs 页 HTML 全文 + ar5iv HTML（两者内容一致），四篇均获取到全文。
- venue 纪律：所有提及论文的发表状态均经 web 复核，检索日期 **2026-08-14**，汇总见 Part D。

## TL;DR

1. **DDPO 的核心贡献是"把 denoising 过程重写为多步 MDP"**：每一步反向去噪是一个高斯策略动作，其精确 log-likelihood 可算，因此可以直接跑 REINFORCE / PPO 式 policy gradient，只需**黑盒 reward**。这一 MDP 形式化被 DPOK、D3PO 共用，也是 Diffusion-DPO 推导的参照系。
2. 四条路线的本质权衡是**样本效率 × reward 可微性要求 × 分布约束强度**：直接反传（ReFL/DRaFT/AlignProp）效率最高（比 DDPO 快约 200 倍）但要求可微 reward 且模式坍缩最严重；policy gradient（DDPO/DPOK）只需黑盒 reward 但 reward query 开销大；偏好优化（Diffusion-DPO/D3PO）唯一可完全离线、只需成对偏好标签；SOC-adjoint（Adjoint Matching 线）是唯一有"收敛到 KL 正则 tilted 分布"理论保证的路线，且与本库 SB/bridge 模型同构。
3. 对 SB-Render-Lite：policy success 是黑盒、稀疏、评估昂贵（一次 reward query = 一次策略 rollout）。推荐三段式：**离线成败标签 → Diffusion-DPO 式预对齐；小规模在线预算 → DDPO 式 policy gradient 验证；能训练可微 success critic 后 → 切换 Adjoint Matching**（memoryless 形式，与翻译器的 bridge 结构天然兼容，且自带 KL-to-base 防坍缩）。

---

# Part A 精读：DDPO — Training Diffusion Models with Reinforcement Learning

## A.1 基本信息

- 论文：Training Diffusion Models with Reinforcement Learning
- 方法名：DDPO（Denoising Diffusion Policy Optimization）
- 作者：Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, Sergey Levine（UC Berkeley / MIT）
- 会议：**ICLR 2024**（proceedings.iclr.cc 2024 收录；另有早期版本见 ICML 2023 ES-FoMO workshop。web 复核 2026-08-14）
- 链接：https://arxiv.org/abs/2305.13301
- 项目页：http://rl-diffusion.github.io（含官方实现入口）
- 全文获取：arXiv abs 页 HTML 全文 + ar5iv，均完整
- 归类：diffusion reward fine-tuning；policy gradient；RLHF/RLAIF for text-to-image。

## A.2 为什么精读 DDPO 而非 DRaFT

三点理由：

1. **谱系地位**：DDPO（与同期 DPOK）确立了"denoising = 多步 MDP"的形式化，是 policy gradient 路线的奠基工作；D3PO 直接继承该 MDP，Diffusion-DPO 的推导也以它为参照系（其附录把自身解释为同一 MDP 设定下的 off-policy 算法）。精读它可以覆盖整条路线的公共骨架。
2. **与 SB-Render-Lite 的信号形态匹配**：本库的核心用例是用 **policy success（成功率）** 微调翻译器，这是典型的黑盒、不可微 reward。DDPO 是四路线中唯一"只需能对终样本打分"的在线方法；DRaFT 则要求 reward 可微，与我们的信号形态存在根本错配。
3. **避免重复**：直接反传路线的"理论修正版"就是库内已精读多篇的 SOC-adjoint 线（Adjoint Matching 明确指出裸反传/截断反传的 value-gradient 偏差并给出修正），DRaFT 的核心思想可由收录条目 + 该线报告覆盖，不需要再花一篇精读额度。

## A.3 一句话总结

DDPO 把迭代去噪过程重写为一个 T 步 MDP——每步的反向核 `p_θ(x_{t-1}|x_t, c)` 是精确可算 likelihood 的高斯策略——从而能用 REINFORCE / PPO 式 policy gradient 直接最大化终样本的任意黑盒 reward，在压缩率、美学分和 VLM 对齐任务上全面超过 reward 加权回归（RWR）基线。

## A.4 解决的问题

Diffusion 模型的训练目标是 log-likelihood 的变分下界，但绝大多数下游用例关心的不是 likelihood，而是"人眼质量、prompt 对齐、药效"这类下游目标。难点有二：

- diffusion 的样本 likelihood `p_θ(x_0|c)` 不可精确计算，传统 RL/加权回归方法只能用变分近似，理论上不严谨（论文明确指出 RWR 类方法"only optimizes J_DDRL very approximately"）；
- 很多有价值的 reward（文件大小、VLM 判断）无法通过 prompt 工程或数据整理表达。

目标形式：`max_θ E_{c~p(c), x_0~p_θ(x_0|c)}[r(x_0, c)]`，只假设 r 可以对最终样本打分（黑盒）。

## A.5 方法核心

**RWR 基线 = 一步 MDP。** 先把 Lee et al. 2023（arXiv 2302.12192，preprint）式的 reward 加权回归写成一步 MDP：state=prompt，action=整张成品图 `x_0`，策略是 `p_θ(x_0|c)`。此时策略 likelihood 不可算，只能用 DDPM 变分下界代替 log-likelihood，加权（指数权重或阈值二值权重）后做回归——这正是它理论不严谨的原因。

**DDPO = 多步 MDP。** 关键重构：

- state `s_t = (c, t, x_t)`，action `a_t = x_{t-1}`，策略 `π(a_t|s_t) = p_θ(x_{t-1}|x_t, c)`；
- 转移是确定性的 Dirac（拼接 prompt、递减时间步、把 action 当作下一状态）；
- reward 只在最后一步给：`R = r(x_0, c)`（t=0），其余为 0。

好处：用标准采样器时每步策略就是各向同性高斯，**精确 log-likelihood 及其梯度可算**，一步 MDP 中不可解的问题变成标准 policy gradient 问题。

**两个梯度估计器。**

- `DDPO_SF`：score function / REINFORCE 估计，`∇J = E[Σ_t ∇log p_θ(x_{t-1}|x_t,c) · r(x_0,c)]`，严格 on-policy，每轮采样只能做一次更新；
- `DDPO_IS`：importance sampling 估计 + PPO 式 clip 信任域（clip range 仅 1e-4，远小于常规 RL），同一批轨迹可做多次更新（实践中 256 样本/轮、4 次梯度更新）。

**实现关键（两处容易被忽视但决定成败）：**

- **CFG training**：只训 conditional 分支会在第二轮起迅速崩坏（guidance 权重失配）。解决办法是训练时也用固定 guidance 权重下的 guided ε-prediction 作为优化对象。这对任何多轮交替"采样-训练"的 diffusion RL 都是必要 trick。
- **per-prompt reward 归一化**：对每个 prompt 独立维护 reward 的 running mean/std 做标准化，等价于 value baseline / advantage 化，处理不同 prompt 难度差异。

## A.6 实验与结果

设置：Stable Diffusion v1.4，50 步采样，只微调 UNet。四个 reward：JPEG 压缩率 / 反压缩率（398 种 ImageNet 动物 prompt）、LAION aesthetic predictor（45 种动物）、VLM 对齐（LLaVA 描述图片 + BERTScore 对 prompt 的召回，45 动物 × 3 活动）。

- **DDPO vs RWR**：以"reward 查询次数"为横轴，DDPO 两个变体在三个任务上全面超过 RWR / RWR_sparse；`DDPO_IS` 最优。定性上，美学任务把自然照片风格推向插画风，压缩任务学会删背景、平滑前景，反压缩学会生成高频噪声。
- **对比可微替代方案**（App B）：aesthetic 任务上 base 5.95 → universal guidance 6.14 → `DDPO_IS`@20k 次 reward 查询 6.63；且 universal guidance 推理慢 ~30 倍/图。说明"即使 reward 可微，训练期 RL 也可胜过推理期 guidance"。
- **VLM 对齐**：不需人工标注，"a dolphin riding a bike" 这类预训练模型成功率为零的组合 prompt 通过跨 prompt 迁移得到改善。
- **泛化**：在 45 种动物上微调，效果迁移到未见动物、非动物日常物体、甚至未见活动（"a duck taking an exam"）。
- **与 DPOK 直接对比**（App C）：换用 SD v1.5 + LoRA（lr 3e-4）、ImageReward、DPOK 的 4 个 prompt，但单模型训 4 prompt、且**不加 KL 正则**——`DDPO_IS` 全面超过 DPOK 报告数字；跨 reward 评估（用 LAION aesthetic 测 ImageReward 训练的模型）显示 25k 查询内开始出现可测的 overoptimization，但不算剧烈。
- **资源**：50k 样本约 4 小时（v4-64 TPU pod）；VLM reward 另需 8×A100 跑 LLaVA。

**Overoptimization / reward hacking（App A，重要负结果）**：反压缩任务最终退化为纯噪声；VLM 计数 prompt（"n animals"）被 typographic attack 攻破——模型学会在图上写形似数字的文字（如八只乌龟配 "sixx ttutttas"）骗过 LLaVA。论文明确不解决该问题，实践中靠人工挑"崩坏前最后一个 checkpoint"。

## A.7 局限性

- **样本（reward 查询）效率低**：DRaFT 论文测得同一 aesthetic 任务上 DDPO 需要多 200 倍以上的 reward 查询（且 DDPO 原论文的美学实验有实现 bug，修正后 50k 查询到 7.4，仍远慢于梯度法）。当单次 reward 评估昂贵时（如机器人 rollout）这是主要瓶颈。
- **无分布约束**：原始 DDPO 不加 KL-to-base，overoptimization 只能靠早停；Diffusion-DPO 的复现也报告 DDPO 在开放词表（Pick-a-Pic 全量 prompt）上训练不稳定、难以超过基线。
- prompt 分布窄（≤398 个模板化 prompt），开放词表能力未验证。
- PPO clip、per-prompt 归一化、CFG training 等超参敏感，工程复杂度不低。

## A.8 与 SB-Render-Lite 的关系 / 可借鉴点

- DDPO 的 MDP 形式化**同样适用于 SB/bridge 翻译器**：把 I²SB/GSBM 式的 sim→real 翻译链条视为多步 MDP（初态是 sim 图像而非纯噪声），每步 posterior 高斯的 likelihood 同样可算，policy gradient 可以原样套用。这是"policy success 信号（黑盒）微调翻译器"的最直接方案。
- 直接可搬的工程组件：per-prompt（对我们是 per-scene/per-task）reward 归一化；PPO 极小 clip range；CFG training 的对应物是"翻译器带条件分支时训练期与推理期 guidance 一致"。
- 必须提前设计 anti-hacking 评估：DDPO 的 typographic attack 教训对应到我们，是翻译器可能生成"骗过 success critic/判别器但物理不可信"的纹理。应在训练前固定一组不参与 reward 的独立指标（几何一致性、inverse-dynamics 一致性、真实域 policy success 的 held-out 评估）。
- reward 查询预算要按"一次查询 = 一次 rollout"来估：DDPO 在 T2I 上需要 1e4–5e4 量级查询才显著超基线，直接搬到真机不现实；这正是 Part C 决策树引入偏好优化路线与 critic + SOC-adjoint 路线的原因。

---

# Part B 收录条目（3 条）

## B.1 DPOK: Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models

- 作者：Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, Kimin Lee（Google Research / UW-Madison / Berkeley / KAIST）
- 会议：**NeurIPS 2023**（proceedings.neurips.cc 收录；web 复核 2026-08-14）
- 链接：https://arxiv.org/abs/2305.16381 ｜ 代码：google-research/dpok
- 定位：与 DDPO 同期独立的 policy gradient 工作，路线相同，差异在**KL 正则**。

**核心机制**：同样把去噪写成 T 步 MDP 跑 REINFORCE（Lemma 4.1），但目标加上逐步条件 KL 的上界作为正则（Lemma 4.2：终样本边际 KL ≤ 各步条件 KL 之和），即 `α·E[-r] + β·Σ_t KL(p_θ(x_{t-1}|x_t,z) || p_pre(...))`，在线样本上评估。论文还给 SFT 加了两种 KL（KL-D/KL-O）以公平对比，并从"在线 vs 离线分布、KL 语义、reward 模型评估点"三方面论证在线 RL 优于监督式 reward 加权微调。

**关键结果**：SD v1.5 + LoRA + ImageReward。单 prompt 设置（color/composition/count/location 四类）RL 在 ImageReward 与 aesthetic 上均优于带 KL 的 SFT，人评一致；KL 消融显示无 KL 的 RL 会出现过饱和、形状畸变。多 prompt 设置（MS-CoCo 104 条：ImageReward 0.22→0.55；Drawbench 183 条：0.13→0.58，aesthetic 基本持平）需要额外学 value function 降方差。还展示了修正预训练偏差的例子（"Four roses" 从威士忌图像修正为玫瑰，ImageReward −0.52→1.12）。

**局限**：主要在单 prompt 或 ~100–200 prompt 规模验证，多 prompt 训练"需要更长时间、更多调参与工程"（作者自述）；DDPO App C 显示在相同设置下 DDPO_IS 不加 KL 也能全面超过其报告数字。

**对库的意义**：DPOK 把 RLHF 的 "reward + KL-to-base" 目标搬进了 diffusion——这正是 Adjoint Matching 所求解的同一 tilted-distribution 目标（`p* ∝ p_base·exp(r/β)`）的 policy gradient 求法。对 SB-Render-Lite，若走 policy gradient 路线，应采用 DPOK 的 KL 正则形式而非裸 DDPO，因为翻译器"保留 sim 内容结构"的需求等价于强 KL-to-base 约束。

## B.2 DRaFT: Directly Fine-Tuning Diffusion Models on Differentiable Rewards

- 作者：Kevin Clark, Paul Vicol, Kevin Swersky, David J. Fleet（Google DeepMind）
- 会议：**ICLR 2024 poster**（OpenReview + DeepMind 出版页确认；web 复核 2026-08-14）
- 链接：https://arxiv.org/abs/2309.17400 ｜ 无官方公开代码（检索日未见）
- 定位：直接反传路线的代表作与统一框架。

**核心机制**：对可微 reward 直接求 `∇_θ r(sample(θ, c, x_T))`——把整条 50 步采样链当 RNN 做 BPTT，用 LoRA + gradient checkpointing 控制显存。两个关键变体：**DRaFT-K** 只反传最后 K 步（发现全链反传会梯度爆炸，K=1 反而最优）；**DRaFT-LV** 在 K=1 基础上对生成图加噪 n 次求平均梯度降方差（n=2 时再快约 2 倍，开销仅 ~10%）。论文给出统一视角：ReFL（NeurIPS 2023，ImageReward 论文提出，在随机中间步对一步预测的 x̂_0 求 reward 梯度）等价于该框架中 stop-gradient 位置的不同选择；ReFL(m=1) ≡ DRaFT-1。

**关键结果**：LAION aesthetic 任务比 DDPO 快 **>200×**（并注明 DDPO 原论文该实验有 bug，与作者通信后修正为 50k 查询达 7.4）；HPSv2 基准上 DRaFT-LV 达到当时最高分；LoRA 权重可缩放（在预训练与微调模型间插值）和线性混合（组合多个 reward）。同时坦率报告 **reward hacking**：训练后期多样性坍缩到少数高 reward 图像，KL 正则与早停都不如 LoRA scaling 好用。

**局限**：只适用于可微 reward；截断反传引入偏差（论文自己指出这与 exploding gradients 的权衡）；模式坍缩比 RL 路线更快出现。**与库内 SOC-adjoint 线的关系**：Adjoint Matching（ICLR 2025）证明了 DRaFT 式裸反传/截断反传对应有偏的 value-gradient 估计，且不收敛到 KL 正则最优分布；AM 用 memoryless schedule + lean adjoint 回归修正了这一偏差。可以把 DRaFT 视为"工程上最快、理论上无保证"的一端，AM 是其理论修正版。

**对库的意义**：若为 SB-Render-Lite 训练了可微 success critic，DRaFT-LV 是最快的迭代工具（适合超参扫描期），但正式训练应换 AM 以获得分布保证。

## B.3 Diffusion-DPO: Diffusion Model Alignment Using Direct Preference Optimization

- 作者：Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, Nikhil Naik（Salesforce AI / Stanford）
- 会议：**CVPR 2024**，pp. 8228–8238（openaccess.thecvf.com 收录；web 复核 2026-08-14）
- 链接：https://arxiv.org/abs/2311.12908
- 定位：偏好优化路线的代表作；首个在开放词表上稳定有效的 diffusion 对齐方法。

**核心机制**：把 LLM 的 DPO 搬到 diffusion。障碍是 `p_θ(x_0|c)` 不可算，解法是三步近似：(1) 把 reward 定义提升到整条链 `x_{0:T}` 上，用路径 KL 上界替代边际 KL；(2) 用 Jensen 不等式把期望推到 log σ 外；(3) 用前向过程 q 近似反向后验采样。最终 loss 极其好实现：对 winner/loser 图各加噪一次，比较当前模型与参考模型的去噪误差之差，过 logistic——**完全离线，无 reward model，无采样循环**。论文还从"同一 MDP 设定下的 off-policy 算法"角度给出第二种推导（与 DDPO/DPOK 直接对话），并指出隐式 reward 可用于偏好分类（DPO-SDXL 在 Pick-a-Pic v2 验证集达 72.0%，超过 PickScore 等显式 reward model）。

**关键结果**：Pick-a-Pic v2 的 851,293 对偏好（58,960 个 prompt）微调 SDXL-base（β=5000，2048 对/批，16×A100）：PartiPrompts 上 General Preference win rate **70%** vs SDXL-base；vs 更大的 SDXL base+refiner 管线仍 69%/64%（Parti/HPSv2）；SDEdit 图生图任务 65% vs 24%。AI feedback 变体（用 PickScore 伪标签重标数据再训）优于人工原始标签（Parti win rate 59.8%→63.3%）。对照：作者在 Pick-a-Pic 全量 prompt 上扫超参也无法让 DDPO 稳定超基线；SFT 在 SDXL 上任何学习率都掉点。

**局限**：三步近似（路径 KL 上界、Jensen、前向代替反向）各引入偏差，理论保证弱于 AM；纯离线，无法主动探索超出数据分布的高 reward 区域；效果上限受偏好数据质量/覆盖限制（Dreamlike 小子集实验提升有限）。

**对库的意义**：这是与 SB-Render-Lite 数据形态最契合的路线之一——"同一 sim 帧的两个翻译版本 + 下游 rollout 成/败"天然构成 `(x^w, x^l | c)` 偏好对，不需要任何在线采样与 reward model。见 C.7。

---

# Part C 四路线谱系综述

## C.1 统一问题形式

四条路线求解的都是（或近似是）：

```
max_θ  E_{x_0 ~ p_θ}[ r(x_0) ]  −  β · KL( p_θ || p_base )
```

其最优解是 tilted 分布 `p*(x_0) ∝ p_base(x_0) · exp(r(x_0)/β)`（β→0 时退化为纯 reward 最大化，坍缩风险最大）。差异在于：**用什么信号（标量 reward / reward 梯度 / 偏好对）、在哪里求导（likelihood / 采样链 / 回归目标）、以及是否真的收敛到 p***。

## C.2 路线一：policy gradient 类（黑盒 reward，在线）

- 代表：**DDPO**（ICLR 2024，本报告精读）、**DPOK**（NeurIPS 2023，B.1）；思想前身为 Fan & Lee 的 shortcut fine-tuning（ICML 2023，GAN 判别器作 reward）。近期延伸：**Flow-GRPO**（NeurIPS 2025 poster，arXiv 2505.05470）把 GRPO 组内相对优势搬到 flow matching 模型，通过 ODE→SDE 等价转换制造探索随机性 + 减步训练提效，GenEval 63%→95%，并报告 reward hacking 很轻——说明该路线在 2025 后仍在快速演进且逐步解决效率短板。
- 机制：denoising 多步 MDP + REINFORCE/PPO/GRPO；每步高斯核 likelihood 精确可算。
- 优点：**只需黑盒标量 reward**（不可微、不可查询梯度均可）；在线探索可以发现 base 分布支撑外的高 reward 区域；KL 正则可显式加入（DPOK）。
- 缺点：reward 查询效率最低（比直接反传慢两个数量级）；方差大、超参敏感；DDPO 无 KL 时靠早停防崩。
- 适用：reward 真的不可微（人评、物理仿真、**策略成功率**）、且单次评估成本可接受时。

## C.3 路线二：直接反传类（可微 reward，在线，最快）

- 代表：**ReFL**（ImageReward 论文提出，NeurIPS 2023，pp. 15903–15935）、**DRaFT**（ICLR 2024，B.2）、**AlignProp**（arXiv 2310.03739；检索日 2026-08-14 未见正式 venue 收录记录，按 preprint 引用——与 DRaFT 同期同思路：LoRA + checkpointing + 随机截断步数反传）。
- 机制：把 reward 梯度直接沿采样链 BPTT 回传到模型参数；截断（DRaFT-K / ReFL 的随机中间步）换取效率与梯度稳定。
- 优点：样本效率最高（DRaFT 比 DDPO 快 >200×）；实现直观。
- 缺点：**要求可微 reward**；截断反传有偏、全链反传梯度爆炸；reward hacking / 多样性坍缩最快最严重（DRaFT 与 Diffusion-DPO 论文都报告了这点）；无收敛到 p* 的保证。
- 适用：有可靠可微 reward（美学/偏好打分器、可微渲染指标）、追求迭代速度、能接受强正则或 LoRA scaling 兜底时。

## C.4 路线三：偏好优化类（偏好对，离线，免 reward model）

- 代表：**Diffusion-DPO**（CVPR 2024，B.3）、**D3PO**（CVPR 2024，pp. 8941–8951，arXiv 2311.13231：不做 ELBO 近似，而是把 DPO 装进 DDPO 的多步 MDP，假设每步 Q 函数可用参考/当前模型比值表示，逐步更新，显存友好；实验用"目标相对大小"作偏好代理，还能做降畸变、生成更安全图像）。
- 机制：Bradley-Terry 偏好模型 + DPO 重参数化，把 reward 学习吸收进策略本身；diffusion 侧用 ELBO（Diffusion-DPO）或逐步 MDP（D3PO）解决 likelihood 不可算。
- 优点：**完全离线**、无需 reward model、无需在线采样；隐式 KL-to-ref 约束内建（Table 1 里 Diffusion-DPO 自评为唯一同时满足开放词表 + 等推理成本 + 分布控制的方法）；工程最简单。
- 缺点：只利用相对偏好、丢弃 reward 幅值信息；离线导致无法探索；近似链条（Jensen、前向代替反向）引入偏差；效果受偏好数据覆盖限制。
- 适用：只有成对比较标签（人评、A/B rollout 成败）、无法或不愿在线采样时。

## C.5 路线四：SOC-adjoint 类（可微 reward，有分布收敛保证）——引用库内报告

该线库内已有完整精读，此处不重写，仅给谱系定位：

- **Adjoint Matching**（Domingo-Enrich, Drozdzal, Karrer, Chen；arXiv 2409.08861，**ICLR 2025**，proceedings.iclr.cc 收录，web 复核 2026-08-14）：把 reward 微调严格写成随机最优控制，证明必须使用 **memoryless noise schedule**（σ(t)=√(2ηt) 型）才能消除噪声变量与生成样本间依赖导致的偏差、真正收敛到 tilted 分布 p*；并把 SOC 解法转化为 lean adjoint 的回归（Adjoint Matching 目标），在一致性、真实感、对未见偏好模型的泛化和多样性保持上都超过反传与 RL 基线。这是对路线二"截断反传有偏"的理论修复。
- **Adjoint Sampling**（ICML 2025）：见 [`../reports/2504.11713_adjoint_sampling.md`](../reports/2504.11713_adjoint_sampling.md) —— 无数据只有能量时的可扩展 SOC 采样器，Reciprocal AM + replay buffer 解耦"昂贵能量评估次数"与"梯度更新次数"。
- **Discrete Adjoint Matching**（ICLR 2026）：见 [`../reports/2602.07132_discrete_adjoint_matching.md`](../reports/2602.07132_discrete_adjoint_matching.md) —— 把 AM 推广到 CTMC/token 空间做 KL 正则 reward 微调，适合离散动作/技能 token。
- 相关扩展（ASBS、FAS、Discrete ASBS）与整线对比见 [`../reports/sb_adjoint_extended_synthesis.md`](../reports/sb_adjoint_extended_synthesis.md)。

谱系定位：该线是四条路线中**唯一同时具备**（i）明确收敛到 KL 正则最优分布的理论保证、（ii）梯度级样本效率、（iii）与 SB/bridge 生成结构同构（本库主线）的路线；代价是要求 reward 可微 + 特定噪声调度，工程栈最重。

## C.6 四路线对照表

| 维度 | ① policy gradient | ② 直接反传 | ③ 偏好优化 | ④ SOC-adjoint |
| --- | --- | --- | --- | --- |
| 代表工作（venue，复核 2026-08-14） | DDPO（ICLR 2024）；DPOK（NeurIPS 2023）；Flow-GRPO（NeurIPS 2025） | ReFL（NeurIPS 2023）；DRaFT（ICLR 2024）；AlignProp（arXiv） | Diffusion-DPO（CVPR 2024）；D3PO（CVPR 2024） | Adjoint Matching（ICLR 2025）；AS（ICML 2025）；DAM（ICLR 2026，均库内有精读） |
| 需要的信号 | 黑盒标量 reward | **可微** reward | 成对偏好标签（可离线） | **可微** reward（终端梯度） |
| 在线/离线 | 在线 | 在线 | **离线** | 在线（AS 线有 replay buffer 复用） |
| 样本/查询效率 | 低（DDPO 需 1e4–5e4 查询；GRPO 系有改善） | **最高**（比 DDPO 快 >200×） | 取决于既有偏好数据量，训练本身廉价 | 高（回归目标 + 可复用采样） |
| 模式坍缩 / reward hacking 风险 | 中（无 KL 时高；DDPO 有 typographic attack 实例） | **最高**（多样性坍缩最快；截断反传有偏） | 低（隐式 KL-to-ref 内建） | 低（KL 正则内建于目标，有收敛保证） |
| 理论保证 | 无偏梯度但高方差；KL 版本近似 | 无（截断有偏，全链爆炸） | 依赖 ELBO/Jensen/前向近似 | **收敛到 p* ∝ p_base·exp(r/β)**（memoryless 条件下） |
| 与 SB-Render-Lite 的相关度 | 高：唯一直接吃 policy success 黑盒信号的在线路线 | 中：需先训可微 critic；适合快速迭代期 | **高**：rollout 成/败天然构成偏好对，零在线开销 | **最高**：与翻译器 bridge 结构同构，库内主线，正式训练首选 |

## C.7 SB-Render-Lite 选型：用 policy success 信号微调翻译器，走哪条路线

**信号形态分析**：policy success 是二值/低分辨率标量、不可微、单次评估 = 一次（仿真或真机）rollout、成本高且有噪声。这直接排除"裸上路线②/④"（都要可微 reward），也让路线①的查询预算成为硬约束。

**推荐三段式路径**：

1. **离线预对齐（路线③，首选起步）**：收集已有实验数据——同一 sim 帧经不同翻译器版本/随机种子得到的多个翻译结果，配上下游 rollout 成/败标签，构成 `(x^w, x^l | x_sim)` 偏好对，跑 Diffusion-DPO 式目标（对 bridge 模型：比较两条翻译在相同加噪点上的去噪误差差）。零在线开销、隐式 KL 保内容结构，先把翻译器推到"成功率友好"的区域。D3PO 的逐步变体在显存受限时是替代。
2. **小规模在线验证（路线①）**：在明确 rollout 预算（如 ≤5k 次）下用 DDPO_IS/DPOK 式 policy gradient 直接吃 success 信号，务必带 DPOK 式 KL 正则 + per-task reward 归一化 + 固定 anti-hacking 评估集（几何/动力学一致性不入 reward）。Flow-GRPO 的组内相对优势 + 减步训练技巧可直接借来降方差、省查询。
3. **critic 化后切换 SOC-adjoint（路线④，正式训练）**：用阶段 1/2 积累的 (翻译图像, success) 数据训练可微 success critic（本质是 learned value/energy），之后按库内 AM 线做 memoryless 微调，获得"收敛到 `p_base·exp(success/β)`"的分布保证，并用 AS 的 replay buffer 思想复用昂贵的 critic/rollout 评估。critic 的 reward hacking 用阶段 2 的在线信号定期校准。
4. 快速迭代期（超参/结构搜索）可临时用 DRaFT-LV 对 critic 直接反传，但不作为正式训练路线。

**一句话**：黑盒稀疏信号先走"偏好优化打底 + policy gradient 验证"，一旦 critic 可微化就收敛到库内 SOC-adjoint 主线——四条路线在本项目里是接力关系而非互斥选择。

---

# Part D venue 复核记录（检索日期：2026-08-14）

| 论文（arXiv） | 待核 venue | 复核结果 | 证据 |
| --- | --- | --- | --- |
| DDPO（2305.13301） | ICLR 2024? | **确认：ICLR 2024**（另见 ICML 2023 ES-FoMO workshop 早期版本） | proceedings.iclr.cc/paper_files/paper/2024（Paper-Conference PDF）；OpenReview |
| DRaFT（2309.17400） | ICLR 2024? | **确认：ICLR 2024 poster** | OpenReview forum 1vmSEVL19f（"ICLR 2024 poster"）；DeepMind 出版页 |
| DPOK（2305.16381） | NeurIPS 2023? | **确认：NeurIPS 2023** | proceedings.neurips.cc/paper_files/paper/2023 |
| Diffusion-DPO（2311.12908） | CVPR 2024? | **确认：CVPR 2024，pp. 8228–8238** | openaccess.thecvf.com CVPR2024 |
| ImageReward/ReFL（2304.05977） | — | NeurIPS 2023，pp. 15903–15935 | proceedings.neurips.cc；官方 repo |
| D3PO（2311.13231） | — | CVPR 2024，pp. 8941–8951 | openaccess.thecvf.com CVPR2024 |
| AlignProp（2310.03739） | — | **arXiv preprint**（检索日未见正式 venue 收录记录） | arXiv/DOI 页标 "Preprint"；项目页 bibtex 仍为 @misc |
| Adjoint Matching（2409.08861） | — | ICLR 2025 | proceedings.iclr.cc/paper_files/paper/2025 |
| Flow-GRPO（2505.05470） | — | NeurIPS 2025 poster | proceedings.neurips.cc 2025；neurips.cc virtual poster 页 |
| Fan & Lee shortcut fine-tuning（2301.13362） | — | ICML 2023（依据 DRaFT ICLR 正式版参考文献标注，未单独复核 OpenReview） | DRaFT 参考文献 |
| Lee et al.（2302.12192） | — | arXiv preprint（DDPO/DPOK 正式版均以 preprint 引用） | 两篇正式版参考文献 |
| 库内 AS（2504.11713）/ DAM（2602.07132） | — | ICML 2025 / ICLR 2026（沿用库内报告记载，本次未重复复核） | 库内报告 |

---

# 并入主库建议

1. **文件安置**：建议将本报告改名为 `reports/2305.13301_ddpo_reward_alignment_survey.md` 并入主库，或拆分为精读（`2305.13301_ddpo.md`）+ 综述（`diffusion_reward_alignment_synthesis.md`）两个文件；DPOK/DRaFT/Diffusion-DPO 三条收录条目信息量已接近库内标准报告的 60–70%，若后续需要可各自升格为独立精读（优先 Diffusion-DPO，因其与 SB-Render-Lite 数据形态最匹配）。
2. **INDEX 更新**：在 `reports/INDEX.md` 新增分区"Reward 微调四路线（policy gradient / 直接反传 / 偏好优化 / SOC-adjoint）"，与现有"Adjoint Sampler 方法线"分区交叉引用——现有分区是四路线中路线④的展开，本报告的 C.6 对照表可作为两个分区的入口。
3. **synthesis 衔接**：`reports/sb_adjoint_extended_synthesis.md` 的 AM 叙述可加一句反向指针（"policy gradient / 直接反传 / 偏好优化三条平行路线见本报告"），使读者从任一路线进入都能看到全景。本报告刻意未重写 AS/DAM 内容，合入时无重复。
4. **后续精读候选**（按对 SB-Render-Lite 的边际价值排序）：(a) Adjoint Matching 原论文（2409.08861）——库内该线唯一缺原始论文的独立精读；(b) Flow-GRPO（2505.05470）——policy gradient 路线在 flow 模型上的效率修复，与翻译器的 flow/bridge 参数化直接相关；(c) Diffusion-DPO 升格精读 + 其在 bridge/paired 条件下的适配推导。
5. **实验层面的直接产物**：C.7 的三段式路径可转成 SB-Render-Lite 的 roadmap 条目：阶段 1（离线 DPO 预对齐）不需要任何新采集，用现有 rollout 日志即可启动。

（报告完。检索/复核日期 2026-08-14；执行人 E07。）
