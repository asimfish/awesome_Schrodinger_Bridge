# E18 扩充报告：扩散/流策略下游接口 —— transport→policy 接口与 co-training 配比

- 研究员：E18（文献扩充）
- 日期 / 检索日期：2026-08-14
- 选题定位：本库（OT/SB for embodied sim2real）此前 25 篇报告集中在 transport 方法本身（SB/OT 的构造、对齐目标、采样器），但 SB-Render-Lite 翻译出来的数据最终要喂给下游 policy（多为 diffusion / flow matching 策略）。**翻译数据如何进入 policy 训练**——预训练+微调、co-training 配比、在线增广——以及**条件生成里 OT coupling 的正确做法**，直接决定翻译增益能否兑现。本报告补上这一环。
- 覆盖：精读 1 篇（COT Policy）+ 半精读 1 篇（Diffusion Policy）+ 收录条目 3 篇（π0 / OpenVLA / Octo）+ 综述补充证据 5 篇（均完成 venue 核验）。

## TL;DR

1. **COT Policy（CoRL 2025）给出一个对本库至关重要的负结果**：在条件生成任务里，naive 的 minibatch OT coupling（只在 noise 与 action 边缘分布之间配对、忽略条件）会产生有偏的条件流，性能甚至差于不做 OT 的 vanilla flow matching。机理是"跨条件错配"：OT 按几何距离把 noise 与其他条件下的目标配对，导致推理时（给定条件 c、noise 落在训练时几乎只与其他条件目标配对的区域）缺少回归目标、生成 OOD 样本。修正办法是把条件写进 OT ground cost（对连续高维观测先 PCA+K-means 离散化）。这与库内 Guided OT co-training（2509.18631）"必须对齐 joint (feature, action) 而非 observation marginal"是同一条教训在两个层面（策略训练内部 coupling / 跨域特征对齐）的体现，可提升为本库的普适设计原则。
2. **Diffusion Policy 类策略是"分布保真"的**：它显式拟合条件分布 p(A|O)，训练数据里有什么模式就学什么模式（包括翻译伪影）；观测只作 conditioning 不被建模，观测域偏移直接成为 OOD 条件输入；视觉编码器必须适配训练分布（frozen pretrained 编码器明显掉点）。因此翻译数据的视觉统计会被完整刻进策略——接口质量没有"容错缓冲"。
3. **co-training 配比已有可用的定量口径**：真实数据的 per-batch 采样权重应显著高于其自然占比（oversample real），但绝不能为零；仿真/翻译数据总量要大（比 real 多 1-2 个数量级）。三个独立证据（RSS 2025 / IROS 2025 / 2026 机理分析）给出一致区间，且 Wei et al. 发现"完美渲染反而有害"（策略需要能辨识域以适配残余动力学差异），对 SB-Render-Lite 的目标设定是关键修正：**视觉翻译不必也不应追求与真实完全不可分，应保留（或显式提供）域可辨识性，同时用 action-aware 对齐保住任务相关结构**。

---

## 一、精读：COT Policy —— Fast Flow-based Visuomotor Policies via Conditional Optimal Transport Couplings

### 1.1 元信息

- 论文：Fast Flow-based Visuomotor Policies via Conditional Optimal Transport Couplings
- 作者：Andreas Sochopoulos, Nikolay Malkin, Nikolaos Tsagkas, João Moura, Michael Gienger, Sethu Vijayakumar（University of Edinburgh + Honda Research Institute Europe）
- arXiv：2505.01179；项目页：ansocho.github.io/cot-policy
- venue：**CoRL 2025（PMLR v305）**——任务方已核验，本报告沿用；arXiv HTML 全文格式（Keywords: Flow Matching, Optimal Transport, Imitation Learning）与 CoRL 模板一致（检索日期 2026-08-14）。
- 全文获取：arXiv abs 页直接提供完整 HTML 正文 + 附录（含实现细节与超参表），精读基于全文。

### 1.2 动机

Diffusion Policy（DP）/ flow matching（FM）策略能拟合多模态动作分布，但推理要数值积分 ODE/SDE，多步 NFE（number of function evaluations）导致真机控制频率低、动作间歇（robot 走完短程轨迹后停下来等下一次推理）。已有加速路线主要是蒸馏（Consistency Policy、One-Step Diffusion Policy 等），但蒸馏需要额外训练阶段或教师模型，训练成本高。作者的路线：用 OT coupling 把 flow 的积分路径"拉直"，让 1-2 步 Euler/midpoint 积分就能出高质量动作——训练成本与 vanilla FM/DP 相同，无蒸馏。

### 1.3 方法核心

#### (a) conditional coupling 的失效机理（本报告最关注的部分）

背景：FM 训练需要选一个 coupling q(x₀, x₁)。独立耦合（I-CFM）无偏但路径弯曲（少步积分误差大）；minibatch OT coupling（OT-CFM，Tong et al. TMLR 2024）能拉直路径、降低目标方差。**但这套逻辑只对无条件生成成立。**

失效机理（论文 Fig. 1 双月牙实验 + Sec. 2.2）：设目标是条件分布 p₁(·|c)。naive 做法是在 noise 边缘 p₀ 与数据边缘 p₁（混合了所有条件）之间算 minibatch OT。OT 只看样本几何距离，于是"上方高斯"的 noise 几乎只与"上月牙"（c=0）的目标配对。训练出的向量场虽然以 c 为输入，但**回归目标的支撑集有系统缺口**：几乎不存在"从上方高斯连到下月牙"的训练对。推理时给定 c=1、noise 恰好采样在上方高斯区域，模型没有学过对应的传输方向，产出 OOD 样本。formally：coupling 决定了条件向量场的监督信号在 (noise, condition) 联合空间上的覆盖；marginal OT 使覆盖集中于"OT 几何下最近的条件"，其余条件组合欠监督。

关键定量证据（Table 1，MimicGen 4 任务 + push-t + cup 平均成功率）：OT-CFM 在 NFE=4 时 0.740、NFE=2 时 0.709，**低于 vanilla CFM 的 0.797 / 0.790**——"拉直路径"换来的收益被条件偏差完全吃掉还倒亏。低维双月牙/fork 分布上更极端：OT-CFM 即使 100 步积分也无法恢复正确条件分布（W₂ 距离劣于 CFM），而 CFM 一步积分近似输出条件均值（无偏但塌缩）。

**对本库的直接含义**：任何"条件传输"——包括 SB-Render-Lite 的 sim→real 翻译（若以任务状态/布局为条件）、以及把翻译器与策略耦合训练时的 batch 配对——只要用 batch 级 OT/SB 配对而忽略条件变量，就会引入同构的偏差。库内 Guided OT co-training（2509.18631）在跨域特征对齐层面得到同样结论（只对齐 observation marginal 不够，必须对齐 joint (feature, action)）；COT Policy 则在策略训练内部的 noise-action coupling 层面独立验证了它。两者合起来可以总结为一条普适原则：**minibatch OT 的 ground cost 必须包含所有决定"哪些样本在语义上可交换"的变量（条件、动作、任务状态），否则 OT 的几何贪心会制造系统性错配。**

#### (b) 修正方案：COT coupling

把条件并入 OT 代价。给定数据 batch B₁ = {(x₁⁽ⁱ⁾, c⁽ⁱ⁾)}，构造 noise batch B₀ = {(x₀⁽ⁱ⁾, c₀⁽ⁱ⁾)}，其中 c₀ 取 **数据条件的均匀随机置换**（保证两侧条件同分布），然后在扩展代价

c((x₀,c₀),(x₁,c₁)) = ‖x₀−x₁‖² + ‖γ(c₀−c₁)‖²

下解 minibatch OT（精确 EMD，POT/torchcfm 实现）。γ→∞ 时恢复"逐条件分别 OT"（条件动态 OT 的精确解）；γ=0 退化为 OT-CFM。γ 不必手调：按公式 γ = 10 × (batch 平均样本距离 / batch 平均条件距离) 自适应设定，使条件项平均比样本项大一个数量级；γ 超过 ~10⁴ 会引发 EMD 数值错误（OT 矩阵退化成类独立耦合），因此"够大但有界"。

#### (c) 连续高维条件的处理：PCA + K-means 离散化

机器人观测（RGB + proprioception）连续且高维，逐条件 OT 无法直接构造（每个条件只有一个样本）。作者利用"动作分布对观测的小扰动不敏感"这一先验，把相似观测量化为同一条件：e = E(o)（E 为 PCA，图像 100 维主成分，proprio 不变），c = Q(e)（Q 为 K-means，K 个簇心）。**量化条件只用于算 OT coupling；流模型 v_θ(t, x | o) 的条件输入仍是原始观测**（经 end-to-end 训练的 ResNet-18 编码）。

K 的定性规律（Sec. E.2 + Fig. 4 消融）：K→1 退化为 OT-CFM（偏差回来）；K→数据集大小退化为 I-CFM（每条件唯一，置换配对=独立耦合）；**K 取接近 batch size（任务里 K=64，batch=64）最优**。K 过小的退化与 OT-CFM 同型。附录 D.3：有聚类比无聚类（只靠调 γ）平均高 ~6 个点（push-t + coffee_d1：0.833 vs 0.774），且免去逐数据集调 γ。

#### (d) 流策略训练细节

- 骨干：与 DP 相同的 CNN U-Net（~240M 参数，不含视觉编码器）；视觉编码器 ResNet-18 从零 end-to-end（沿用 DP 结论：frozen 预训练表征次优）。
- 线性插值路径 x(t) = t·x₁ + (1−t)·x₀，CFM 回归目标 x₁−x₀；训练超参与 DP/CFM 完全一致（AdamW、cosine、EMA 0.9999、1000 epochs、obs horizon 2 / action pred horizon 16 / exec horizon 8）。
- 采样：midpoint solver 2 步（NFE=2）；PCA/K-means 用 GPU 实现，逐 batch 开销相对 U-Net 前反向可忽略；收敛速度与 CFM/DP 相同（无蒸馏、无第二阶段）。

### 1.4 实验

- 仿真（MimicGen threading_d0/stack_d1/coffee_d1/square_d0 + push-t + dm-control cup；每任务 100 demos，150 rollouts×2 seeds）：**NFE=2 的 COT Policy 平均 0.818，超过 NFE=20 的 DP（DDIM）0.781、NFE=4 的 CFM 0.797、NFE=4 的 OT-CFM 0.740、Adaflow 0.783**。即 ~10× 推理加速 + ~4% 成功率增益。
- 多模态性：提出 Trajectory Variance（TV）指标——以 DTW Barycenter Average 为"均值轨迹"、平均 DTW² 距离为轨迹方差。NFE=1-2 时 COT Policy 的 TV 与成功率同时高于 CFM（CFM 少步时多样性塌缩），说明拉直路径没有牺牲模式覆盖。
- 真机（KUKA IIWA 14 + 2×RealSense D415，push-T / cup-stacking / cup-in-drawer，各 30 条 spacemouse 遥操演示，RTX 2080 推理）：1 步 Euler / 2 步 midpoint 下 COT 全面优于 CFM（如 push-T 成功率 0.8 vs 0.2，完成时间 35.7s vs 54.4s）；CFM 低 NFE 时常卡死或动作抖动。1 步推理即可再现数据集的双模式（顺/逆时针推 T）。

### 1.5 局限

- 引入两个超参：γ（自适应公式基本免调）与 **K（敏感**，推荐 K≈batch size，但跨数据集普适性未证明）。
- 高维动作空间下 low-NFE 与 high-NFE 之间仍有残余差距；作者建议与单阶段蒸馏（consistency FM / shortcut）正交组合。
- 评估集中在中小规模单任务 BC；未验证大规模多任务 / VLA 场景下 COT coupling 的可扩展性（条件聚类在极多样条件下如何设 K 是开放问题）。
- MimicGen 演示是程序化生成的，多模态性弱于人类演示（作者自述避开了 DP 原任务因为已饱和）；真机每任务仅 5 rollouts，统计功效有限。

### 1.6 对 SB-Render-Lite 的接口启示

1. **SB/OT 翻译器的 pairing 必须 condition/action-aware**：SB-Render-Lite 若用 minibatch coupling 训练（I2SB/SB Flow 风格），ground cost 应加入任务状态/proprio/action 一致性项，权重可仿照 γ 自适应公式（条件项 ≈ 10× 像素/特征项）。这与 2509.18631 的 joint (feature, action) cost、EgoBridge 的对齐选择互为印证。
2. **PCA+K-means 条件量化是零成本可移植组件**：可直接作为 SB-Render-Lite 的 pairing heuristic（与 2509.18631 的 DTW temporally aligned sampling 互补：一个按视觉/状态聚类，一个按轨迹相似度配对）。
3. **下游策略若用 flow matching，可叠加 COT coupling 拿到 1-2 NFE 实时推理**——真机评估回路更快，对我们"所有视觉指标服从 downstream real-domain policy success"的评估哲学是工程利好。

---

## 二、半精读：Diffusion Policy —— Visuomotor Policy Learning via Action Diffusion

### 2.1 元信息（venue 已 web 复核，检索日期 2026-08-14）

- 论文：Diffusion Policy: Visuomotor Policy Learning via Action Diffusion
- 作者：Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, Shuran Song（Columbia / TRI / MIT / Stanford）
- arXiv：2303.04137（v5 为期刊扩展版，本次精读版本）
- venue：**RSS 2023**（roboticsproceedings.org/rss19/p026，DOI 10.15607/RSS.2023.XIX.026）；**扩展版 IJRR 2024**（DOI 10.1177/02783649241273668，2024-10-11 上线）。两处均已核验。
- 定位：半精读，只提炼与"数据分布敏感性 / 翻译数据接口"相关的结论；方法本身（DDPM 条件化、receding horizon、CNN vs Transformer 骨干）不展开。

### 2.2 与数据分布敏感性相关的核心结论

1. **DP 是"分布保真"的策略类**。它学习动作分布的 score 并用随机 Langevin 采样，能表达任意可归一化分布：短程多模态（push-T 左右绕行两模式都保留并在单次 rollout 内 commit 一个）与长程多模态（Block Push / Franka Kitchen 子目标顺序任意，p2/p4 指标分别 +32%/+213%）都如实复现；对照的 LSTM-GMM/IBC 偏向单模式、BET 无法 commit。**接口含义**：喂给 DP 的混合数据（真实 + 翻译）的每一个模式、偏差、伪影都会被当作条件分布的一部分学下来——表达力越强，对数据组成越"诚实"，翻译质量问题不会被模型平均掉，而是变成可被采样出来的行为。
2. **观测是 conditioning，不是被建模的对象**。DP 拟合 p(A|O) 而非 p(A,O)：观测编码一次、不参与去噪。这带来实时推理优势（DDIM 10 步 0.1s@3080），但也意味着 **p(O) 没有任何生成式正则**——测试时观测域偏移（sim→real 的残余视觉差距）就是纯粹的 OOD 条件输入，策略行为未定义。这是"为什么必须做视觉翻译/对齐"在策略侧的形式化理由。
3. **视觉编码器必须适配训练分布**（IJRR 版新增消融，robomimic square/PH）：ResNet-18 从零训练是默认；**frozen 预训练编码器（R3M 等）表现差**；预训练 + 10× 小学习率微调最好（CLIP ViT-B/16 微调 50 epochs 达 98%）。**接口含义**：编码器统计会向训练观测分布收敛——翻译数据的颜色/纹理/伪影统计将直接刻进 encoder。若翻译数据与真实数据混训，encoder 是两域"抢表征"的主战场（这正是 2509.18631 用 OT 对齐 latent 的动机）。
4. **对演示数据瑕疵的鲁棒性有选择性**：action-sequence 预测 + receding horizon 使 DP 对 idle actions（遥操作暂停产生的近零动作段）鲁棒——单步策略（BC-RNN/IBC）会过拟合停顿卡死，DP 不需要像惯例那样把 idle 段滤掉（真机 push-T 保留 idle 段训练仍 95% 成功）。对 latency 鲁棒至 4 步。但 position control 显著优于 velocity control（复合误差 + 多模态表达），vision-based DP 对更长的 observation horizon 反而退化（obs horizon=2 最优）。**接口含义**：翻译数据的时间维伪影（帧间闪烁、不一致）会破坏 action-sequence 的时间一致性前提，是视频级翻译（3MSBM 方向）需要优先解决的点。
5. **数据效率与 BC 上限**：每个训练集规模下都优于 LSTM-GMM（附录 A.5）；但作者自述继承 BC 的根本局限——演示数据不足或次优时性能受限。归一化细节也值得记录：action 逐维 min-max 到 [−1,1]（DDPM 每步 clip 到 [−1,1]），**normalizer 统计以哪个域/哪份混合数据为准，是 co-training 实现里容易被忽略的接口细节**。

### 2.3 对本库的含义

DP（及其 flow matching 变体）是 SB-Render-Lite 下游最可能的策略形态。上述性质合起来说明：**翻译数据进 policy 的收益/风险都被放大**——分布保真意味着好数据直接变成好行为、伪影直接变成坏行为；conditioning-only 意味着残余视觉 gap 无生成式缓冲；encoder 可塑意味着配比与对齐损失（而非模型本身）决定表征偏向哪个域。这为第四节三种接口的取舍提供了策略侧依据。

---

## 三、收录条目：π0 / OpenVLA / Octo（发表状态均 web 复核，检索日期 2026-08-14）

### 3.1 π0: A Vision-Language-Action Flow Model for General Robot Control

- 作者：Kevin Black, Noah Brown, Danny Driess, …, Sergey Levine 等（Physical Intelligence）
- arXiv：2410.24164；venue：**RSS 2025**（roboticsproceedings.org/rss21/p010，DOI 10.15607/RSS.2025.XXI.010）✅ 已核验
- 要点：PaliGemma 3B VLM 骨干 + 300M **flow matching action expert**（双专家共享 self-attention），50 Hz 动作 chunk；预训练 >10,000 小时自采数据（7 种机器人构型、68 任务）+ OXE。
- 接口相关：明确采用 **pre-training / post-training 两阶段配方**——预训练数据求覆盖与多样（含低质数据，提供错误恢复模式），post-training 用高质量精选数据求流畅执行；哲学是"只用高质量数据学不会纠错，只用杂数据学不会利落"。**混合权重规则：每个 (task, robot) 组合按 n^0.43 加权**（n 为样本数），压低过代表任务——这是目前 VLA 界少有的显式配比公式。局限自述：预训练数据应如何构成/加权仍是 open problem。
- 对本库：翻译数据若进 VLA 管线，最自然的位置是预训练混合（当作又一个"embodiment/domain 源"，按 n^0.43 类规则加权），而非 post-training——post-training 应留给高质量真实演示。

### 3.2 OpenVLA: An Open-Source Vision-Language-Action Model

- 作者：Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, …, Chelsea Finn 等（Stanford/UCB/TRI/Google DeepMind/MIT）
- arXiv：2406.09246；venue：**CoRL 2024**（PMLR v270:2679-2713，proceedings.mlr.press/v270/kim25c）✅ 已核验
- 要点：7B（Llama-2 骨干 + DINOv2/SigLIP 融合视觉编码器），OXE 970k 演示预训练，离散 action token（对照组：DP/Octo 的连续生成头）；29 任务上超过 55B 的 RT-2-X 绝对成功率 +16.5%。
- 接口相关：**微调是其核心卖点**——在新场景微调后超过 from-scratch Diffusion Policy +20.4%（多物体、强语言 grounding 场景优势最大）；LoRA 可在消费级 GPU 上微调、量化推理不掉点。这是"generalist 预训练 + 目标域微调"接口的最强公开证据之一。
- 对本库：微调阶段单一域数据直接覆盖表征——若微调数据 = 翻译数据 + 少量真实，OpenVLA 未提供配比指引（其微调实验用目标域真实数据 10-150 demos），需借第四节 co-training 证据补位。

### 3.3 Octo: An Open-Source Generalist Robot Policy

- 作者：Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees 等（UCB/Stanford/CMU/Google DeepMind）
- arXiv：2405.12213；venue：**RSS 2024**（roboticsproceedings.org/rss20/p090，DOI 10.15607/RSS.2024.XX.090）✅ 已核验
- 要点：transformer 骨干 + **diffusion action head**（DDPM 目标；仅骨干一次前向、去噪在小 head 内完成），OXE 25 个数据集共 800k 轨迹；token 化模块设计允许微调时增删观测/动作空间而不重初始化骨干。
- 接口相关：(i) **预训练混合是手工 curation + 启发式加权**：剔除无图像流/非 delta-EE/重复/低分辨率/过窄数据集后，把"更多样"的数据集权重加倍、压低超大数据集——配比靠人工判断而非公式（与 π0 的 n^0.43 对照）。(ii) **微调配方标准化**：~100 条目标域演示、50k 步、cosine decay，**全参微调优于冻结任何子模块**；单张 A5000 约 5 小时。(iii) 消融：diffusion head 显著优于 MSE head（hedging、动作迟缓）与离散 head（精度损失）——支持"下游生成式动作头"作为本库默认假设。
- 对本库：Octo 的"~100 demos 全参微调"配方 + DP 的编码器微调结论，共同构成接口 1（预训练+微调）的操作基线。

---

## 四、综述段：翻译数据进 policy 训练的三种接口与 co-training 配比

> 本节为可并入主库 synthesis 的综述段。承接库内 `2509.18631_guided_ot_sim_real_policy_cotraining.md`"与当前方向的关系"一节：那里论证了 transport 应与策略学习耦合（joint feature-action cost、UOT 处理不平衡）；本节回答其后一个问题——**耦合之后，翻译/仿真数据以什么形式、什么比例进入 policy 训练**。

### 4.0 补充证据条目（本节新引入，均已核验 venue，检索日期 2026-08-14）

| 论文 | arXiv | venue | 一句话贡献 |
| --- | --- | --- | --- |
| Maddukuri et al., Sim-and-Real Co-Training: A Simple Recipe for Vision-Based Robotic Manipulation | 2503.24361 | **RSS 2025**（DOI 10.15607/RSS.2025.XXI.109）✅ | 系统化 co-training 配方：digital cousin 仿真数据平均 +38% 真实成功率；α（sim 采样率）=99% 最优 |
| Wei et al., Empirical Analysis of Sim-and-Real Cotraining of Diffusion Policies for Planar Pushing from Pixels | 2503.22634 | **IROS 2025**（DOI 10.1109/IROS60139.2025.11246304）✅ | 平面推动单任务穷举分析：配比敏感性、物理 gap > 视觉保真、"完美渲染有害" |
| Lei, Liu, Maddukuri, Jiang, Zhu, A Mechanistic Analysis of Sim-and-Real Co-Training in Generative Robot Policies | 2604.13645 | arXiv preprint（ICML 格式，2026；检索日未见正式收录）⚠️ | co-training 机理：结构化表征对齐（对齐+可辨识）为主效应；给出 balanced 配比区间与 CFG-ADDA 组合方法 |
| Rao et al., RL-CycleGAN | — | **CVPR 2020**（openaccess.thecvf.com，pp. 11157-11166）✅ | 翻译器与 RL 联训：RL-scene consistency（Q 值不变性）约束翻译 |
| Ho et al., RetinaGAN | 2011.03148 | **ICRA 2021**（DOI 10.1109/ICRA48506.2021.9561157）✅ | 检测器一致性约束的 task-decoupled 翻译，RL/IL 通用，比 RL-CycleGAN 易训 |

### 4.1 接口 1：预训练 + 微调（translate → pretrain → real finetune）

**做法**：翻译/仿真数据进大规模预训练混合，真实数据留作微调。VLA 界的标准范式。

**证据**：π0（RSS 2025）的 pre/post-training 两阶段配方及 n^0.43 加权；OpenVLA（CoRL 2024）微调后超 from-scratch DP +20.4%；Octo（RSS 2024）~100 demos、50k 步全参微调配方，diffusion head 优于 MSE/离散 head；DP（IJRR 2024）编码器消融——预训练表征 frozen 使用必掉点、小 lr 微调最优——从策略侧解释了为什么"微调而非冻结"是共识。

**适用与风险**：适用于翻译数据量大、任务多、允许两阶段训练的场景。优点是真实数据在最后阶段"说了算"，翻译伪影可被微调部分洗掉；风险是 (i) 微调遗忘预训练获得的覆盖（π0 用 post-training 混入部分预训练数据缓解思路，见其配方哲学）；(ii) 若翻译数据与真实分布系统性偏移，预训练表征的先验可能拖慢而非加速微调（OpenVLA/Octo 均未在"预训练数据 = 翻译数据"设定下验证——这是 SB-Render-Lite 可以贡献的实验空白）。

### 4.2 接口 2：co-training 配比（单阶段混合训练）

**做法**：真实与翻译/仿真数据在同一阶段混合训练，用 per-batch 采样概率（等价于 loss 加权）控制配比。这是 SB-Render-Lite 最相关的接口，证据链最完整：

- **Maddukuri et al.（RSS 2025）**：loss = α·L_sim + (1−α)·L_real，α 为 batch 内采样仿真数据的概率。核心数字：**1:1（α=50%）次优；α=99% 最优（CupPnP 达 95%）；再推到 99.5%/99.9% 从 95% 崩到 60%**。仿真演示总量必须足够（10k→500 使 Panda 任务 67%→53%；1k→100 使 GR-1 任务 95%→75%）；即使真实演示加到 400 条，co-training 仍优于 real-only。数据点口径：**每任务 real 20-50 条、sim（MimicGen digital cousin）1k-10k 条，即 real 自然占比 <1-5%，但 real 采样权重保持在 1% 而非趋零**。相机视角对齐是 DC 数据生效的前提（重渲染仿真以匹配真实相机位姿）。
- **Wei et al.（IROS 2025）**（注意其 α 约定相反：α = 采样 real 的概率）：真实数据少时 co-training 提升 2-7×；收益随 sim 规模幂律改善但会平台化，抬升天花板要靠加 real。**性能对 α 敏感（|D_R|=10 时尤甚）；α→0 时性能近乎不连续地崩掉——batch 里必须有非零比例的 real**；最优 α 随 real 数据量增大而上移（real 多时 overfit real 的代价小）。**物理 gap 的影响大于视觉保真**（无物理偏移比 Level-1 偏移高 15.5%）；**反直觉发现：一定视觉差距反而有益——binary probe 显示高性能策略必须能区分 sim 与 real（因为两域物理不同、所需动作不同）；完美渲染的 sim 数据使两域不可辨识，性能下降；给 one-hot 域标签则提升性能**。
- **库内 Guided OT co-training（2509.18631, NeurIPS 2025）**：配比数据点为每任务 **real 10-25 条 + sim 200-1000 条**（约 1:40 至 1:100），用 UOT 松弛边缘约束处理不平衡、DTW temporally aligned sampling 降低错配；sim 规模消融（100/300/500/1000 条 + 25 real）显示增大 sim 覆盖持续改善泛化。它未做 α 扫描——配比证据须由上两篇补位，这正是本节与主库的衔接点。
- **Lei et al.（2026, preprint）机理层**：理论 + 实验识别两个内在效应——**结构化表征对齐**（跨域对齐 + 域可辨识，主效应，与成功率中强相关）与 importance reweighting（次效应）。给出配比原则：real 权重 w 应落在 (自然占比 N/(N+M), √(N/M)) 区间（其实验中约 (0.016, 0.13)）——**即 oversample real 但不超过规模比的平方根**。对三类 co-training 技术的统一解读：OT 特征正则（即 2509.18631 一系）与 adversarial 对齐偏"对齐"侧，balanced 配比下有效、极端配比下反而退化；CFG/域标签偏"可辨识"侧；两者组合（CFG-ADDA：one-hot 域标签 + 对其余维度做对抗对齐，推理时 guidance scale 取 λ=−0.5 主动从 surrogate 域抽取知识）最稳，真实任务 ~74% 成功率、比单一技术再 +~20%。

**综合推荐（统一为 per-batch real 采样概率 w_real）**：
1. w_real 显著高于自然占比（oversample real），经验锚点：Maddukuri 的 1%（real:sim ≈ 1:50 自然比时）、Lei 的上界 √(N/M)；**任何情况下 w_real > 0 且每个 batch 保证 real 出现**（Wei 的不连续崩塌）。
2. 翻译/仿真数据总量优先做大（比 real 多 1-2 个数量级），收益平台化前不必担心"稀释"——配比用采样权重控制，不靠删数据。
3. 配比之外叠加对齐正则时注意机理层教训：**OT/UOT 对齐损失在 balanced 配比区间内有效，在极端配比下会把表征拽向主导域、产生负迁移**——2509.18631 的 UOT 松弛正是对这一风险的部分防御，但更稳的做法是"对齐 + 可辨识"双管齐下（域标签/CFG 与 OT 对齐并用）。

### 4.3 接口 3：在线增广（on-the-fly translation as augmentation）

**做法**：翻译器不预先固化数据集，而是在策略训练循环内对 sim 样本实时翻译（每次采样可得不同翻译结果），等价于把翻译器当作一个可学习的 augmentation 分布。

**证据**：RL-CycleGAN（CVPR 2020）把 CycleGAN 与 RL 联训，用 RL-scene consistency（翻译前后 Q 值不变）保证任务相关信息不被翻译抹掉——是"transport 与 policy 耦合"的最早实证之一；RetinaGAN（ICRA 2021）改用预训练检测器的一致性损失，task-decoupled、可跨任务复用、更易训练，支撑 RL 与 IL 两类下游。二者共同确立了在线/近线翻译的两个设计轴：**一致性约束的载体**（任务 Q 值 vs 通用感知任务）与**翻译器-策略耦合强度**（联训 vs 预训练后固定）。库内 SB 侧对应物：I2SB（paired 强基线）、SB Flow（unpaired 基线）提供比 GAN 更稳的翻译骨干；COT Policy 的教训在此同样适用——翻译若以 sim 内容为条件，训练翻译器时的 batch coupling 也应 condition-aware，否则会在内容-风格错配上翻车。

**适用与风险**：优点是 (i) 随机翻译提供数据多样性、避免"一次翻译错误被固化并被分布保真的 DP 反复学习"（见 §2.2 第 1 点）；(ii) 翻译器可随策略需求继续改进。代价是训练时延（每 batch 过一次翻译器）与质量耦合（翻译器坏一段时间、策略就学一段坏数据）。工程折中是"近线"方案：定期用最新翻译器重刷数据集，介于接口 2 与 3 之间。

### 4.4 对 SB-Render-Lite 的整体建议（三接口取舍）

1. **默认走接口 2（co-training）**：SB-Render-Lite 的翻译数据天然是"带完整 action 标注的 sim 轨迹 + 更真实的观测"，与 Guided OT / Maddukuri 的设定同构。起步配比：real 采样权重取 max(自然占比×10, 1%) 并按 Lei 区间 (N/(N+M), √(N/M)) 网格搜索 2-3 个点；翻译数据总量按 real 的 20-100× 生成。
2. **不要把"翻译到与真实不可分"当作目标函数**（Wei 的核心修正）：SB 翻译应对齐 action-relevant 结构（物体位姿、接触几何、光照大形），保留或显式标注域身份（one-hot domain tag 进策略条件几乎零成本，且 Lei 表明推理时还可用负 guidance 主动利用它）。视觉指标（FID 类）与 downstream success 的错位在此有了机理解释：残余动力学 gap 存在时，视觉不可分反而有害。
3. **coupling 纪律贯穿两层**：翻译器训练的 minibatch 配对（condition-aware，COT Policy）与策略特征对齐（joint feature-action UOT，2509.18631）用同一套 cost 设计语言，可在论文里作为统一贡献叙述。
4. **实验空白即机会**：目前没有工作在"预训练数据 = SB 翻译数据"的设定下检验接口 1，也没有 SB 翻译器 + 在线增广（接口 3）与 GAN 系（RL-CycleGAN/RetinaGAN）的对照。SB-Render-Lite 若在同一任务集上跑通接口 2 为主、接口 1/3 为消融的对比，就是一个自然的实验矩阵。

---

## 五、并入主库建议

1. **新增报告文件**（由主库维护者操作，本报告不改动现有文件）：
   - `reports/2505.01179_cot_policy.md`：可直接取本报告第一节内容；建议在 INDEX 中归入新分组"**扩散/流策略下游接口**"。
   - `reports/2303.04137_diffusion_policy.md`（半精读版）：取第二节；同组。
   - π0 / OpenVLA / Octo 建议以轻量条目（元信息 + 接口相关段）进同组或 `metadata/papers.tsv`，不必成篇。
2. **papers.tsv 增补行**（元信息均已核验，检索日期 2026-08-14）：2505.01179（CoRL 2025, PMLR v305）、2303.04137（RSS 2023 + IJRR 2024）、2410.24164（RSS 2025）、2406.09246（CoRL 2024, PMLR v270）、2405.12213（RSS 2024）、2503.24361（RSS 2025）、2503.22634（IROS 2025）、2604.13645（preprint ⚠️ 待正式收录后更新）、RetinaGAN 2011.03148（ICRA 2021）；RL-CycleGAN 无 arXiv 主号可用 CVPR 2020 openaccess 链接。
3. **synthesis.md 衔接点**：在"对当前 SB-Render-Lite 的直接启发"一节之后追加本报告第四节综述段；并把"ground cost 必须包含条件/动作变量"提升为贯穿翻译器训练与策略对齐两层的库级设计原则（COT Policy × Guided OT 双证据）。
4. **2509.18631 报告的交叉引用建议**：其"与当前方向的关系"一节可加一句指向本报告——配比的 α 扫描证据由 Maddukuri（RSS 2025）与 Wei（IROS 2025）补齐，机理解释见 Lei et al. 2604.13645。
5. **后续扩充候选**（本次未精读）：2604.13645 正式发表后值得升级为精读（其对 OT 对齐类方法在极端配比下负迁移的判断直接关系 SB-Render-Lite 的损失设计）；Kerrigan et al. NeurIPS 2024（dynamic conditional OT，COT Policy 的理论源头）可作为 SB 侧条件传输的理论补充。
