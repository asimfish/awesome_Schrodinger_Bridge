# E12｜世界模型数据引擎：DreamGen 精读 + UniSim 半精读 + 评估逻辑笔记

## 选题定位

- 扩充方向：**世界模型 / 视频生成作为机器人数据引擎**（来自缺口分析 R09：主库 25 篇集中在 OT/SB 迁移方法线，缺少"视频世界模型生成合成数据"这条正在快速成型的平行路线）。
- 与主库关系：`SB-Render-Lite` 做 sim RGB → real RGB 的 transport/翻译，本质上也是在造"喂给 policy 的合成视觉数据"。世界模型路线（DreamGen、UniSim、Cosmos）解决的是同一个终局问题——**如何低成本判断"生成/翻译出来的数据"对下游 policy 有没有用**。DreamGen Bench 是目前唯一显式验证了"代理指标 ↔ 下游 policy success 正相关"的工作，其方法论正是 SB-Render-Lite 评估协议需要的逻辑。
- 本文产出：DreamGen 精读（1 篇）、UniSim 半精读（1 篇）、收录条目 5 条（Cosmos 平台、Cosmos-Transfer1、Video Prediction Policy、UniPi、VideoPhy）、"transport 质量 ↔ policy success"评估逻辑笔记 1 份。
- 检索与核验日期：2026-08-14。精读全文来源：arXiv abs 页返回的 HTML 全文（arXiv 官方 HTML 渲染；ar5iv 作为备用入口，DreamGen 的 ar5iv 请求被重定向到同一 HTML 内容）。两篇精读均成功获取全文，无"仅摘要"情况。

## TL;DR

1. **DreamGen（CoRL 2025）**证明了一条极简路线：视频世界模型 LoRA 微调 → 初始帧+指令生成视频 → IDM/latent action 打伪动作 → 训 policy。仅用单环境单任务的遥操数据，就让 GR1 人形学会 22 个全新行为动词（11.2%→43.2%）并泛化到 10 个从未采过数据的环境（0%→28.5%）。
2. **DreamGen Bench 的核心方法论**：用两个 VLM 代理指标（Instruction Following + Physics Alignment，人评校准 Pearson>0.9）给视频世界模型打分，再用固定配方（每模型 7k 条合成轨迹训同一 policy）验证 bench 分数与 RoboCasa policy success **跨模型正相关**——这把"生成质量评估"从美学指标转向了"下游收益预测"，是可以直接搬到 SB-Render-Lite 的评估逻辑。
3. **UniSim（ICLR 2024 Outstanding Paper）**是另一极：不做离线数据集，而是把世界模型当**交互式环境**（观测预测模型自回归 rollout），在里面做 hindsight relabeling、RL 训练和视频字幕数据生成，policy 零样本迁移真机。它用 FVD/CLIP 选模型、用任务指标验证收益，但**没有**建立两者之间的相关性方法论——这个空缺正是 DreamGen Bench 补上的。
4. 对 SB-Render-Lite 最重要的迁移结论：翻译图像的代理指标应按"离 policy 的距离"分层——**动作可恢复性 >（前景）结构/几何保持 > 指令语义一致性 > 全局分布指标（FID/FVD）**；且我们比 DreamGen 条件更好：翻译数据自带 sim 动作真值，可以直接算动作一致性而不必依赖 replay。

---

## 一、精读：DreamGen — Unlocking Generalization in Robot Learning through Video World Models

### 基本信息

- 论文：DreamGen: Unlocking Generalization in Robot Learning through Video World Models（早期版本副标题为 "through Neural Trajectories"，arXiv 当前版本已改为 "Video World Models"）
- 作者：Joel Jang, Seonghyeon Ye, Zongyu Lin, Jiannan Xiang 等（共同一作）；Scott Reed, Yuke Zhu, Linxi "Jim" Fan 共同指导。NVIDIA GEAR Lab 联合 UW/KAIST/UCLA/UCSD 等
- 时间：arXiv 2025-05-19（2505.12705）
- venue：**CoRL 2025（PMLR v305）**，已由主库 venue 核验（R05）确认，本次未重复核验
- 链接：https://arxiv.org/abs/2505.12705 ｜ 项目页：https://research.nvidia.com/labs/gear/dreamgen
- 全文获取：arXiv HTML 全文，2026-08-14
- 归类：视频世界模型作为合成数据引擎；IDM 伪动作标注；生成质量基准与下游相关性

### 一句话总结

DreamGen 把 SOTA 图像→视频生成模型（WAN2.1 等）当作合成数据生成器而非实时规划器：LoRA 微调适配目标机器人 → 初始帧+语言指令 rollout 出"神经轨迹"（neural trajectories）→ 用 IDM 或 latent action model 补伪动作 → 训练视觉运动 policy；并配套 DreamGen Bench 证明视频模型的指令跟随/物理合理性分数能预测下游 policy 成功率。

### 解决的问题

机器人基础模型依赖人工遥操数据，逐任务逐环境采集成本极高；传统仿真合成数据受 sim2real gap、难仿真对象（液体、可变形物、铰接物）和 TAMP/插值式动作模板的限制。已有的生成式增广（inpainting、图像扩散、video2video）只提升视觉鲁棒性，**不产生新行为**。DreamGen 的问题定义是：能否直接利用互联网预训练视频模型的物理先验、自然运动先验和语言 grounding，规模化产出带动作标注的、覆盖新行为和新环境的训练数据。

### 方法核心：四阶段管线

**Stage 1 视频世界模型微调**。在目标机器人的遥操轨迹上 LoRA 微调（默认 rank 4、alpha 4；避免遗忘互联网视频先验）。默认底模 WAN2.1（14B 级 I2V），bench 中另测 Hunyuan、CogVideoX、Cosmos。多视角数据（RoboCasa、DROID）拼成 2×2 网格（左/右/腕相机+黑块）一起生成。微调程度用 bench 的两个指标（instruction following / physics following）判定何时"适配到位"——即**训练视频模型时就用下游代理指标做模型选择**，这是第一处评估逻辑闭环。

**Stage 2 视频 rollout**。用新初始帧 + 语言指令批量生成。仿真实验从模拟器随机化物体位置取新初始帧；真机实验人工拍摄新初始帧（含 10 个全新环境）；新行为则人工撰写新动词 prompt。注意：初始帧仍需少量人工，但远便宜于遥操。

**Stage 3 伪动作标注（两条路线）**。
- **IDM**：diffusion transformer + SigLIP-2 视觉编码器，flow matching 目标；输入首尾两帧、输出中间 action chunk；**刻意不输入语言与本体感知**，让 IDM 只学机器人动力学。推理时滑窗逐段标注。IDM 训练数据与视频模型微调数据相同。
- **LAPA latent action**：VQ-VAE 目标，在 438M 帧（5721 小时，含真机/仿真/人类视频）上预训练，取前后帧（间隔 1 秒）之间的 pre-quantized 连续嵌入作为 latent action。优点是**目标 embodiment 无需任何动作真值**。
- 两者在 RoboCasa 上收益相当；默认选 IDM，因为 IDM 动作可以只用神经轨迹训练并直接评估 policy，且各实验都有足够遥操数据训一个够强的 IDM。

**Stage 4 policy 训练**。神经轨迹没有 state，就把 state 置零（初步实验表明无损）。两种模式：与真实轨迹 1:1 co-training，或**纯神经轨迹训练**。GR00T N1 把神经轨迹当作独立 embodiment（单独 action encoder/decoder），作者推测这是 GR00T N1 增益大于 Diffusion Policy 和 π0 的原因。管线对 policy 架构无关，在 DP、π0、GR00T N1 上都验证有效。

### 实验与结果

**数据增广（已有任务）**：
- RoboCasa 24 任务：合成数据最多扩到人类演示的 333×，policy 成功率随神经轨迹数量呈**对数线性**提升；三个真值数据档位（720/2.4k/7.2k）下 co-training 平均成功率分别 17.4→23.3、32.1→39.9、49.6→57.6；**纯神经轨迹**也能到 20.6%（接近 30 条真实轨迹的水平）。
- 真机 9 任务、3 个 embodiment（GR1 人形、Franka、SO-100），每任务只用 10–13 条真实轨迹：GR1 平均 37→46.4%，Franka 23→37%，SO-100 21→45.5%。任务含锤钉、擦液体、叠毛巾、舀 M&M 等**难以仿真**的接触/可变形任务——这正是"神经渲染数据引擎"相对传统仿真管线的差异化生态位。

**行为泛化**：视频模型只在 2,884 条 GR1 pick-and-place 上微调，却能按 prompt 生成完全没见过的动词（倒水、开合铰接物、用工具）。14 个新行为任务、每任务 50 条神经轨迹、**纯神经轨迹训练**：11.2%→43.2%。

**环境泛化**：只在单一实验室环境微调，用 10 个新环境的初始帧 prompt。13 个任务（新环境×旧行为 + 新环境×新行为）：基线 0%→28.5%。作者强调这不同于靠扩多环境采集数据的路线（如 π0.5），是"零额外物理采集"的 zero-to-one。

**成本**：240k 条 RoboCasa 神经轨迹耗 1500 张 L40 × 54 小时——数据引擎的算力成本非常可观，这也是"先用便宜代理指标筛模型/筛配方，再上大规模生成"的动机所在。

### DreamGen Bench：本篇最值得搬走的方法论

设计目标：不需要真机在环，量化"视频世界模型适配某个 embodiment 并泛化到新物体/行为/环境"的能力，并验证该分数可预测下游 policy 收益。

**两个代理指标**：
- **Instruction Following (IF)**：把生成视频喂给 Qwen2.5-VL-7B（另测 GPT-4o），按固定 prompt 让其给"视频是否完成指令任务"打 0/1 分。zero-shot 模型另加"出现人手直接判 0"等严格规则。
- **Physics Alignment (PA)**：VideoCon-Physics（VideoPhy 论文配套的 7B 物理合理性评估器）与 Qwen2.5-VL 物理判断二者取平均——因为 VideoCon-Physics 没见过多视角网格视频和多样机器人环境，单用会失真。

**人评校准**：对 3 个微调模型的全部样本做人评，IF(GPT-4o) 与人评的 Pearson r 在四个子集上为 0.94 / 0.93 / 0.96 / 1.00，IF(Qwen) 为 0.92–0.97。注意：**Qwen 与 GPT-4o 的绝对分差异巨大**（如 RoboCasa 上 IF-GPT 79.2 vs IF-Qwen 29.2），说明这类 VLM 代理分只有**排序意义**，绝对值不可跨评估器比较。

**相关性验证（核心实验）**：对 Table 2 的每个视频模型（4 底模 × zero-shot/sft）各生成 7k 神经轨迹，用**同一配方**训 GR00T N1、在 RoboCasa 评估。DreamGen Bench 分数（IF-GPT 与 PA 的平均）与 policy 成功率呈正相关（Fig. 6）。结论：更强的（指令+物理）视频世界模型 → 更高下游收益，代理指标可作为选型替身。

**中间验证层（重要但藏在附录）**：有数字孪生时，把 IDM 伪动作在仿真里 replay，可以区分"瓶颈在视频质量"还是"瓶颈在 IDM"。作者的经验结论：**瓶颈主要在神经轨迹（视频）质量而非 IDM**——即视频模型的指令/物理质量是主要矛盾，动作标注是次要矛盾。

### 局限性（作者自述 + 批判性补充）

作者自述：任务较简单、未覆盖机器人全运动能力；算力大；初始帧靠人工；轻量开源评估器会幻觉（尤其物理判断）。

批判性补充：
1. **相关性证据薄弱**：Fig. 6 的相关只有 8 个点（4 模型×2 设置），没有报告相关系数的显著性/置信区间；zero-shot 模型几乎全是 0 分，实际有效点更少。方法论方向正确，但统计强度不足，照搬时应加秩相关+bootstrap。
2. bench 分数 = IF 与 PA 的简单平均，权重未标定；两指标对 policy 增益的边际贡献未做分解。
3. 相关性是**模型级**（哪个生成器更好）而非**样本级**（哪条数据更好）——不能直接推出"用 IF/PA 分数过滤单条数据能提升 policy"，样本级过滤是未验证的开放问题。
4. IDM 训练与视频微调用同一批遥操数据，二者误差可能相关；纯神经轨迹训练的 20.6% 里无法完全归因。
5. 环境/行为泛化的评分含部分分（如"拿起瓶子"给 0.5），跨论文比较时注意口径。

### 与 SB-Render-Lite 的关系

- DreamGen 的角色是"**造新数据**"（新行为/新环境覆盖），SB-Render-Lite 是"**翻译旧数据**"（降低 sim→real 视觉 gap）——两者正交且可组合：SB 翻译后的 real-style 视频完全可以进 DreamGen 式管线（伪动作再标注甚至不需要，因为翻译保留了 sim 动作真值）。
- DreamGen 的评估闭环（代理指标→固定配方→policy success 相关）是 SB-Render-Lite 评估协议的直接模板，详见第四节笔记。
- DreamGen 对"多视角拼网格"的处理、state 置零、独立 embodiment head 等工程细节，对我们后续"翻译视频 co-training"实验有直接参考价值。

---

## 二、半精读：UniSim — Learning Interactive Real-World Simulators

### 基本信息

- 论文：Learning Interactive Real-World Simulators
- 作者：Sherry (Mengjiao) Yang, Yilun Du, Seyed Kamyar Seyed Ghasemipour, Jonathan Tompson, Leslie Kaelbling, Dale Schuurmans, Pieter Abbeel（UC Berkeley / Google DeepMind / MIT / U Alberta）
- 时间：arXiv 2023-10-09（2310.06114）
- venue：**ICLR 2024 oral，Outstanding Paper Award（5 篇之一）**。web 复核 2026-08-14：iclr.cc awards 页、ICLR 官方博客与新闻稿、OpenReview forum sFyTZEqmUY 三源一致
- 链接：https://arxiv.org/abs/2310.06114 ｜ 项目页：https://universal-simulator.github.io
- 全文获取：arXiv HTML 全文，2026-08-14
- 归类：观测预测世界模型；交互式神经仿真器；仿真内训练+零样本真机迁移

### 一句话总结

UniSim 把"真实世界模拟器"形式化为**观测预测模型** \(p(o_t \mid h_{t-1}, a_{t-1})\)：以视频扩散模型（5.6B U-Net）统一"动作进、视频出"接口，靠编排多源异构数据（互联网图文、人类活动视频、真机/仿真机器人数据、全景扫描）学出可自回归 rollout 的交互式环境，并演示高层 VLM policy、低层 RL policy 和视频字幕模型都能在这个神经仿真器里训练后零样本迁移到真实世界。

### 方法核心

**数据编排（贡献之一）**：关键观察是各数据集沿不同轴各有富余——图像数据物体丰富、机器人数据动作稠密、导航数据运动多样。处理策略：文本动作统一成 T5 嵌入；连续控制离散化为 4096 bins 后与文本嵌入拼接；全景扫描裁剪出"左转"等伪动作；无动作图像当单帧视频。训练混合含 Habitat HM3D（仅 710 条）、Language Table 仿真 160k/真实 440k、Bridge 2k、RT-1 70k、Ego4D 3.5M、SSV2 160k、EPIC-KITCHENS 25k、Matterport 3.5M、LAION-400M、ALIGN 400M、杂项互联网视频 13M。消融显示**去掉互联网数据 FVD 显著变差**（211→308）；低资源域加 dataset 标识符可提升域内质量但伤跨域泛化。

**观测预测模型**：条件于最近 4 帧历史（消融：4 recent > 4 distant > 1 frame；FVD 315.7→211.3）+ 动作嵌入，classifier-free guidance 控制动作条件强度，自回归拼接实现长时程交互（8 步连续指令，物体持久性基本保持）。架构为 video U-Net（时空交错注意力/卷积），基础模型 16×24×40 + 两级空间超分到 192×320；512 TPU-v3 训 20 天。模型规模消融：500M→1.6B→5.6B FVD 单调改善但**收益递减**（277.9→224.6→211.3）。

### 三个应用（每个都是"仿真内训练→真实收益"）

1. **高层 VLM policy + hindsight relabeling**：在 UniSim 里 rollout 3–5 条脚本指令拼出 10k 条长时程轨迹，以末帧为 goal 训练 PALM-E 风格的 goal-conditioned VLM。长时程 Language Table 任务上 RDG（距目标距离缩减比）从 0.07–0.11 提到 0.34（约 3–4×）。执行时由 VLM 出指令、UniSim 出视频、**逆动力学模型从视频恢复低层动作**上真机。
2. **低层 RL policy**：PaLI-3B VLA 做 BC 初始化，用"距完成剩余步数"训练 reward model，把 UniSim 当环境跑 REINFORCE。48 任务成功率 0.58→0.81，点指类任务 0.12→0.71，零样本部署真实 Language Table 成功。
3. **视频字幕数据引擎**：用 UniSim 按 ActivityNet 训练集文本生成 4× 视频训 PaLI-X：CIDEr 15.2→46.23（达到真数据微调的 84%），且**跨数据集迁移优于真数据微调**（MSR-VTT/VATEX/SMIT 上更高）——生成数据比真数据"更干净对齐"的少见证据。

### 实验评估的口径（与 DreamGen 对照的关键点）

UniSim 的模型选型/消融全部用**通用视觉指标**（FID、FVD、IS、CLIP score），下游收益则分别用 RDG、任务成功率、CIDEr 单独度量——**论文从未把两层指标做相关性连接**。且 RL policy 的成功率是"在 UniSim 里 rollout 后定性评估"的，存在"用世界模型自己评在世界模型里训的 policy"的自洽偏置风险（真机部分为定性展示）。这个方法论空缺正是 DreamGen Bench 两年后补上的位置：UniSim 证明"世界模型可以当数据引擎/环境"，DreamGen 证明"世界模型的质量可以被便宜地预测并映射到 policy 收益"。

### 局限性

作者自述四条：不可行动作会诱发幻觉（桌面机器人被指令"洗手"时凭空生成水槽）；有限记忆（4 帧历史外的物体状态会丢）；域外 embodiment 泛化差（只训过 4 种形态）；只模拟视觉（力/触觉/声音缺失，视觉不变的动作无法区分）。补充：5.6B+512 TPU 门槛极高；动作接口以文本嵌入为主，低层控制只在 Language Table 一类平面任务上验证了精确性；FVD 收益随规模递减暗示数据编排比堆参数更关键。

### 与 SB-Render-Lite 的关系

- UniSim 的"observation prediction + 自回归"是**动力学级**世界模型；SB-Render-Lite 是**外观级**翻译（不改动力学、逐帧/逐段翻译）。两者的评估差异提示我们：外观翻译的质量指标里必须显式保护"动力学可读性"（动作可从画面恢复），否则翻译再逼真也是废数据——这恰是 UniSim 用逆动力学从生成视频恢复动作、DreamGen 用 IDM replay 验证的共同直觉。
- UniSim 的 hindsight relabeling 思路提示：翻译后的数据不只可以 1:1 替换原 sim 数据，还可以重组（改 goal、拼接段落）放大有效数据量。

---

## 三、收录条目（catalog entries）

以下条目不做精读，仅记录元信息、定位与主库相关性。venue 均于 2026-08-14 web 复核。

### C1. Cosmos World Foundation Model Platform for Physical AI

- arXiv 2501.03575（2025-01-07），NVIDIA。**venue：arXiv 技术报告/预印本，未见正式会议发表**（复核 2026-08-14）。代码开源、权重开放（NVIDIA Open Model License）
- 定位：Physical AI 世界基础模型平台——视频策管管线、视频 tokenizer、扩散/自回归两族预训练 WFM（后演化为 Predict / Transfer / Reason 三分支），以及 post-training 示例（相机控制、机器人操作指令跟随、自动驾驶多视角）。"预训练通用世界模型 + 下游微调成定制世界模型"的平台化主张
- 与主库关系：DreamGen Bench 中 Cosmos 是 zero-shot 最强底模（zero-shot PA 22.9–32.0，其余底模几乎为 0），说明"物理先验强的底模在目标域适配前就有可测优势"——底模选择本身就是代理指标可以介入的决策点。备注：2026 年 NVIDIA 已发布后继 Cosmos 3（omnimodal 世界模型技术报告，统一语言/图像/视频/音频/动作），本库暂不展开

### C2. Cosmos-Transfer1: Conditional World Generation with Adaptive Multimodal Control

- arXiv 2503.14492（2025-03-18），NVIDIA。**venue：arXiv 预印本**（复核 2026-08-14），代码与权重开源
- 定位：在 Cosmos-Predict1-7B 上加 ControlNet 分支（vis/blur、edge、depth、segmentation，AV 版另有 LiDAR/HDMap），支持**时空自适应控制权重图**：同一画面不同区域可用不同模态、不同强度控制。官方给出 robotics Sim2Real 案例：20 个 Isaac Lab 厨房操作场景（TAMP 生成动作），**前景机器人用 Edge+Seg+Vis 强约束、背景只用 Seg 弱约束**，实现"机器人保真 + 背景随机化"的神经渲染增强
- 评估口径（对第四节笔记很重要）：自建 TransferBench（600 例，含 AgiBot World 机器人子集 200 例），指标为控制信号遵从度——Blur SSIM、Edge F1、Depth si-RMSE、Mask mIoU（可分前景/背景统计）+ 多样性（LPIPS）+ VLM 质量分。**全程没有闭环到 policy success**，即"结构保持指标是否预测下游收益"在该论文中是未验证的假设
- 与主库关系：这是 SB-Render-Lite 最直接的**工业级竞争基线**（同样做 sim→real 神经渲染翻译），应进主库 baseline 表。其"FG 保真 / BG 多样化"权重设计与主库 EgoBridge/GSBM 的 task-aware cost 思想同构，可翻译成 SB 的空间加权 cost

### C3. Video Prediction Policy (VPP)

- arXiv 2412.14803（2024-12），Yucheng Hu, Yanjiang Guo 等（清华/星动纪元等）。**venue：ICML 2025 Spotlight**（项目页与第三方论文笔记一致，复核 2026-08-14）
- 定位：不把视频模型的**像素输出**当数据，而是取视频扩散模型单次前向的**中间表征**（隐含"当前帧+未来预测"的 predictive visual representation）直接条件化动作头，等价于在预测表征上学隐式逆动力学；绕开多步去噪，实现高频闭环控制。CALVIN ABC-D 相对 SOTA +18.6%（v2 摘要报 41.5%），真实灵巧手任务 +31.6%
- 与主库关系：提示"transport 质量"不必以像素为载体——若 SB 桥在 latent 空间搭（主库 EgoBridge/SB Flow 一线），其中间表征本身可能就是 policy 的更好输入；也提示评估翻译质量时可以直接测"翻译表征对动作预测的有效性"而非绕道像素指标

### C4. UniPi: Learning Universal Policies via Text-Guided Video Generation

- arXiv 2302.00111（2023-02），Yilun Du, Sherry Yang 等。**venue：NeurIPS 2023**（DreamGen 参考文献注录为 NeurIPS volume 36: 9156–9172；复核 2026-08-14 一致）
- 定位："视频生成即策略"路线原点：文本条件视频模型生成未来帧序列作为 plan，逆动力学模型从帧间恢复可执行动作。DreamGen 明确把自己与这条"世界模型当实时 planner"的路线区分开（planner 推理太慢、难上真机高频控制），改为离线数据生成器
- 与主库关系：UniPi→UniSim→DreamGen 构成"视频模型进机器人"的谱系（planner → 交互环境 → 数据引擎）；SB-Render-Lite 属于第四种用法（域翻译器），写综述定位时可用这个谱系

### C5. VideoPhy: Evaluating Physical Commonsense for Video Generation

- arXiv 2406.03520（2024-06），Hritik Bansal, Zongyu Lin 等（UCLA/Google）。**venue：ICLR 2025 poster**（iclr.cc virtual 页复核 2026-08-14）
- 定位：688 条涉及 solid-solid / solid-fluid / fluid-fluid 交互的 prompt，人评"语义遵从+物理合理"联合达标率——最好的开源模型 CogVideoX-5B 也只有 39.6%，说明**物理合理性是当前视频生成的主要短板**。配套开源 7B 自动评估器 VideoCon-Physics（在人评标注上微调 VideoCon 得到）
- 与主库关系：DreamGen Bench 的 PA 指标直接复用 VideoCon-Physics（并因其没见过多视角/机器人域而混入 Qwen 分数取平均）。教训有二：(a) 代理评估器有自己的训练分布，跨域使用必须重新校准；(b) 物理合理性同样是 SB 翻译的风险轴——翻译若破坏接触/遮挡/影子等物理线索，policy 学到的视觉-动力学关联会失真

---

## 四、评估逻辑笔记："transport 质量 ↔ policy success 相关性"的可迁移做法

这是本次扩充的核心产出：从 DreamGen Bench（显式闭环）、UniSim（通用指标+单独任务验证）、Cosmos-Transfer1（结构保持指标、未闭环）三种口径中，提炼 SB-Render-Lite 可直接采用的评估协议。

### 4.1 三篇工作的评估口径对照

| 工作 | 代理指标 | 是否闭环到 policy | 结论强度 |
| --- | --- | --- | --- |
| UniSim | FID/FVD/IS/CLIP（模型选型消融用） | 否（应用收益单独测 RDG/成功率/CIDEr，与视觉指标无相关性分析） | 证明世界模型有用，未证明"哪个指标预测有用" |
| Cosmos-Transfer1 | Blur SSIM / Edge F1 / Depth si-RMSE / Mask mIoU（分 FG/BG）+ LPIPS 多样性 + VLM 质量分 | 否（robotics Sim2Real 案例只评结构保持与画质） | 给出结构保持的**测量工具箱**，但"结构保持→policy 收益"是未验证假设 |
| DreamGen | IF（VLM 判任务完成，人评校准 r>0.9）+ PA（VideoCon-Physics+Qwen 平均） | **是**：8 个生成器 × 各 7k 轨迹 × 同一 policy 配方 → bench 分与 RoboCasa 成功率正相关；另有 IDM replay 中间层 | 方向性结论成立；统计强度弱（n=8、无显著性检验、模型级而非样本级） |

### 4.2 可迁移的方法论骨架（DreamGen Bench 模式的抽象）

1. **代理指标分层**，按"离 policy 的因果距离"从远到近：
   - L0 全局分布指标：FID/FVD/CLIP/DINO 距离。便宜、与任务成功因果最远，只做 sanity check；
   - L1 结构/几何保持：edge/depth/mask 对齐（Cosmos-Transfer1 工具箱），衡量"翻译没有破坏场景骨架"；
   - L2 任务语义：VLM 判"指令相关的物体、关系、结果是否仍成立"（DreamGen IF）；
   - L3 物理/时序合理性：VideoCon-Physics 类评估器 + 时序一致性（DreamGen PA）；
   - L4 动作可恢复性：从生成/翻译序列恢复动作并验证（DreamGen 的 IDM replay；UniSim 的逆动力学恢复执行）——**最接近 policy、最值得投入的代理**；
   - L5 终极验证：固定训练配方的小规模 policy 扫描 + 秩相关。
2. **自动评估器必须人评校准**：DreamGen 用 Pearson>0.9 作为可用门槛；且证据显示不同 VLM 的绝对分不可比（IF-GPT 79.2 vs IF-Qwen 29.2），**只用排序、不用绝对值**。
3. **相关性研究的实验设计**：变体（生成器/翻译器）× 固定生成量 × 固定 policy 配方 × 固定评估任务集；改进空间：点数 ≥8、报告 Spearman + bootstrap CI、区分模型级与样本级结论。
4. **中间验证层**用来做故障归因：digital twin 里 replay 恢复动作，可区分"数据质量瓶颈"与"动作标注瓶颈"（DreamGen 的结论是瓶颈几乎总在视频质量端）。

### 4.3 映射到 SB-Render-Lite：翻译图像的哪些指标最可能与 policy success 相关

SB-Render-Lite 与 DreamGen 有一个关键差异，使我们的处境**更有利**：DreamGen 的生成视频没有动作真值（必须 IDM 伪标注，replay 只能间接验证）；而我们是**翻译已有 sim 轨迹**，翻译前后动作真值 \(a_{1:T}\) 不变。因此：

**建议的代理指标集（按预期相关性排序）**：

1. **动作一致性（L4，首选）**：取一个在 real（或 real-style）数据上训练的 IDM/逆动力学模型 \(f\)，对翻译后序列预测 \(\hat a = f(\tilde o_{t}, \tilde o_{t+k})\)，直接算 \(\|\hat a - a\|\)（动作 MSE / chunk DTW 距离）。它同时测两件事：翻译是否保留了动作可读的视觉线索（末端执行器、接触、物体位移），以及翻译是否把外观搬进了 real 域（否则 real 域 IDM 读不懂）。这等价于把 DreamGen 的 replay 检查升级成**有真值监督的回归检查**，无需数字孪生在环。
2. **前景结构保持（L1，加权版）**：借 Cosmos-Transfer1 的 FG/BG 拆分——只在 end-effector 与任务相关物体的 mask 内算 Mask mIoU、keypoint 偏移、Depth si-RMSE；背景区域宽容甚至鼓励多样化（等价于"神经域随机化"，对 policy 是收益而非损失）。均匀全图的 SSIM/LPIPS 会把"背景变化"误判为质量下降。
3. **指令/任务语义一致性（L2）**：VLM 二值判别"目标物体是否仍在、可抓、空间关系是否保持"；需按 DreamGen 流程先抽样人评校准（r>0.9 再采信），并预期需要为操作场景定制 prompt。
4. **全局分布指标（L0）**：FID/DINO-to-real、FVD 只报作 reviewer-facing 数字与 sanity check；UniSim 的证据（FVD 随模型规模改善但收益递减）与 DreamGen 的证据（zero-shot 模型 FVD 类指标看不出的差距在 IF/PA 上巨大）都表明它们对下游收益的分辨率低。

**提前验证协议（在大规模翻译之前做的 correlation study）**：

- 变体集合 K≥6：不同 SB 训练步数的 checkpoint、不同 cost 设计（纯视觉 vs +geometry/action cost）、外部基线（CycleGAN、I²SB、SB Flow、Cosmos-Transfer1[Seg]/[Edge]）——checkpoint 序列是免费的变体来源，可显著加点数；
- 每个变体翻译**同一批** sim 轨迹（固定数量），用**同一配方**训同一 policy（学 DreamGen：低数据档位更能放大差异），在同一任务集评 success；
- 计算每个代理指标与 success 的 Spearman 秩相关 + bootstrap CI；秩相关稳定 ≥0.8 的指标晋级为后续超参扫描的替身指标；
- 顺带回答样本级问题（DreamGen 未做）：在最好的变体内部，用代理分数把翻译样本分高/低两半分别训 policy，检验样本级过滤是否有效。

**已知陷阱清单**：

- 评估器幻觉与域外失效（VideoCon-Physics 没见过多视角机器人视频→DreamGen 被迫混 Qwen 分）；
- 绝对分不可跨评估器/跨 prompt 比较，只用排序；
- 天花板/地板效应：变体都很好或都很差时相关性测不出（DreamGen 的 zero-shot 组几乎全 0）；
- 模型级相关 ≠ 样本级相关；
- 多样性指标的角色随任务而变：DreamGen 里行为多样性是收益，我们这里**前景保真是收益、背景多样是收益、前景多样是灾难**——必须分区统计。

---

## 并入主库建议

1. **建议新增分类**："世界模型 / 视频生成数据引擎"（或挂在"核心：仿真/真机迁移"下作扩展小节），收录本文两篇精读与五条条目；DreamGen 报告建议按主库格式拆出独立文件（`2505.12705_dreamgen.md`、`2310.06114_unisim.md`）时可直接复用本文第一、二节内容。
2. **INDEX.md / synthesis.md 的互链点**（本次未改动任何现有文件，仅给建议）：
   - `synthesis.md` 的评估协议一节应引用第四节的"代理指标分层 + 相关性预研协议"，把主库原则"所有视觉指标都应服从 downstream real-domain policy success"（INDEX.md 结语）落成可操作流程；
   - `2509.18631_guided_ot_sim_real_policy_cotraining.md`（sim-real co-training）与 DreamGen 的 co-training 配比（1:1、state 置零、独立 embodiment head）可互引；
   - `2409.09347_schrodinger_bridge_flow_unpaired_translation.md` 与 C2 Cosmos-Transfer1 应同列为 SB-Render-Lite 的 unpaired/conditional 翻译基线。
3. **对 SB-Render-Lite 实验计划的三条直接行动**：
   - 在评估套件加入 L4"动作一致性"（real 域 IDM 回归检查）与 L1"前景结构保持"（FG mask 内 mIoU/keypoint/depth），并在第一批 checkpoint 上跑 K≥6 变体的相关性预研；
   - 把 Cosmos-Transfer1[Seg]（开源）纳入 baseline 表，作为"工业级神经渲染"对照；其 FG/BG 时空控制权重思想翻译成 SB 的空间加权 cost（前景 geometry/action cost 高权、背景低权）；
   - 若翻译质量验证通过，按 DreamGen 配方做一次"翻译数据 co-training"端到端实验（低数据档位、1:1 采样比），直接复用其 RoboCasa 式 log-linear 扫描叙事作为论文主图模板。

（完；检索/核验日期均为 2026-08-14）
