# E13 扩散语义增广基线：ROSIE 精读 + inpainting 增广对照协议

## 选题定位

- 缺口来源：库内 25 篇覆盖了 OT/SB 的 transport 方法线（对齐、翻译、bridge），但缺少"**不训练 transport 模型、直接用现成扩散模型做语义增广**"这条低成本竞争基线。SB-Render-Lite 的实验设计需要它作对照：如果 text-guided inpainting 就能造出足够的视觉多样性提升真机成功率，那 SB transport 的附加值必须被明确界定。
- 本文件：ROSIE（arXiv 2302.11550）精读 ×1；GenAug（2302.06671）、CACTI（2212.05711）、SynthER（2303.06614）收录条目 ×3；"扩散增广 vs SB transport"公平对照协议 ×1。
- 全部四篇均为库内新增，不与现有 25 篇报告重复；venue 已于 **2026-08-14** web 复核（见 §4）。
- 全文获取方式：ROSIE 经 ar5iv HTML 全文精读；GenAug / CACTI / SynthER 经 arXiv abs + HTML 正文获取方法与实验主体（收录级深度，非精读）。

## TL;DR

1. **ROSIE（RSS 2023，Google）证实文本引导 inpainting 是强语义增广基线**：在 RT-1 的 130k 真实演示上，用 OWL-ViT 定位 + Imagen Editor 补绘造出新物体/新背景/新干扰物，真机上把"从未采过数据的任务"（放罐头进水槽）从 0% 拉到 60%，新背景抓取从 33% 提到 71%，且不需要任何额外真机采集。
2. **但它只重绘 mask 内区域的语义外观，不动全图外观统计**：光照模型、材质 BRDF、传感器噪声、sim 渲染整体风格这类 appearance gap 完全不在覆盖范围（论文也自认不增广物理与动作）。这恰好是 SB transport 的主场——两类方法吃的是 domain gap 的不同分量，天然构成分解对照。
3. **公平对比的关键陷阱是"先验知识通道"不对称**：ROSIE 的增广 prompt 直接瞄准评测物体（先补绘 sink 再去真 sink 评测），相当于把测试语义注入训练。协议必须区分 P-oracle / P-blind 两种 prompt 设置，并在同等生成预算、同等训练配比下用 S-gap / A-gap 双测试套件做归因（§3）。
4. venue 纪律：ROSIE = RSS 2023（DOI 10.15607/RSS.2023.XIX.027）、GenAug = RSS 2023（XIX.010）、SynthER = NeurIPS 2023 均确认；**CACTI 仅为 CoRL 2022 PRL workshop poster + arXiv 预印本**，引用时不得写成主会论文。

---

## 一、精读：ROSIE — Scaling Robot Learning with Semantically Imagined Experience

### 基本信息

- 论文：Scaling Robot Learning with Semantically Imagined Experience
- 作者：Tianhe Yu, Ted Xiao, Austin Stone, Jonathan Tompson, Anthony Brohan, Su Wang, Jaspiar Singh, Clayton Tan, Dee M, Jodilyn Peralta, Brian Ichter, Karol Hausman, Fei Xia（Robotics at Google / Google Research）
- 时间：arXiv 2023-02-22（2302.11550）
- venue：**RSS 2023**（Robotics: Science and Systems XIX，DOI [10.15607/rss.2023.xix.027](https://doi.org/10.15607/rss.2023.xix.027)；项目页亦标注 RSS 2023；复核日期 2026-08-14）
- 链接：https://arxiv.org/abs/2302.11550 ｜ 项目页：https://diffusion-rosie.github.io
- 全文获取：ar5iv HTML 全文（含附录 A–C）✅
- 归类：diffusion semantic augmentation；text-guided inpainting；robot data scaling；**SB-Render-Lite 的低成本竞争基线**。

### 一句话总结

ROSIE 不采集任何新真机数据，直接在已有真实演示轨迹上用开放词表分割定位可改区域、用文本引导的扩散补绘（Imagen Editor）换掉操作目标/容器/背景或加入干扰物，动作标签原样保留、指令同步改写，再与原数据 1:1 混合微调 RT-1，即可让策略学会"只在扩散模型想象里见过"的新任务并对干扰物更鲁棒。

### 动机与问题定义

真机数据的采集成本是机器人学习规模化的瓶颈：RT-1 的 130k 演示（744 条指令）花了 13 台机器人 17 个月。仿真是一条出路但要解决 sim-to-real；传统数据增广（裁剪、色彩抖动、噪声）只提供低层不变性，**造不出语义上全新的经验**（新物体、新容器、新背景）。ROSIE 的赌注是：互联网规模预训练的 text-to-image 扩散模型已经"见过"这些语义变体，可以零样本把它们蒸馏进机器人数据——本质上是把生成模型当作"语义先验的数据放大器"，而不是当作 domain translator。

值得注意的定位差异：ROSIE 在 related work 里明确把自己与 RL-CycleGAN、RetinaGAN 这类 sim-to-real 翻译方法区分开——它**直接增广真实数据**，不做跨域翻译。这意味着把 ROSIE 拿来当 SB-Render-Lite 的基线时，需要把配方重新实例化到 sim 帧上（见 §3.3 的 A2-sim 臂），这是一次超出原论文验证范围的外推，协议里必须说明。

### 方法核心

管线三步（对每条 episode 的每一帧执行，mask 与 prompt 全轨迹共享）：

1. **增广区域定位（open-vocab segmentation）**：用 OWL-ViT（冻结主干 + 在 Open-Images-V5 上微调的实例分割头）按语言查询检测目标区域。关键工程细节是 **passthrough objects**：把机械臂、夹爪、手中物体的 mask 从目标区域中减掉，保证不改到执行主体。例如"往打开的抽屉里加干扰物"= detect(抽屉) − detect(机械臂) − detect(手中罐头)。不同任务用不同检测置信度阈值（0.03–0.3，附录 A.1）。
2. **增广文本提议**：两种来源。(a) 人工指定——用于刻意制造训练分布外的评测目标；(b) **LLM 提议（GPT-3，few-shot）**——给出源任务/目标任务，让 LLM 同时产出 OWL-ViT 检测 prompt、passthrough prompt 和 inpainting prompt。附录 C 显示 zero-shot 提示会幻觉，few-shot 是必要条件。
3. **文本引导补绘（Imagen Editor）**：级联扩散（64×64 base + 256×256 SR），对 mask 区域按增广文本重绘，mask 外像素严格不动。若增广产生新任务则同步改写指令 ℓ→ℓ̃（"pick green rice chip bag"→"pick blue microfiber cloth"），**动作序列 {aᵢ} 原样保留**。

下游训练：预训练 RT-1（35M 参数）上以原数据 : 增广数据 = **1:1** 混合微调 85k 步（LR 1e-6）。

**增广范围与约束**（这是对照协议的直接输入）：

- 可改：操作目标物体（含手中可形变物体，如把薯片袋换成超细纤维布）、目标容器（碗→篮子/砂锅/玻璃罐）、桌面/背景（桌布、把抽屉/台面换成金属水槽）、干扰物（桌上多个可乐罐、抽屉内杂物）。
- 不可改 / 隐含约束：
  - **动作等价性假设**：增广后的场景必须让原动作仍然有效 ⇒ 物体位置、几何尺度、接触点不能大改；mask 形状即物体轮廓，本质上限制为"同位置换皮/换物"。
  - **布局不可改**：不能移动物体、不能改机器人形态。
  - **物理不可改**：摩擦、质量、形变行为全部继承原轨迹（论文 §7 自认）。
  - **逐帧独立生成，无时序一致性保证**：同一 prompt 在相邻帧可能生成外观漂移的补丁；作者报告 RT-1 未因此掉点，但明确这是架构依赖的经验观察，并推测换 text-to-video 模型会牺牲照片真实感。
- 计算开销（附录 A.2）：OWL-ViT 1 TPU·1h / 1k episodes；Imagen Editor 4 TPUs·2h / 1k episodes；策略训练 16 TPUs·1 天。**离线增广，无法 on-the-fly**——这构成生成预算对照的基础数据点。

### 真机实验与结果

数据底座为 RT-1 kitchen 数据集（130k 演示、744 指令），评测共 243 个真机 rollout，对照组为预训练 RT-1（NoAug）和只改指令不改图像的 InstructionAug。

RQ1（学新任务，越往下越难）：

| 任务族 | NoAug | InstructionAug | ROSIE |
|---|---|---|---|
| 移动物体到新容器旁（254 eps 增广源） | 0.86 | 0.78 | **0.94** |
| 抓取全新可形变物体（布，1309 eps 源） | 0.25 | 0.30 | **0.75** |
| 放物体进新容器 | 0.13 | 0.25 | **0.44** |
| 放罐头进（补绘出来的）水槽（779 eps 源） | 0.00 | – | **0.60** |

最后一行是论文的招牌结果：真机**从未**采过任何带水槽的数据，把"放罐头进抽屉"的演示批量补绘成"放罐头进金属水槽"，策略即可在真实厨房的真水槽上做到 60%（NoAug 完全找不到目标，0%）。InstructionAug 的普遍失败说明收益确实来自图像语义变化，不是指令 relabel。

RQ2（鲁棒性）：新背景抓取（桌布/水槽旁等 8 个设置）0.33→**0.71**（7/8 设置占优）；杂乱抽屉放置 0.38→**0.55**；OOD 干扰物抓取 0.33→0.37（增益很小——只加干扰物的增广对已有强预训练的 RT-1 边际效益有限）。

RQ3（成功检测）：用 ROSIE 增广 22764 条放置 episodes 微调 CLIP 成功检测器，困难 OOD 集（杂乱抽屉）F1 从 0.19 提到 0.57，in-distribution 不掉点（0.66）——证明语义增广对高层具身推理任务同样有效，且增广量单调有益。

批判性阅读：(1) 评测规模不大（每设置 8–10 rollouts，无置信区间），个别子任务噪声明显；(2) **增广 prompt 与评测目标是同一批人选的**——"选训练分布外的物体"实际上是选"即将拿去评测的物体"，语义先验直接对准了测试分布（详见 §3.4 的公平性讨论）；(3) 失败模式（附录 C）：手中物体 mask 不规则时补绘退化（布只换了一半）、生成的"玻璃罐"长成碗形——作者的辩护是错误生成只造成指令-图像失配，实测伤害不大。

### 局限性：它覆盖不了哪些域差（本库最关心的问题)

把 sim→real 的 domain gap 拆成四个分量，逐一对照：

| gap 分量 | 典型内容 | ROSIE 覆盖？ | 说明 |
|---|---|---|---|
| **语义/内容差** | 物体类别与外观身份、干扰物集合、背景语义 | ✅ 核心覆盖 | 这是它的全部设计目标 |
| **外观/渲染差** | 光照方向与色温、阴影模型、材质 BRDF、传感器噪声/白平衡/动态范围、镜头畸变、sim 渲染整体风格 | ❌ 结构性不覆盖 | inpainting 只重绘 mask 内像素，**mask 外原样保留**。若把配方搬到 sim 帧上，产物是"照片级补丁 + sim 底图"的混合体，补丁与底图之间的风格断层甚至可能成为新的伪相关特征 |
| **物理/动力学差** | 摩擦、质量、形变、接触动力学 | ❌ 论文自认 | "只增广外观，不生成新运动"；提出未来混仿真数据补运动多样性 |
| **时序一致性** | 帧间外观漂移 | ⚠️ 无保证 | 逐帧独立生成；RT-1 上侥幸无害，不保证迁移到对时序敏感的策略（如 diffusion policy 的视觉编码器） |

这张表就是"扩散语义增广 vs SB transport"分工的理论依据：ROSIE 类方法吃**语义差**，SB transport（I²SB / SB Flow 一系，见库内 `2302.05872_i2sb.md`、`2409.09347_schrodinger_bridge_flow_unpaired_translation.md`）学的是整图分布间的传输，吃**外观差**，但不凭空创造语义新模式（transport 在既有模式间搬运质量，不会长出训练分布里没有的新物体类别）。两者天然互补，也天然构成对照。

### 与 SB-Render-Lite 的关系 / 可借鉴点

- **必须收的基线**：ROSIE 配方零训练成本（全部用现成模型推理），如果它在 sim2real 设置下就够用，SB-Render-Lite 的 transport 训练开销需要用 A-gap 上的增益来辩护。
- `passthrough objects` 的 mask 减法值得直接搬进 SB-Render-Lite 的评估：transport 后的图像同样应检查机械臂/手中物体区域是否被错误改写（可用同一 OWL-ViT 管线做 region-split 一致性检查）。
- 1:1 原始:生成混合比 + 小学习率微调，是经过真机验证的下游训练配比，协议 §3.4 直接沿用作默认值。
- RQ3 提示：增广/transport 数据不仅可以喂策略，还可以喂 success detector——SB-Render-Lite 的产物同样可以在这条更便宜的评估线上先做筛选。
- 反向借鉴（ROSIE 做不到、SB 应该验证的）：全图外观统计的系统性搬移、少 NFE 的结构化先验启动（I²SB 已证明 paired 场景下源图启动可大幅降步数）、时序一致的 video bridge（库内 `2506.10168_momentum_multi_marginal_sbm.md` 的 3MSBM 路线）。

---

## 二、收录条目（×3）

### 2.1 GenAug: Retargeting Behaviors to Unseen Situations via Generative Augmentation

- arXiv 2302.06671（2023-02-13）；**RSS 2023**（XIX，DOI [10.15607/RSS.2023.XIX.010](https://doi.org/10.15607/RSS.2023.XIX.010)，Daegu；复核日期 2026-08-14）
- 作者：Qiuyu (Zoey) Chen, Shosuke Kiami, Abhishek Gupta, Vikash Kumar（UW + Meta AI）｜项目页 genaug.github.io
- 定位：与 ROSIE 同期同类（ROSIE 的 related work 称其 concurrent），但技术路线是 **depth-guided 扩散补绘 + 3D mesh 辅助**，作用在 RGBD 上。
- 方法要点：形式化"语义等价类"——生成模型只造观测 o 的变体、不造动作 a，增广集 {g(t₁,o,z₁),…} 必须共享同一动作标签。三类增广：(1) 目标物体/容器：in-category 用文本换颜色材质（mask 固定 ⇒ 位置形状不变）；cross-category 先渲染随机 3D mesh（GoogleScan/Free3D 共 40 个）保证几何/透视正确，再用 depth-guided 模型上纹理；(2) 干扰物：mesh 渲染 + bounding-box 碰撞检查，避免与目标/容器重叠；(3) 背景：反转 mask 重绘桌面与房间。需要人工标注的物体 mask、标定相机与 RGBD。
- 关键结果：10 任务 × 各 10 条演示（单一环境采集），每条演示增广 100 次；CLIPort 骨干。真机：未见环境 **85%** 成功率，未见放置目标 52%，未见抓取目标 45%；affordance 对照下 GenAug vs 无增广 = 80%/38%（环境）、54%/8%（放置）、46%/10%（抓取）。仿真消融：胜过 Random Copy-Paste、Random Background、R3M 微调等所有对照——**"物理上合理的语义增广"显著优于随意贴图**；增广次数 0/10/50/100 单调提升。
- 局限：不增广动作与物理（摩擦/形变）；假设原轨迹与增广出的杂物不碰撞；帧间无视觉一致性；每场景约 30 s，做不了 on-policy。
- 与本库关系：给 SB-Render-Lite 提供了第二种增广配方（depth-guided、几何一致性更强，代价是要 mask + mesh + RGBD）。它的"随意贴图 vs 语义增广"消融证明**增广的分布真实性本身就值 20–40 个百分点**，这正是 SB transport 声称的核心能力，可以直接对表。

### 2.2 CACTI: A Framework for Scalable Multi-Task Multi-Scene Visual Imitation Learning

- arXiv 2212.05711（v1 2022-12-12，v2 2023-02-16）；**venue 复核（2026-08-14）：CoRL 2022 Workshop on Pre-training Robot Learning (PRL) poster（OpenReview dRHW9-QFj9），未检索到主会/期刊正式发表——引用时按 workshop/preprint 处理**。注意 workshop 版与 arXiv v2 数字不同（前者 5 真机任务/50 布局，后者 10 任务/100 布局），本条目按 arXiv v2。
- 作者：Zhao Mandi, Homanga Bharadhwaj, Vincent Moens, Shuran Song, Aravind Rajeswaran, Vikash Kumar（Columbia + Meta + CMU）
- 定位：四阶段框架 Collect–Augment–Compress–TraIn 中的 Augment 阶段用了 **zero-shot Stable Diffusion inpainting**——是这条线里最早（2022-12）把现成扩散补绘用于真机数据的工作，但 mask 人工指定、只做干扰物/场景变体，**不产生新任务**（ROSIE 对它的差异声明：ROSIE 自动选区 + 造新任务）。
- 方法要点：真机侧 5 条动觉示教 × 20 次带布局扰动重放（10 任务，Franka），再对图像做 SD 补绘加干扰物；仿真侧 900 个单任务 RL 专家（NPG）重放渲染 45k episodes 并做视觉/布局随机化。Compress 用冻结 R3M / in-domain MoCo，策略只是 frozen embedding 上的 MLP BC。
- 关键结果：真机单策略 10 任务平均 ≈30% 成功率；**补绘增广消融：比只用 color-jitter/random-crop 高 15–20 个百分点（绝对值）**；仿真 18 任务 × 100 布局 ≈62%，heldout 布局泛化随训练布局数 10→50→100 从 14%→32%→47% 单调升；端到端 RL 完全失败（0%）。冻结 out-of-domain 表征 ≈ in-domain 表征。
- 与本库关系：提供"增广强度→泛化"的 scaling 证据和最朴素的补绘配方下限；其 frozen-representation 设定与 SB-Render-Lite 的 latent-space transport 选项同构——如果 transport 作用在 R3M/DINO latent 上，CACTI 就是"不 transport 只增广"的对应物。引用纪律：workshop poster，不可写成 CoRL 正式论文。

### 2.3 SynthER: Synthetic Experience Replay

- arXiv 2303.06614；**NeurIPS 2023**（37th，proceedings 收录 + 官方 poster 页；复核日期 2026-08-14）；代码 github.com/conglu1997/SynthER
- 作者：Cong Lu*, Philip J. Ball*, Yee Whye Teh, Jack Parker-Holder（Oxford）
- 定位：与前三篇不同——不是图像语义增广，而是**在低维 transition 空间 (s,a,r,s′,d) 上训练扩散模型做"分布内上采样"**：residual-MLP 去噪器 + Karras (EDM) 采样器（128 步），把 replay/offline 数据任意放大。
- 关键结果：(1) offline：D4RL 上完全用 5M 合成样本替换原数据，TD3+BC/IQL/EDAC 全线打平或更好（maze2d 大涨，EDAC large 95.6→143.3）；(2) 小数据放大：15% walker2d medium-replay 上大幅超过加噪/动力学噪声等显式增广，3% medium-expert 即可逼近全量性能；机制分析：合成样本离数据集**更远**（更多样）但 dynamics MSE **更低**（更合法）——生成式增广同时赢多样性与有效性；(3) 放大数据使更大网络可训（TD3+BC 加宽 +11.7%）；(4) online：SAC + 每 10k 真实样本生成 1M 合成、真:合 = 0.5 混采、UTD=20，打平专门设计的 REDQ 且 wall-clock 更快；(5) 像素环境：在冻结 CNN 的 50 维 latent 里生成，V-D4RL 上 DrQ+BC +9.5%。
- 与本库关系：**概念对照的另一极**——SynthER 从同一分布采更多样本（放大器），SB 在两个分布之间搬运（翻译器）。对 SB-Render-Lite 的三个直接输入：(a) r=0.5 的真:合混采比是经 NeurIPS 验证的默认配比，与 ROSIE 的 1:1 互相印证（§3.4 沿用）；(b) latent-space 生成的做法与 SB-Render-Lite 的 latent transport 选项完全同构，证明冻结编码器 + 低维生成的可行性；(c) "多样性 vs 动力学合法性"双指标（L2-距离 vs dynamics MSE）可以平移为本库的"多样性 vs 动作有效性"评估。警示：SynthER 只在 gap≈0 的同分布内验证，**不能**引为"生成数据可跨域"的证据。

---

## 三、对照协议：扩散语义增广 vs SB transport

### 3.1 目的

在同一 sim→real 策略学习任务上回答：**给定相同生成预算与相同下游训练配比，text-guided inpainting 语义增广与 SB 视觉 transport 各自消掉 domain gap 的哪个分量、消掉多少；两者是否互补。** 这是 SB-Render-Lite 立项合理性的直接检验：若增广臂全面 ≥ transport 臂，SB-Render-Lite 需要转向（如 dynamics/trajectory bridge）。

### 3.2 gap 分解的操作化定义

把 sim→real 总 gap 拆成两个可独立操控的轴（物理/动力学差本协议不动，冻结为共同背景）：

- **语义差 S-gap**：测试场景包含训练（sim + 少量 real）未见的物体类别/外观身份、干扰物集合、背景语义；但光照、相机、材质风格保持 canonical。
- **外观差 A-gap**：测试场景语义内容与训练一致（同物体集、同布局分布，用 digital-twin 方式在 real 侧复刻 sim 场景），但承受真实域的光照/材质/传感器统计，及其扰动子集（曝光、白平衡、色温、噪声）。

理论预期（由 §1 局限表导出）：inpainting 增广主要压 S-gap 敏感度；SB transport 主要压 A-gap；都不压物理差。

### 3.3 实验臂

固定同一任务套件（建议桌面 pick-place + 放入容器，2–3 个任务即可起步）、同一策略骨干（与库内实验计划一致，BC/Diffusion Policy 二选一）、同一 sim 数据量 N_sim 与真实演示量 N_real（N_real ≪ N_sim）。

| 臂 | 内容 | 角色 |
|---|---|---|
| A0 | sim-only BC（可叠加常规 color-jitter/crop） | 下限 |
| A0′ | real-only BC（N_real 条） | 数据稀缺参考 |
| A1 | domain randomization（渲染端随机光照/纹理） | 传统基线 |
| **A2-sim** | ROSIE 配方作用于 **sim 帧**：OWL-ViT 定位 + inpainting 换物体/背景/加干扰物，动作不变、指令 relabel | 本选题主臂 |
| A2-real | 同配方作用于 N_real 条真实演示（忠实复现 ROSIE/GenAug 原设置） | 增广臂的原生形态 |
| **A3** | SB-Render-Lite：unpaired sim/real 帧训练 SB transport（像素或 frozen-latent 二选一，与 A2 同分辨率），推理时把 sim 帧整图搬到 real 风格 | 本库主方法 |
| A4 | 组合：先 A2-sim 语义增广、再 A3 transport（增广帧同样过桥） | 互补性检验 |

可选扩展臂：GenAug 式 depth-guided 变体（若任务套件有 RGBD）；CycleGAN/CUT（库内 synthesis §4.2 已列，作 transport 家族的弱基线）。

### 3.4 公平性控制（三条硬约束）

**(1) 同等生成预算。** 三种计量一并报告，避免任何一方挑对自己有利的口径：

- B_frames：生成/翻译的帧数（增广 episodes × 平均帧长 = transport 处理帧数）；
- B_nfe：去噪网络前向总次数（inpainting ≈ 步数 × 帧数 ×（base+SR 级联）；SB ≈ NFE × 帧数，注意 I²SB 类结构化先验的少步优势要如实计入）;
- B_gpu：同一硬件上的 wall-clock GPU·h（**SB 臂须把 transport 模型的训练成本摊入**，这是它相对零训练增广臂的真实劣势，不许隐藏）。

主操作点取 **B_frames 相等**；敏感性操作点取 **B_gpu 相等**（此时 SB 因训练成本可用帧数变少、或增广因级联采样变贵，两个方向都要跑）。参考锚点：ROSIE 为 4 TPUs·2h/1k episodes（Imagen Editor 推理），GenAug 约 30 s/场景。

**(2) 同等下游训练配比。** 默认 原始 : 生成 = 1:1（ROSIE 真机验证值，与 SynthER r=0.5 一致）；敏感性扫 {25%, 50%, 75%} 生成占比。所有臂共享优化器、总梯度步数、LR schedule、seed 集（≥3 seeds）。

**(3) 先验知识通道对称。** 两类方法的"外部知识"入口不同，必须显式建模：

- 增广臂的知识通道是**文本先验**（prompt + 互联网预训练权重）。设两档：
  - **P-oracle**：prompt 允许覆盖测试语义（复现 ROSIE 原设置——先补绘水槽再考水槽）；
  - **P-blind**：prompt 由 LLM 盲提议，物体清单与测试集物体做字符串/词向量双重排除。
- SB 臂的知识通道是 **unpaired real 帧**（外观先验，无任务标签）。设两档：R-full（全量可得 real 帧）与 R-k（少量 real 帧，模拟真实预算）。
- **主对比线：A2-sim(P-blind) vs A3(R-k)**——双方都被剥夺测试分布的直接信息；P-oracle 与 R-full 作为各自的上限参考单独报告。若不做这层控制，ROSIE 式评测天然高估增广臂（其原论文 prompt 就是对准评测物体人工挑的）。

### 3.5 分解评估：三套件矩阵

- **In-domain 套件**：real 域、训练见过的物体 + canonical 光照（由 N_real 同分布场景构成）——校准项。
- **A-suite（外观差）**：digital-twin 复刻 sim 训练场景的语义内容，承受真实光照/材质/传感器 + 扰动子集（±曝光、±色温、加噪）。
- **S-suite（语义差）**：canonical 光照下的未见物体/容器/干扰物/桌布（对 P-blind 臂这些物体同样是 prompt 未见的）。
- **AS-suite（双重差）**：新语义 × 新外观，考察组合臂 A4。

产出一张 方法臂 × 套件 的成功率归因矩阵，定义：

```text
Δ_app(方法) = success(A-suite) − success_A0(A-suite)
Δ_sem(方法) = success(S-suite) − success_A0(S-suite)
```

### 3.6 指标

- 主指标：各套件真机/仿真评测 downstream success rate（每格 ≥20 rollouts，报二项置信区间——修正 ROSIE 每格 8–10 次无区间的缺陷）。
- 生成质量诊断（次级，全部离线可算）：
  - **region-split FID/KID**：对 A2 产物分别在"被补绘区域"与"未编辑区域"对 real 域计算——量化"照片级补丁 + sim 底图"的混合体问题；对 A3 产物整图计算并同样分区，检查 transport 是否错误改写机械臂/手中物体（用 ROSIE 的 passthrough mask 管线）。
  - 语义保持：增广/翻译前后 OWL-ViT 检测一致性、keypoint/物体位姿一致性、mask IoU。
  - 动作有效性：inverse-dynamics 一致性（库内 synthesis §4.3 既有指标）；时序漂移：相邻帧 DINO 特征距离（暴露逐帧增广的抖动）。
  - 覆盖度：生成集的 CLIP 特征簇多样性（语义覆盖）vs 颜色直方图/功率谱统计覆盖（低层外观覆盖）——预期 A2 赢前者、A3 赢后者。
- 成本报表：B_frames / B_nfe / B_gpu 三口径 + 显存。

### 3.7 预注册假设与判读规则

- **H1**：A2-sim(P-blind) 的 Δ_sem 显著 > A3(R-k) 的 Δ_sem；
- **H2**：A3(R-k) 的 Δ_app 显著 > A2-sim(P-blind) 的 Δ_app（尤其 region-split FID 的未编辑区域分量应显示 A2 无改善）；
- **H3**：A4 在 AS-suite 上 ≥ max(A2, A3)，且增益近似可加。

判读：H1∧H2 成立 ⇒ 互补性证实，SB-Render-Lite 主管线应内置语义增广前置阶段（A4 形态），论文叙事从"SB 打败增广"改为"gap 分解 + 分工"。若 A2 在两套件都 ≥ A3（同 B_frames 与同 B_gpu 两个操作点均如此）⇒ 视觉 transport 附加值不成立，SB-Render-Lite 转向 dynamics/trajectory bridge（库内 BDGxRL、3MSBM 线）。若 A3 全面占优 ⇒ 语义增广降级为次要基线，只保留 P-oracle 上限对照。

---

## 四、venue 复核汇总（检索日期 2026-08-14）

| 论文 | arXiv | venue 结论 | 证据 |
|---|---|---|---|
| ROSIE | 2302.11550 | **RSS 2023** ✅（传言证实） | DOI 10.15607/rss.2023.xix.027；项目页标注 RSS 2023 |
| GenAug | 2302.06671 | **RSS 2023** ✅（传言证实） | roboticsproceedings.org/rss19/p010；DOI 10.15607/RSS.2023.XIX.010（Daegu, 2023-07） |
| CACTI | 2212.05711 | **CoRL 2022 PRL Workshop poster + arXiv 预印本**；未见主会/期刊发表 | OpenReview forum dRHW9-QFj9 标注 "PRL 2022 Poster"；多个 awesome-list 标 "CoRL 2022 Workshop PRL" |
| SynthER | 2303.06614 | **NeurIPS 2023** ✅（传言证实） | proceedings.neurips.cc 2023 hash 911fc798…；官方 poster 72742；GitHub README 标注 |

---

## 并入主库建议

1. **报告拆分**：本文件 §1 的 ROSIE 精读可拆出为 `reports/2302.11550_rosie_semantic_augmentation.md`（沿用库内命名与八段口径）；GenAug/CACTI/SynthER 三条以简报形式并入，或在 `reports/synthesis.md` 增设小节引用本文件。
2. **INDEX.md**：在"重要对照 / SB 方法支撑"与"SB 图像、科学数据与确定性 OT 应用"之间新增小节 **"重要对照：扩散语义增广 / 生成式数据放大"**，收 ROSIE（精读）、GenAug、CACTI、SynthER 四条，并在页尾"对 SB-Render-Lite 的直接启发"段补一句：所有视觉 transport 的收益声明都应相对 inpainting 语义增广基线报告。
3. **synthesis.md §4.2 Baseline 清单**：现有清单（sim-only BC / DR / CycleGAN / MMD / OT-UOT / SB Flow）缺增广线，补一行 "text-guided diffusion inpainting 语义增广（ROSIE/GenAug 配方，P-blind 与 P-oracle 两档）"；§4.3 指标表补 "region-split FID" 与 "S-gap/A-gap 分解成功率"。
4. **metadata/papers.tsv**：增补四行，venue 字段按 §4 表填写；CACTI 务必标注 "CoRL 2022 PRL workshop (poster), arXiv preprint"，避免后续引用错标主会。
5. **实验计划**（`../generative_policy/sb/reports/sb_render_lite_experiment_plan.md`）：采纳 §3 对照协议，重点是三条硬约束（B_frames/B_gpu 双操作点、1:1 配比默认值、P-blind/P-oracle 知识通道控制）与 S/A 双套件归因矩阵；判停规则与 §3.7 对齐。
6. **后续扩充线索**（本次未收录，留待下一轮）：ROSIE 局限段指向的 text-to-video 一致性增广（Dreamix 等）与 10× 更快的 mask-transformer 生成器（Muse）；GenAug 后续的 scaling 工作（如 GenAug 作者组后续把增广推广到 RoboAgent/MT-ACT 一线）值得跟踪，与 3MSBM 的 video bridge 路线对表。
