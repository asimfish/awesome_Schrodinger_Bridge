# E11：SimplerEnv 评测协议精读 + SB-Render-Lite 评测方案草案

## 0. 选题定位与 TL;DR

### 选题定位

本报告来自 2026-08-14 扩充批次（E11），响应前序审查 R10 的 T2 号高严重度缺口：库内对 `SB-Render-Lite`（用 SB 类方法做 sim→real 视觉翻译、以提升机器人策略真机表现）的所有方法建议都缺"真机评估统计协议"——没有试次数、种子数、置信区间规定，也没有"代理指标↔真机成功率"的验证流程。本报告精读 sim 评估与真机表现相关性的标杆工作 **SimplerEnv**（SIMPLER, arXiv 2405.05941, CoRL 2024），半精读扰动鲁棒性基准 **COLOSSEUM**（arXiv 2402.08191, RSS 2024），并把两者的协议转化为 SB-Render-Lite 可直接执行的两档评测方案（最小可行版 / 完整版）。

两篇均不在库内已有 25 篇精读之列；库内相关报告（`2509.18631` Guided OT co-training、`2409.09347` SB Flow、`2302.05872` I²SB 等）按需引用、不重复精读。

### TL;DR

1. **SimplerEnv 的核心立场：sim 评估的目标不是复刻绝对成功率，而是保持策略间相对排序**。它提出 MMRV（Mean Maximum Rank Violation）度量排序一致性——用真机性能差作为错排的权重——与 Pearson r 搭配使用；其 Visual Matching 环境在 Google Robot 任务上做到平均 MMRV 0.056 / Pearson r 0.924。
2. **视觉差距靠"绿幕合成 + 纹理烘焙"而非数字孪生解决**；控制差距靠离线 SysID（重放演示动作轨迹、模拟退火优化 PD 参数）解决。消融显示视觉匹配必须"整场景联合应用"（背景+物体+机械臂），只改一部分甚至会更差；而物理参数（质量、摩擦）不精确对排序相关性几乎无影响。
3. **离线代理指标可以完全失效**：validation action MSE 在 Bridge 任务上与真机成功率呈负相关（Pearson 低至 -1.0）。这是给 SB-Render-Lite 最重要的警告——FID/LPIPS 等图像指标在通过"与下游 success 的相关性验证"之前，不得用于选型。
4. **COLOSSEUM 给出可直接复用的 14 维扰动分类法**（MO/RO 外观与尺寸、背景 6 项、物理 2 项），并证明其中 7/14 因子的 sim 扰动退化与真机退化强相关（R² 0.74–0.94）；但组合扰动的相关性不稳定（某组合 R²=0.01），且其协议只用单训练种子、每测试集 25 episodes——统计上恰是我们要避免的下限。
5. 评测方案草案落地为"四层指标 + 统计功效基础 + 两档协议"：L1 翻译图像质量 → L2 下游 policy success（王指标）→ L3 sim-real 相关性验证（MMRV+Pearson 门槛）→ L4 扰动鲁棒性（借 COLOSSEUM 分类法做"对齐 vs 增广"签名检验）；真机功效计算表明检测 15pp 效应需约 170 rollouts/臂，因此协议把大规模筛选全部放进 sim，真机只做加大 rollout 的终检。

---

## 1. 精读：SimplerEnv — Evaluating Real-World Robot Manipulation Policies in Simulation

### 1.1 元信息（venue 复核日期：2026-08-14）

- 论文：Evaluating Real-World Robot Manipulation Policies in Simulation（系统名 SIMPLER：Simulated Manipulation Policy Evaluation for Real Robot Setups）
- 作者：Xuanlin Li*, Kyle Hsu*, Jiayuan Gu*, Karl Pertsch, Oier Mees, Homer Rich Walke, Chuyuan Fu, Ishikaa Lunawat, Isabel Sieh, Sean Kirmani, Sergey Levine, Jiajun Wu, Chelsea Finn, Hao Su, Quan Vuong, Ted Xiao（UCSD / Stanford / UC Berkeley / Google DeepMind）
- arXiv：2405.05941（v1 2024-05-09）；链接 https://arxiv.org/abs/2405.05941
- **发表状态（web 复核）：CoRL 2024（8th Conference on Robot Learning, Munich）正式发表；PMLR v270:3705–3728（2025 年出版）**。复核来源：OpenReview（forum id LZh48DTg71，标注 CoRL 2024）与 proceedings.mlr.press/v270/li25c.html。注意 camera-ready 摘要与 arXiv v1 略有差异：CoRL 版明确"over 1500 paired sim-and-real evaluations across two embodiments and eight task families"。
- 项目页 / 代码：https://simpler-env.github.io ；https://github.com/simpler-env/SimplerEnv
- 全文获取：arXiv HTML 全文（含附录 A–G）已获取并通读，无缺失段落。

### 1.2 动机与问题定义

真机评估贵、慢、难复现，且随 generalist policy 能力谱变宽，评估负担线性增长。作者主张 real-to-sim 评估：把在真实数据上训练的策略放进专门构建的仿真环境里评。这与 sim-to-real 训练方向相反，也与"数字孪生"路线不同——论文的关键立场是**不需要精确复刻真实环境，只需要"足够真实以保持策略间相对性能排序"**。

形式化：对策略 \(\pi_a, \pi_b\) 及其真机性能 \(R_a, R_b\)，目标是构建仿真 \(\mathcal{S}\)，使 sim 性能 \(R_{\mathcal{S},a}, R_{\mathcal{S},b}\) 与真机性能的**相对关系**强相关。sim 评估定位为研发迭代的 proxy 信号，不取代最终真机验证。

### 1.3 方法核心

#### 1.3.1 相关性度量：Pearson r 的缺陷与 MMRV 定义（本报告最重要的可移植资产）

Pearson r 作为唯一度量有两个缺陷：(a) 只衡量线性拟合——但 sim 评估只需保序，不需线性；(b) 对取值范围不敏感——当各策略真机性能彼此接近时，r 会被评估噪声大幅扰动。常规秩相关（Spearman）也不够：它忽略错排策略之间的真实性能差距，把"错排两个性能差 1% 的策略"和"错排两个性能差 40% 的策略"同罚。

因此论文提出 **MMRV（Mean Maximum Rank Violation，范围 [0,1]，越低越好）**。给定 N 个策略的真机性能 \(R_{1..N}\) 与 sim 性能 \(R_{\mathcal{S},1..N}\)：

\[
\textrm{RankViolation}(i,j) = |R_i - R_j| \cdot \mathbf{1}\big[(R_{\mathcal{S},i} < R_{\mathcal{S},j}) \neq (R_i < R_j)\big]
\]

\[
\textrm{MMRV}(R, R_{\mathcal{S}}) = \frac{1}{N}\sum_{i=1}^{N} \max_{1 \le j \le N} \textrm{RankViolation}(i,j)
\]

即：每对策略若被 sim 错排，罚分等于二者**真机**性能差；对每个策略取其最坏错排，再对策略平均。这样小噪声导致的近邻错排几乎不罚，真正的排序失败重罚。论文全程 MMRV 与 Pearson r 并报。附录 G-B 还补充了 Kruskal-Wallis 检验：对每个策略检验 sim 与真机的逐 trial 成功指示分布是否显著偏移（统计绝对差距，即使不追求绝对复刻）。

#### 1.3.2 控制差距：离线 System Identification

目标：同一动作序列在 sim 中开环重放时，末端 6D 位姿轨迹与真机吻合。从**已有开源演示数据集**（RT-1、Bridge V2，无需采新数据）抽取动作与末端位姿轨迹，定义损失：

- 平移损失 \(\mathcal{L}_{\textrm{transl}} = \frac{1}{T}\sum_i \lVert \mathbf{x}_i - \mathbf{x}'_i \rVert_2\)；
- 旋转损失 \(\mathcal{L}_{\textrm{rot}} = \frac{1}{T}\sum_i \arcsin\big(\tfrac{1}{2\sqrt{2}}\lVert R_i - R'_i \rVert_F\big)\)；
- \(\mathcal{L}_{\textrm{sysid}} = \mathcal{L}_{\textrm{transl}} + \mathcal{L}_{\textrm{rot}}\)。

对控制器 stiffness/damping（PD）参数做 3 轮模拟退火、逐轮收缩搜索范围。直接用真实控制器 PD 值反而跟踪失败（重放抓可乐罐 miss grasp），SysID 后成功复现。控制器实现细节：Google Robot 用 Ruckig 做带速度/加速度/jerk 约束的时间最优关节规划，sim 频率 501 Hz、控制频率 3 Hz；WidowX 用 IK 直接设定关节目标，sim 500 Hz、控制 5 Hz。

#### 1.3.3 视觉差距：Visual Matching（绿幕 + 纹理烘焙）与 Variant Aggregation

**Visual Matching（推荐默认）**由两部分组成：

1. **Green-screening**：取真机评估视频首帧 \(I_{\textrm{real}}\)，用图像 inpainting 工具抹去机器人与前景物体得到真实背景；在 sim 中查询 ground-truth 分割掩码 \(M\)（机械臂 + 可交互物体）；合成 \(I' = M \odot I_{\textrm{sim}} + (1-M) \odot I_{\textrm{real}}\)。
2. **Texture matching（纹理烘焙）**：对视觉差距明显的物体，(a) SAM 分割真实图像中的目标物体；(b) 粗对齐 sim 资产位姿使分割掩码重叠；(c) 用可微渲染（Nvdiffrast）精化位姿；(d) 把真实 RGB "unproject" 到 sim mesh；(e) 可选用 Zero123++ 生成剩余视角补全纹理。半自动化，命令行脚本已开源（GeTex）。对纹理图已接近真实的资产（如机械臂），直接在贴图上复制粘贴真实颜色值。另外因真机机械臂在运动中颜色观感会变，制作多套微调臂色并**对评估结果取平均**以消除该混淆因子。

**Variant Aggregation（对照方案）**：不缩小差距而是随机化——基础环境沿背景/光照/干扰物/桌面纹理四轴各做 2 个变体（源自 ReplicaCAD 场景），对所有变体的评估结果取平均。实验证明它不如 Visual Matching（见 1.4），因为对视觉分布偏移敏感的策略在大偏移下崩溃。

资产管线要点：常见物体来自 Objaverse 或 3D 扫描 / 单视图重建（One-2-3-45++）；CoACD 生成凸碰撞体；密度用材料常识（GPT-4/搜索）设定；铰接物体（柜子）手工建模——这是全流程人工成本最高处。仿真基于 SAPIEN（消费级 RTX 4090 上 640×512 达 3.5k sim steps/s，相对真机评估约 7× 加速），并在 Isaac Sim 上复现了全部结论（模拟器无关性）。

#### 1.3.4 评估任务集与 trial 结构（协议细节，直接可抄）

真机与 sim 严格配对评估，初始状态用**确定性网格**而非随机采样：

- Google Robot（RT 系列设置）：
  - "pick coke can"：可乐罐 3 种朝向（横躺/竖躺/立放）× 桌面矩形内 25 个网格位置 = **75 trials**；
  - "move {obj1} near {obj2}"：8 物体中选 5 个三元组 × 2 种三角摆位 × 每组 6 种源/目标/干扰角色分配 = **60 trials**；
  - "open/close top/middle/bottom drawer"：机器人在地面矩形 9 个网格位 × 3 抽屉 × 开/关 = **54 trials**；
  - 长时程 "open top drawer; place apple into top drawer"：3 机器人位 × 9 苹果网格位 = **27 trials**（中途按策略输出 terminate 或时限过半切换指令）。
- WidowX + Bridge V2：spoon-on-towel / carrot-on-plate / stack-block / eggplant-in-basket 各 **24 trials**（正方形顶点摆位 + 朝向或尺寸二态），并报告部分成功（如 grasp 率）。
- sim 侧 trial 数 = 真机 trial 数 × 环境变体数 ×（Google 臂色 4 套）×（Octo 类扩散头策略 3 个推理种子）。

被评策略集覆盖强弱锚点：RT-1（Converged / 15% steps / Begin 三个训练阶段）、RT-1-X、RT-2-X、Octo-Base/Small——刻意用训练阶段拉开真机性能范围，使相关性度量有效。

### 1.4 关键结果

- **主结果（Google Robot，6 个 checkpoint）**：Visual Matching 平均 MMRV **0.056**、Pearson r **0.924**；Variant Aggregation 0.143 / 0.778；作为对照的 validation action MSE 0.375 / 0.308。Bridge 任务上 Visual Matching 各任务 MMRV 全 0（仅一处 0.111），成功率 Pearson r 0.575–1.000。
- **代理指标失效证据（Table XII）**：validation MSE 在 Bridge 四个任务上与真机成功率 Pearson 为 **-0.951 / -0.342 / -0.857 / -1.000**（负相关！）；sim 评估同任务为 0.827 / 0.575 / 1.000 / 0.990。
- **分布偏移行为复现（Table VI）**：沿背景/光照/干扰物/桌面纹理/相机位姿五轴做受控偏移，\(\Delta\text{Success}(\text{shift}) = \frac{1}{2}\sum_{k=1}^{2}(\text{Success}_{\text{shift},k} - \text{Success}_{\text{base}})\)；sim 准确复现两个 RT-1 变体（带/不带增广训练）的真机敏感性排序（MMRV 0.000/0.041，r 0.831/0.970）。细粒度到"纯色桌面 vs 花纹桌面"：真机退化 4% vs 25%，sim 2% vs 24%。
- **新分布偏移预测（Table VIII）**：sim 中发现 Octo-Base 对臂纹理极敏感（成功率区间 [0.000, 0.293]），RT-1-X 稳健（[0.507, 0.653]）；据此设计真机新实验——用礼品包装纸缠机械臂——真机复现该预测（Octo-Base 0.293→0.000，RT-1-X 0.760→0.520）。
- **消融**：(a) SysID 有效（control loss 0.131 vs 0.267/0.432，MMRV 0.031 vs 0.070/0.100）；(b) 视觉匹配必须整体应用——绿幕+抽屉纹理+机械臂纹理三者齐上才降到 MMRV 0.050 / real-sim 绝对差 0.136，只做其中一两项与不做无异甚至更差（作者假设：场景各部分外观不一致本身就是分布偏移）；(c) **物理参数鲁棒**：可乐罐质量 10–80g × 夹爪摩擦 0.25–2.0 全组合下 MMRV 恒 0.031、r 0.957–0.990；柜门关节摩擦 6 档同样稳健——即绝对成功率可变 ≤15%，但排序不变；(d) Isaac Sim 复现（pick coke can MMRV 0.064 / r 0.973）。
- 单任务小数据策略（只用 pick-coke-can 演示训练的 RT-1）加入后 MMRV 仍 0.027 / r 0.959，说明协议对小数据策略同样有效。

### 1.5 结论

sim 评估可以成为真机评估的可靠、可复现、可扩展 proxy——前提是 (i) 用离线 SysID 关控制差距，(ii) 用整场景联合的 Visual Matching 关视觉差距，(iii) 用 MMRV+Pearson 这样"保序 + 保裕度"的度量验证 pipeline 本身。物理参数精确性对排序目标不关键，视觉一致性与控制一致性关键。

### 1.6 局限

- 仅覆盖刚体操作任务；软体/流体/布料超出当前物理仿真置信范围。
- Green-screening 只支持固定相机，不能表现物体投影/阴影等细节。
- 资产制作仍需人工（铰接物体最贵）；尚非全自动 pipeline。
- MMRV/Pearson 是对"策略集合"定义的：点数少（Bridge 侧仅 3 个策略）或性能范围窄时统计意义弱；论文靠训练阶段锚点缓解，这一技巧必须随协议一起移植。
- 论文自身未报告 sim 评估的多种子方差（仅对扩散头策略平均 3 个推理种子）。

### 1.7 对 SB-Render-Lite 的直接启示

1. SB-Render-Lite 的最终主张是"翻译提升真机成功率"，验证这一主张的评估基础设施本身要先通过 MMRV/Pearson 验证——**先花小预算验证 sim 评估器，再用它做大规模筛选**，这是 SimplerEnv 的方法论核心。
2. 我们的"策略集合"= 不同翻译变体 + 基线共训练出的一组 policy checkpoint；要刻意纳入弱锚点（sim-only BC 早期 checkpoint）拉开范围，否则相关性度量无效。
3. Visual Matching 的绿幕 + 纹理烘焙本身就是一种"确定性 sim→real 渲染对齐"，可以直接充当 SB-Render-Lite 的**非学习对照基线**：如果 SB 翻译打不过绿幕合成，图像级生成翻译的价值不成立。
4. validation MSE 的负相关教训 → 库内 synthesis §4.3 的辅助指标清单（DINO/CLIP、keypoint 一致性等）在未通过相关性验证前一律只做记录、不做决策。

---

## 2. 半精读：COLOSSEUM — A Benchmark for Evaluating Generalization for Robotic Manipulation

### 2.1 元信息（venue 复核日期：2026-08-14）

- 论文：THE COLOSSEUM: A Benchmark for Evaluating Generalization for Robotic Manipulation
- 作者：Wilbert Pumacay*, Ishika Singh*, Jiafei Duan*, Ranjay Krishna, Jesse Thomason, Dieter Fox（UCSP / USC / UW / AI2 / NVIDIA）
- arXiv：2402.08191（v1 2024-02-13）；链接 https://arxiv.org/abs/2402.08191
- **发表状态（web 复核）：RSS 2024（Robotics: Science and Systems XX, Delft, 2024-07）正式发表，DOI 10.15607/RSS.2024.XX.133**。复核来源：roboticsproceedings.org/rss20/p133.html（官方 BibTeX `Pumacay-RSS-24`）与 NVIDIA SRL 出版页。GitHub README 的 BibTeX 仍写 arXiv preprint，引用时应以 RSS 条目为准。
- 项目页 / 代码：https://robot-colosseum.github.io ；https://github.com/robot-colosseum/robot-colosseum
- 全文获取：arXiv HTML 全文（含附录）已获取，半精读聚焦扰动分类法与 sim-real 相关性章节。

### 2.2 扰动维度分类法（本次借用重点）

COLOSSEUM 在 RLBench 20 个任务上实现 **14 个扰动因子**，形式化为 covariate shift：\(p(x_{test}) \neq p(x_{train})\) 而 \(p(y|x)\) 不变（任务本身不变）。分类法四大类：

| 类别 | 因子 | 实现参数（借用时的默认值参考） |
|---|---|---|
| Manipulation Object（被直接操作的任务相关物体） | MO_Color / MO_Texture / MO_Size | 颜色 20 色离散集；纹理 213 张离散集；尺寸连续缩放，范围任务相关（如 basketball [0.75,1.25]，hockey [0.95,1.05]），waypoint 随缩放重定位 |
| Receiver Object(任务相关但不被直接操作，如酒架) | RO_Color / RO_Texture / RO_Size | 同上；无 RO 或复合形状（PyRep 限制）的任务不适用 |
| Background（与任务物体无关的场景属性） | Light_Color / Table_Color / Table_Texture / Distractor / Background_Texture / Camera_Pose | 光照 RGB 在 [0,0,0]–[0.5,0.5,0.5] 采样并应用于 3 个方向光；干扰物从 78 个 YCB 模型采样；相机位置扰动 ±0.1、姿态（欧拉角）±0.05，作用于 front/left-shoulder/right-shoulder 三相机 |
| Physical | Object_Friction / Object_Mass | 摩擦系数 [0.75,1.0]；质量范围任务相关（如 slide block [1.0,15.0]） |

任务按 horizon（waypoint 数）分 simple/intermediate/complex 三档，共 20,371 个唯一任务实例。因子可单独或组合施加（YAML 配置）。

### 2.3 评估协议与 sim-real 相关性结果

- **协议**：训练用每任务 100 条无扰动演示（保留 RLBench 语言/目标变体）；测试为 14 因子 + No Perturbation + All Perturbations，共 235 个测试集，**每测试集固定 25 episodes**；排行榜按相对 No Perturbation 的成功率百分比变化排名。**只用 1 个训练种子、1 个评估种子**。
- **主发现**（5 个基线：R3M-MLP、MVP-MLP、PerAct、RVT、VoxPoser）：单因子使成功率退化 30–50%；全因子叠加退化 ≥75%；影响最大的因子是干扰物数量、目标物体颜色、光照；物体尺寸影响最小。3D 方法（PerAct/RVT）整体优于 2D 且更鲁棒；体素/重渲染类方法对 Camera_Pose 鲁棒而 2D RGB 方法脆弱。
- **训练侧消融**：即便把 All Perturbations 加进训练数据，也只挽回 +21.1%（zero-shot 掉 28.1%）——扰动不仅是分布偏移，也实质加大任务难度。
- **sim-real 相关性**：4 个任务 3D 打印复刻（insert onto square peg / slide block to target / scoop with spatula / setup chess），Franka Panda，真机 PerAct 用每任务 5 条演示训练；每因子 10 episodes × 3 runs。总体 \(\bar{R}^2 = 0.614\)；**7/14 因子 R² 0.74–0.94**（Background_Texture、Distractor、Table_Color、Light_Color、RO_Color、RO_Texture、RO_Size，其中 Table_Color 最高），MO_Color / Table_Texture / Camera_Pose 中等（0.46–0.52）。
- **组合扰动落地检验**：用三个真实场景（工作台/餐桌/书房桌）对照 sim 组合扰动：[Distractor+MO_Size] R²=0.75，[Light+Table_Texture+Distractor+MO_Size] R²=0.83，**但 [Distractor+Light_Color] R²=0.01**——组合扰动的 sim-real 相关性不保证，逐因子验证不可省。

### 2.4 局限与借用注意事项

- 单训练种子 + 单评估种子 + 每测试集 25 episodes：读它的任意两格数字差异都可能在噪声内（25 次试验 95% Wilson 区间半宽约 ±18pp，见 §3.2）。借用其分类法，不借用其统计规格。
- 真机侧每因子只有 2 个替代变体（3D 打印成本），结论是"分布对比"而非逐点匹配。
- 只评了 BC/keypoint 类方法；对连续控制/扩散策略的因子敏感性排序未必相同（SimplerEnv 中 RT-1 对光照/干扰物不敏感、对相机位姿最敏感，与 COLOSSEUM 的"干扰物/颜色/光照最伤"并不一致——**因子敏感性排序依赖模型类别与训练数据**，评测设计要覆盖因子全集而非只挑"最伤"的因子）。

---

## 3. SB-Render-Lite 评测方案草案

### 3.0 设计原则（从两篇标杆提炼）

1. **success rate 为王，但入口是相关性验证**：所有图像/特征代理指标（L1）须先对下游 success（L2）通过相关性验证（L3 的方法论用于 L1 验证），才可用于选型与早停（响应 R10 T2-(2)；证据：SimplerEnv Table XII）。
2. **筛选在 sim，终检在真机**：真机功效昂贵（§3.2），大规模消融/扫参放在已通过 MMRV/Pearson 验证的 sim 评估器中；真机只对 top 候选做加大 rollout 的确认。
3. **确定性初始状态网格**：所有评估（sim 与真机）使用 SimplerEnv 式初始状态网格而非随机摆位——降方差，且允许逐 trial 配对比较（McNemar）。
4. **因子化扰动 + 签名检验**：借 COLOSSEUM 分类法把"鲁棒性"分解到因子层；SB 翻译的预期签名是"外观类因子退化显著收窄、几何类因子（Camera_Pose、Size）退化不变"——这正是区分"学到了 sim→real 分布对齐"与"等价于随机增广正则化"（R10 T4）的可检验证据。
5. **固定 policy 配方 + compute-matched**：所有翻译变体之间冻结同一 BC/policy 训练配方与超参；各翻译方法参数量与训练预算声明并尽量对齐（R10 T2-(5)(6)）。

### 3.1 四层指标体系

**L1：翻译质量图像指标（只记录，不先验决策）**

- 分布级：FID（注意小样本偏差，样本 <10k 帧时并报 **KID**（无偏估计）；帧间自相关高，抽帧去相关后再计算）；可选 CMMD。
- 成对级（仅在有 GT 配对处适用）：LPIPS、PSNR/SSIM——**sim-to-sim v0（双渲染配置）天然有逐像素配对**，是这些指标唯一可信的场景；真 sim2real 无配对，不硬算。
- 结构/语义保持（对接 synthesis §4.3 既有清单）：翻译前后 DINO/CLIP 特征余弦；sim GT 分割掩码 vs 翻译图重分割（SAM）的 mIoU；keypoint/物体位姿重投影误差；深度一致性；inverse dynamics 一致性。
- 时序（若做视频/序列翻译）：相邻帧翻译的 flicker（LPIPS(t,t+1) 相对原始序列的膨胀率）。

**L2：下游 policy success（王指标）**

- 协议：固定 BC 配方；每翻译变体训练 ≥3 个种子；随机推理头（扩散/SB 采样）另加 ≥3 个推理种子并平均（SimplerEnv 对 Octo 的做法）。
- 双栏报告 ID / OOD：OOD 用 only-seen-in-sim 场景（库内 `2509.18631` 的做法），即某些物体摆位/背景组合只在 sim 训练数据中出现、真机测试时首次出现。
- 除 success 外单列 safety（碰撞次数/工作空间违例，R10 T2-(4)）与部分成功（grasp 率，SimplerEnv Bridge 表的做法——低成功率区间的分辨率来自部分成功）。

**L3：sim-real 相关性验证（评估器资格考试）**

- 点集构造：≥6 个方法点（建议：sim-only BC、domain randomization、绿幕合成对照（§1.7-3）、CycleGAN/CUT、SB Flow 翻译、SB-Render-Lite 主方法），再加 1–2 个弱锚点（sim-only BC 的早期 checkpoint），拉开真机性能范围。
- 对该点集做配对 sim & real 评估，计算 **MMRV + Pearson r**（Spearman 作参考）；可选 Kruskal-Wallis 检查绝对分布偏移。
- 用途 (a)：验证我们的 sim 评估环境（若自建 SIMPLER 式 visual-matching 环境）；用途 (b)：验证每个 L1 代理指标——把指标值当"sim 性能"代入 MMRV/Pearson，与真实（或 sim 评估）success 对照；|Spearman| ≥ 0.7 且 MMRV ≤ 0.15 的指标才获准进入选型流程。
- **门槛建议**：sim 评估器达到 MMRV ≤ 0.10 且 Pearson r ≥ 0.85（任务平均）即可作为筛选工具（SimplerEnv Visual Matching 实测 0.056/0.924，Variant Aggregation 0.143/0.778 已明显不够用）。

**L4：扰动鲁棒性（借 COLOSSEUM 分类法）**

- MVP 因子子集（5 个，取两篇论文"最伤因子"的并集）：MO_Color、Table_Texture、Light_Color、Distractor、Camera_Pose；完整版扩展到 MO_Texture、MO_Size、RO_Color/Texture/Size、Background_Texture（物理两项对视觉翻译假设无关，列为可选）。
- 每因子 ≥2 个变体，报告 SimplerEnv 式 \(\Delta\text{Success}\)（式见 §1.4），并做 All-Perturbations 组合档。
- **签名检验（主假设 H-robust）**：相对 sim-only 基线，SB 翻译使外观因子（MO_Color/Table_Texture/Light_Color/Distractor）的 \(|\Delta\text{Success}|\) 显著收窄，而 Camera_Pose 的 \(|\Delta\text{Success}|\) 无显著变化；同时"增广强度匹配对照"（R10 T4：与 SB 输出同等像素/latent 位移幅度的随机风格增广）不产生同等收窄。两条都成立才能归因于分布对齐。
- 注意：组合扰动的 sim-real 相关性不可外推（COLOSSEUM R²=0.01 的反例），组合档只在 sim 内比较、不用于真机结论。

### 3.2 统计功效基础（回答 R10 T2-(1)）

二项成功率的基本量（95% Wilson 区间半宽，p̂=0.5 最坏情形）：

| 每配置 rollouts n | 25 | 50 | 100 | 200 |
|---|---|---|---|---|
| CI 半宽 | ±18pp | ±13pp | ±10pp | ±7pp |

两方法独立比较（双侧 α=0.05，power 0.8，两比例 z 检验）所需**每臂**样本量：

| 预期效应量 Δ | 30pp（如 0.35→0.65） | 20pp（0.4→0.6） | 15pp（0.5→0.65） |
|---|---|---|---|
| 每臂 rollouts | ≈45 | ≈100 | ≈170 |

推论与规定：

1. 参考效应量：库内 Guided OT co-training 报告真机平均提升约 30pp 量级，COLOSSEUM 因子退化 30–50pp；但 SB-Render-Lite 相对强基线（OT 共训练）的**边际**效应可能只有 10–20pp——独立比较在真机上养不起，因此：
2. **真机采用配对设计**：同一初始状态网格上逐 trial 配对（同网格点、同物体摆位跑两个方法），用 McNemar 检验只看不一致对，功效显著高于独立比较；每任务网格 ≥25 点（SimplerEnv 规格），关键结论任务 ≥50 点。
3. **任务聚合**：主结论以任务平均 success 报告（≥3–4 个任务），聚合置信区间用任务分层 bootstrap；单任务数字必须带 Wilson 区间。
4. **种子**：训练种子 ≥3（完整版 ≥5），种子间标准差与均值并报；任何"方法 A 优于 B"的声明需在种子平均意义上成立且区间不交叠（或配对检验 p<0.05）。
5. **多重比较**：sim 大规模扫描（十数个变体 × 因子）用 Benjamini-Hochberg 控 FDR，或者只做秩选择、把假设检验留给真机终检的 top-2。
6. 每测试配置 25 episodes（COLOSSEUM 规格）只够看 ≥35pp 的因子效应，凡引用其数字须带此保留。

### 3.3 两档协议

#### 档位 A：最小可行版（MVP，纯 sim 起步，约 2–3 周 + 1 周真机窗口）

適用：验证 SB-Render-Lite 第一个 head-to-head 假设（R10 T1：纯 OT 共训练 vs 图像级 SB 翻译）。

1. **sim-to-sim v0 代理环境**：同一模拟器（SAPIEN/ManiSkill 或 Isaac）两套渲染配置——"源渲染"（低保真：平光、默认材质）与"目标渲染"（高保真：光追/烘焙纹理/真实背景贴图）——冒充 sim/real 双域。**GT 逐像素配对存在**，L1 全指标可信；"真机"评估 = 目标渲染域内 rollout，无限便宜。
2. 任务：3–4 个刚体操作任务（建议直接用 SimplerEnv 任务的 ManiSkill 实现：pick-can、move-near、drawer + 1 个 Bridge 式放置任务），每任务 25 点初始状态网格。
3. 方法点集（≥6 + 2 弱锚点）：sim-only BC / domain randomization / 绿幕合成对照 / CycleGAN 或 CUT / SB Flow / SB-Render-Lite；外加增广强度匹配对照（进 L4 签名检验）。
4. 预算：L2 每方法每任务 100 rollouts（目标渲染域）× 3 训练种子；L4 五因子 × 2 变体 × 25 episodes（因子扰动实现直接搬 COLOSSEUM 参数表，§2.2）。
5. L3（v0 内部版）：以"源渲染域评估"预测"目标渲染域评估"，算一次 MMRV/Pearson，练手并校准指标实现；同时产出 L1↔L2 代理指标相关性矩阵，筛出获准指标。
6. **真机终检（最小）**：top-2 方法 + sim-only + 最强非 SB 基线共 4 臂；2–3 个任务 × 每任务 25 个配对网格点；McNemar + 任务聚合 Wilson 区间。此预算只能确认 ≥25–30pp 的效应；若 sim 中边际效应 <15pp，如实报告"真机不可判定，仅 sim 证据"并停止扩大真机投入。
7. 产出物：方法点集的 (sim success, real success) 散点 + MMRV/Pearson；L4 因子退化条形图（含签名检验结论）；代理指标相关性矩阵。

#### 档位 B：完整版（需自建评估环境与更多真机窗口）

在 MVP 全部通过后启动：

1. **自建 SIMPLER 式 visual-matching 评估环境**（针对我们自己的机器人/相机设置）：按 §1.3.2–1.3.3 复刻——离线 SysID（重放自采演示，3 轮模拟退火）、绿幕合成（inpaint 背景 + GT 掩码）、纹理烘焙（GeTex 脚本）、多套臂色平均。验收：用 MVP 的方法点集跑 MMRV ≤ 0.10 / Pearson ≥ 0.85，通过后该环境成为默认筛选器。
2. 任务扩到 ≥5（含 1 个长时程、1 个 only-seen-in-sim OOD 场景）；方法点集扩到 ≥8（加入 I²SB 配对上限对照与 GSBM 变体，引库内 `2302.05872`、`2310.02233` 报告）。
3. L4 扩到 10+ 因子（COLOSSEUM 全外观因子），sim 全量扫描 + 真机抽查 3–4 个因子（每因子 2 变体 × 10 episodes × 3 runs，即 COLOSSEUM 真机规格），报告逐因子 sim-real R²；组合扰动只在 sim 报告。
4. 统计规格升级：真机每任务每臂 ≥50 配对 rollouts（关键对比 ≥2 任务 ×50，聚合后可判 ~15pp）；训练种子 ≥5；预注册主假设（H-main：SB 翻译 vs OT 共训练的任务平均 success 差；H-robust：L4 签名），其余分析标注为探索性。
5. 全程记录 compute-matched 表（各方法参数量/训练 GPU 时/翻译 NFE 与 wall-clock），对接 synthesis §4.3 的效率指标行；若未来考虑部署期 real→sim 翻译（R10 T3），NFE 须换算控制频率延迟预算。

### 3.4 验收门槛与决策规则（汇总）

| 检查点 | 门槛 | 不达标动作 |
|---|---|---|
| sim 评估器资格（L3） | MMRV ≤ 0.10 且 Pearson r ≥ 0.85（任务平均，≥6 方法点） | 修 visual matching / SysID，参照 §1.4 消融顺序（先整场景视觉一致性，后控制，物理参数最后） |
| 代理指标资格（L1→L2） | \|Spearman\| ≥ 0.7 且 MMRV ≤ 0.15（对方法点集） | 该指标降级为"仅记录"，不得用于早停/选型 |
| 主假设 H-main | 任务平均 success 提升，配对检验 p<0.05 且 ≥3 种子方向一致 | 如实报告；检查是否被增广对照解释 |
| 鲁棒性签名 H-robust | 外观因子退化收窄显著、Camera_Pose 不变、增广对照不复现 | 若增广对照复现同等收窄 → 结论降级为"正则化效应"，SB 对齐主张不成立 |
| 真机终检 | 每臂聚合 ≥75（MVP）/ ≥100（完整版）配对 rollouts，Wilson 区间 + McNemar | 效应 <可判阈值时明确写"真机不可判定"，禁止用点估计差做结论 |

---

## 4. 并入主库建议

1. **INDEX.md**：在"核心：仿真 / 真机迁移"区新增两条目（SimplerEnv、COLOSSEUM，均为评测方法学而非迁移方法，建议单开"评测协议与基准"小节），并把本报告挂在末节"对当前 SB-Render-Lite 的直接启发"之后作为评测协议入口。
2. **落实 R10 建议 #3**：本报告 §3 可直接充当 R10 要求的"评估协议附录"底稿——六件套逐项已覆盖（统计协议 §3.2、代理指标验证 §3.1-L3、OOD 拆分 §3.1-L2、safety 单列 §3.1-L2、compute-matched §3.3-B5、policy 配方固定 §3.0-5），另补了 R10 未明确的配对设计（McNemar）与多重比较（FDR）两项。建议主库把 §3.2 的两张功效表与 §3.4 门槛表原样收编为 canonical。
3. **metadata/papers.tsv**：新增两行——2405.05941（venue=CoRL 2024, PMLR v270:3705–3728, last_verified=2026-08-14, code=github.com/simpler-env/SimplerEnv）；2402.08191（venue=RSS 2024, DOI 10.15607/RSS.2024.XX.133, last_verified=2026-08-14, code=github.com/robot-colosseum/robot-colosseum）。注意 COLOSSEUM 官方 GitHub BibTeX 仍是 arXiv 条目，以 RSS proceedings 为准。
4. **synthesis.md §4.2/§4.3 衔接**：基线清单建议补"绿幕合成对照"（非学习翻译上限，成本几乎为零）与"增广强度匹配对照"（R10 T4）；指标清单头部加一句"所有辅助指标须先通过 L3 式相关性验证"（引 SimplerEnv Table XII 的负相关证据）。
5. **后续精读候选**（本报告引用但未精读，不占本次名额）：Xie et al. 2023 (arXiv 2307.03659, Decomposing the Generalization Gap)——SimplerEnv 分布偏移轴的来源，若要细化 L4 因子设计值得补读；AutoEval / 后续 real-to-sim 评估自动化工作可在下一批检索确认。
6. 风险提示随库：SimplerEnv 的结论建立在刚体任务 + 固定相机上；SB-Render-Lite 若扩展到 contact-rich 或腕部相机设置，L3 资格考试必须重跑，不能沿用旧验证结论。
