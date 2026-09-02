# E08：Domain Randomization + GAN 翻译经典基线 —— RCAN / RetinaGAN 精读与 SB-Render-Lite 基线协议规格

> 扩充研究员：E08 ｜ 日期：2026-08-14 ｜ 选题来源：内部审查 R09 缺口 **G2**（视觉 domain randomization 与 GAN 翻译经典基线线，P0 优先级）
> 任务：精读 RCAN（arXiv 1812.07252）与 RetinaGAN（arXiv 2011.03148）；Tobin et al. DR（1703.06907）、GraspGAN（1709.07857）、CycleGAN（1703.10593）、CUT（2007.15651）做收录条目；产出 SB-Render-Lite 实验的"经典基线协议规格清单"。
> 全文获取方式：六篇均经 arXiv 官方 HTML 全文精读/通读（arXiv abs 页现直接返回 HTML 全文，含附录与超参表）；venue 信息全部经 web 独立复核（检索日期 2026-08-14）。非二手转述。

## 选题定位

`SB-Render-Lite` 的主张是"用 Schrödinger Bridge 类 transport 做 unpaired sim→real 视觉迁移，以下游真实域 policy success 为最终指标"。这条主张天然站在两条经典路线的延长线上：**(a) domain randomization（DR）**——不翻译图像，用随机化覆盖真实域；**(b) GAN 图像翻译**——用对抗训练学 sim↔real 映射。`synthesis.md` §4.2 早已把 DR 与 CycleGAN/CUT 列为必比 baseline，`generative_policy/sb/reports/sb_render_lite_experiment_plan.md` 的对照组 B1（DR）与 B2（CycleGAN/CUT）也已定死，但库内此前没有任何一篇对应论文的正式条目。reviewer 必问的三个问题——"和 DR 比怎样？和 GAN 翻译比怎样？你的语义保持约束相对 RetinaGAN 的感知一致性新在哪？"——全部落在本报告覆盖的六篇上。本报告把这条线的**方法机制、真实数据需求、真机评估协议、失败模式**钉死成可引用的事实，并把它们转译成 SB-Render-Lite 可直接执行的对照实验协议。

## TL;DR

1. **RCAN（CVPR 2019，注意：不是 CoRL）**：用 pix2pix 式 cGAN 学"重随机化 sim → canonical sim"翻译，配对数据由仿真免费生成，全程 0 真实数据；部署时把真实图像也拉回 canonical 域。QT-Opt 抓取 zero-shot 真机成功率 **70%，约为直接在 DR 上训 policy（33–37%）的两倍**；+5,000 真实 on-policy 抓取联合微调后 91%，超过用 580,000 真实抓取训练的 QT-Opt（87%），真实数据缩减 >99%。核心洞见：**DR 的正确用法是"喂翻译器"而不是"直接喂 policy"**。
2. **RetinaGAN（ICRA 2021）**：CycleGAN + 冻结 EfficientDet 的 perception consistency loss（对翻译前后 6 张图的检测框/类别做 Huber + Focal Consistency 一致性）。纯 sim-to-real 数据训 policy：**裸 CycleGAN 67.8% → RL-CycleGAN 68.9% → RetinaGAN 80.0%（+12 pp，>2σ）**——"语义/感知一致性约束"是 GAN sim2real 路线的胜负手，与 SB-Render-Lite 的 geometry/temporal 约束、GSBM 的 task-aware path cost 动机同构，是必须正面引用与对比的最直接前驱。同一 GAN 零新增数据复用到 pushing 任务 0%→90%；低数据（10k episodes）下仍有效但单 GAN 语义漂移需 3-GAN ensemble 补救（96.6% door opening）——侧面暴露 GAN 训练方差大的固有弱点。
3. **对 SB-Render-Lite 的直接行动项**：(i) B2 若只放裸 CycleGAN/CUT 会被视为 strawman，必须增加"感知一致性增强 GAN"（B2b，RetinaGAN 思想的低成本复现）；(ii) B1 的 DR 至少设 mild/heavy 两档（RCAN 显示档位间差 4 pp、且 DR 是强预训练起点）；(iii) 各方法真实数据需求谱系已钉死——RCAN 0 / GraspGAN 9.4M 无标签图 / RetinaGAN 135k episodes + 81k 检测标注图——SB-Render-Lite 的 50k unpaired real RGB 在谱系中位于低成本端，这本身就是可写进论文的定位论据；(iv) GAN 推理是 1 NFE，SB 多步推理的成本劣势必须诚实报告。

---

# 精读一：RCAN — Sim-to-Real via Sim-to-Sim: Data-efficient Robotic Grasping via Randomized-to-Canonical Adaptation Networks

## 基本信息

- 论文：Sim-to-Real via Sim-to-Sim: Data-efficient Robotic Grasping via Randomized-to-Canonical Adaptation Networks（RCAN）
- 作者：Stephen James（Imperial College London，工作完成于 X）、Paul Wohlhart、Mrinal Kalakrishnan（X）、Dmitry Kalashnikov、Alex Irpan、Julian Ibarz（Google Brain）、Sergey Levine（UC Berkeley/Google）、Raia Hadsell（DeepMind）、Konstantinos Bousmalis（DeepMind）
- 发表：**CVPR 2019**，pp. 12627–12637，DOI 10.1109/CVPR.2019.01291（web 复核 2026-08-14：CVF Open Access + researchr 双源确认。任务给定的"RCAN=CoRL?"为误记，**正确 venue 是 CVPR 2019**）；arXiv 2018-12（1812.07252）
- 链接：https://arxiv.org/abs/1812.07252 ｜ 项目页：https://sites.google.com/view/rcan/
- 数字口径备注：项目页摘要仍是早期版本数字（zero-shot 66%、微调后 86%）；CVPR 正式版与 arXiv 最新版为 **70% / 91%**，本报告统一采用正式版数字。
- 归类：DR 的翻译式用法；real→canonical-sim 方向的 paired image translation；SB-Render-Lite 的经典对照组。

## 一句话总结

把 domain randomization 从"policy 的训练数据增广"改造成"图像翻译器的监督来源"：学一个把任意重随机化图像映射回 canonical 仿真外观的 cGAN，真实图像作为"又一种随机化变体"也被映射回 canonical 域，从而让完全在 canonical 仿真里训练的 QT-Opt 策略直接在真机上运行。

## 动机

三条既有路线各有致命伤：(a) 域适应方法需要大量真实无标签图像，机器人场景下采集仍昂贵；(b) DR 直接作用在 policy 输入上，迫使算法同时建模"任意视觉变化"和"任务动力学"，浪费建模容量，且已有证据表明 DDPG/A3C 等 RL 算法会被重随机化直接训崩（Matas et al. 2018; Zhang et al. 2019）；(c) 逐场景调渲染保真度不可扩展。RCAN 的判断是：随机化本身没错，错在用法——应该用它监督一个**可解释的中间翻译层**，把视觉复杂度从 policy 中剥离出来。

## 方法核心

**三域设定**：randomized sim（源）、canonical sim（目标）、real（部署期的"额外源"）。训练一个 image-conditioned GAN（pix2pix 式，isola2017image）\(G(x)\rightarrow\{x_a, m_a, d_a\}\)：输入任意域图像，输出 canonical RGB + 分割 mask + 深度图（后两者仅作训练期辅助任务，不喂给 policy）。

**配对数据免费生成**（这是与 unpaired 方法的本质分界）：在 QT-Opt 仿真训练过程中，每个 timestep 冻结场景 → 施加一次全新随机化渲染一帧（源）→ 复位到 canonical 版本再渲染一帧（目标）→ 语义逐像素对齐的 (randomized, canonical, seg, depth) 四元组即得。随机化强度：>5,000 张纹理随机贴到 tray/物体/机械臂/地面，光照位置/方向/颜色随机，臂与 tray 位姿尺寸轻微扰动（同变换施加于两侧以保语义对齐）。物体池 = 1,000 个程序化生成物体 + 51,300 个 ShapeNet 物体；引擎 PyBullet 默认渲染器。canonical 域的人工设计选择：背景/tray/臂上均一色（臂各连杆独立着色便于追踪），**物体保留随机纹理**（保物体身份，为 instance-specific 抓取留路），光源固定（迫使网络学几何以重绘正确方向的阴影）。

**损失**：图像项用 MPSE（mean pairwise squared error，比 L1/L2 收敛快）、seg/depth 辅助项用 L2、外加 sigmoid 交叉熵 GAN 项保高频锐度；\(\lambda\) 全部取 1。生成器 U-Net（下采样为 stride-2 卷积 + 平均池化，双线性上采样，skip 用 concat+1×1 卷积，InstanceNorm+ReLU）；判别器为多尺度 patch 设计（472/236/118 三个尺度联合，架构继承 GraspGAN）。输入分辨率 472×472。

**与 policy 的接口**：QT-Opt 状态 = (图像, 夹爪开合, 夹爪高度)；图像项改为 **G(x) 与原图 x 的通道拼接（6 通道）**——保留原始视觉信息让 Q 网络自行取舍，事后证明这是真实微调阶段 regrasping 行为恢复的关键。动作空间与 Kalashnikov et al. 2018 完全一致。

## 实验与结果（真机规模与提升幅度）

**评估协议**：多台 Kuka IIWA，每台对自己的 5–6 个 **unseen** 测试物体执行 102 次抓取（每次 episode 上限 20 步，抓完放回随机位置），肩上视角单目 RGB。

**主表（Table 1）**，全部为真机 unseen 物体抓取成功率：

| 训练数据源 | 离线真实抓取数 | 仿真成功率 | 真机 zero-shot | +5k 真实 on-policy | +28k |
|---|---:|---:|---:|---:|---:|
| 全真实（QT-Opt 原版） | 580,000 | — | 87% | 85% | 96% |
| Canonical sim 直转 | 0 | 99% | 21% | 30% | — |
| DR-mild 直训 policy | 0 | 98% | 37% | 85% | — |
| DR-medium 直训 policy | 0 | 98% | 35% | 77% | — |
| DR-heavy 直训 policy | 0 | 98% | 33% | 85% | 92% |
| **RCAN** | 0 | 99% | **70%** | **91%** | 94% |

关键读数：
1. **zero-shot：RCAN 70% ≈ 2× DR（33–37%）**。且 DR 档位（mild/medium/heavy）之间差别不大（±4 pp），说明单纯堆随机化强度收益早饱和。
2. **+5,000 真实抓取（约一天采集量，<1% of 580k）**：RCAN 91%，**超过** 580k 真实数据从头训的 87%；+28k 后 94% vs 96% 基本追平。真实数据缩减 >99%。
3. **意外发现**：QT-Opt 对 heavy DR 训练稳定（与 DDPG/A3C 相反），且"DR 直训 + 5k 微调"也能从 33–37% 跳到 77–85%——**DR 是极强的预训练/特征学习机制**，只是 zero-shot 天花板低。微调收益大部分在前 2,000 次抓取内兑现。

## 局限性

- **翻译 artifact 破坏精细行为**：real→canonical 翻译在夹爪与小物体处不完美，policy 无法区分"夹爪里是物体还是 artifact"，导致 QT-Opt 标志性的 regrasping 行为在 zero-shot 阶段基本消失（5k 微调后借助 6 通道拼接的原图恢复）。这是"图像翻译层误差直接变成 policy 观测噪声"的典型案例。
- **canonical 域是手工设计**：颜色方案、光源、保不保物体纹理均为人工决策，无原则性准则；换环境需重新设计并重建整套 randomized/canonical 配对渲染管线。
- **隐含假设：随机化分布"覆盖"真实域**。真实图像被当作 randomized 分布的一个样本处理；若真实域偏出随机化范围（新传感器噪声、极端光照），翻译器行为未定义。
- 论文自己指出：未利用任何真实无标签数据是优点也是天花板，作者把"翻译器融合真实数据"列为 future work。

## 与 SB 路线的本质区别

1. **方向相反**：RCAN 是 real→sim（把真实拉回 canonical 仿真域，policy 永远活在仿真外观里）；SB-Render-Lite 是 sim→real（把仿真数据推向真实外观，policy 活在真实外观里）。方向差异决定了 RCAN 的翻译器必须部署期在线运行（每帧过一次 G，472×472 U-Net），而 SB-Render-Lite 的 transport 只在离线训练数据生成时运行，**部署期零开销**——这是 SB 路线可主张的工程优势。
2. **监督形态**：RCAN 是 paired supervised translation（配对由仿真免费造），不需要学分布间耦合；SB 是 unpaired marginal 之间的熵正则 OT，耦合是学出来的。RCAN 范式无法利用真实数据分布信息，SB 天然利用。
3. **最小改动性**：RCAN 无任何"改动量"控制——canonical 目标离源图可以任意远；SB 的熵正则 \(\varepsilon\) 提供原则性的位移-保真权衡旋钮。
4. **可组合性提示**：RCAN 证明"DR 喂翻译器"优于"DR 喂 policy"。对 SB-Render-Lite 的启示是可做 **DR+SB 组合消融**（对 sim 端先施加轻度 DR 再做 SB transport），检验两者是否互补——RCAN 的结论预言互补。

---

# 精读二：RetinaGAN — An Object-aware Approach to Sim-to-Real Transfer

## 基本信息

- 论文：RetinaGAN: An Object-aware Approach to Sim-to-Real Transfer
- 作者：Daniel Ho（Everyday Robots, X）、Kanishka Rao（Robotics at Google）、Zhuo Xu（UC Berkeley）、Eric Jang（Robotics at Google）、Mohi Khansari、Yunfei Bai（Everyday Robots, X）
- 发表：**ICRA 2021**，DOI 10.1109/ICRA48506.2021.9561157（web 复核 2026-08-14：IEEE DOI + Google Research 官方博客"presented at ICRA 2021"双源确认）；arXiv 2020-11（2011.03148）
- 链接：https://arxiv.org/abs/2011.03148 ｜ 项目页：https://retinagan.github.io
- 归类：语义约束 GAN 翻译；sim→real 方向 unpaired；SB-Render-Lite 语义保持设计的最直接前驱与必比对象。

## 一句话总结

在 CycleGAN 之上加一个"冻结物体检测器的感知一致性损失"：要求 EfficientDet 在翻译前后的图像上给出相同的框与类别预测，从而在不依赖任务损失、不需要配对数据的前提下保住物体级结构与纹理，且同一翻译器可跨任务复用。

## 动机

裸 GAN 翻译的核心风险：对抗目标只约束"看起来像目标域"，不约束"内容不变"——GAN 可以任意增删物体、挪动结构，把任务关键信息抹掉。前作 RL-CycleGAN（Rao et al., **CVPR 2020**，arXiv 2006.09001）的解法是把 RL 任务的 Q 值一致性作为约束联合训练，但这要求：任务特定的真实 episodes、GAN 与 RL 模型联合优化（训练难）、换任务重训。RetinaGAN 的判断是：**物体检测是任务无关但物体敏感的中间监督**——检测标注比任务 episodes 便宜、检测器预训练后冻结（训练稳定）、跨任务复用。

## 方法核心

**基座**：标准 CycleGAN（双向生成器 \(G:X\rightarrow Y\)、\(F:Y\rightarrow X\)，双判别器，cycle 一致性），\(\lambda_{cycle}=10\) 直接继承 RL-CycleGAN 不调参。生成器 U-Net（架构同 RL-CycleGAN），输入 512×640 crop 到 472×472，spectral normalization，batch 512（sim/real 各 256），4×TPUv3 Pods 训 50k–100k 步。

**感知一致性损失**：一次前向产生 6 张图——sim 三张 \(\{x, G(x), F(G(x))\}\)、real 三张 \(\{y, F(y), G(F(y))\}\)。冻结的 EfficientDet-D1 对每张推理，输出 anchor 级 box 回归与 class logits；对域内所有两两图像对计算一致性损失 = box 的 Huber loss + 类别的 **Focal Consistency Loss（FCL）**——论文提出的 focal loss 插值推广，把 one-hot 标签松弛为概率标签 \(y\in[0,1]\)（\(\text{FCL}(y,p)=|y-p|^{\gamma}\,\text{BCE}(y,p)\)，按 anchor 概率归一化），使"预测对预测"的一致性可导可训。含 cycled 图像的对权重减半。\(\lambda_{prcp}=0.1\)（0.1–1.0 区间均稳定）。

**检测器**：EfficientDet-D1，59 类，混合训练数据 = 真实 44,000 张回收站物体标注图 + 37,000 张桌面物体图 + 625,000 张仿真图（PyBullet）。**全部三个任务实验复用同一个检测器**，包括视觉域完全不同的门开启任务（检测器只对机械臂有高置信预测，论文假设门框等结构靠低置信区域的一致性维持——这是承认的弱支撑点）。

## 实验与结果（真机规模与提升幅度）

**任务一：RL instance grasping（Q2-Opt，垃圾分拣站，抓指定类别物体）**。GAN 训练数据：135k 真实 off-policy episodes（低数据版 10k）+ 0.5–1M 仿真 on-policy episodes；Q2-Opt 训练：211k 真实 + 1–2M 仿真。评估：6 台机器人 × 90 次抓取/评估，报告均值与 Bernoulli 标准差估计。

| 配置 | 成功率 | est. std |
|---|---:|---:|
| Sim-Only（光度扰动，无 GAN） | 18.9% | 4.1% |
| Randomized Sim（纹理+光照 DR） | 41.1% | 5.2% |
| Real-only @10k episodes | 22.2% | 4.4% |
| RetinaGAN @10k（纯 GAN 数据训 policy） | 47.4% | 5.3% |
| RetinaGAN+Real @10k | 65.6% | 5.0% |
| Real-only @211k | 30.0% | 4.9% |
| Sim+Real @135k/211k | 54.4% | 5.3% |
| RetinaGAN+Real @135k/211k | **80.0%** | 4.2% |
| —— 纯 sim-to-real 数据对比（policy 不见任何真实 episode）—— | | |
| CycleGAN | 67.8% | 5.0% |
| RL-CycleGAN | 68.9% | 4.9% |
| **RetinaGAN** | **80.0%** | 4.2% |

关键读数：
1. **裸 CycleGAN 67.8% → RetinaGAN 80.0%（+12.2 pp，超过两个标准差）**——感知一致性约束的净贡献。RL-CycleGAN（任务耦合约束）只到 68.9%，说明**任务无关的物体级约束反而赢过任务耦合约束**（作者归因于 RL-CycleGAN 的 Q 一致性是为 indiscriminate grasping 设计、迁到 instance grasping 后失配，同时联合训练更难）。
2. **数据效率 >10×**：RetinaGAN+Real@10k（65.6%）> Sim+Real@135k/211k（54.4%）。
3. RetinaGAN（纯 GAN 数据，80.0%）≈ RetinaGAN+Real（80.0%）——真实域知识大部分已被 GAN 吸收。

**任务二：3D object pushing（把立着的茶瓶推到目标点不推倒）**：**同一个 GAN、零新增真实数据**，policy 纯仿真训练。Sim-only 0% → RetinaGAN 90%（10 次试验）。证明翻译器跨任务复用（同一视觉环境内）。

**任务三：IL 门开启（ResNet-FiLM-18 行为克隆，机械臂固定、控制底盘开会议室门）**：1,500 仿真 + 29,000 真实人类演示；视觉域与检测器训练域完全不同。30 次试验：

| 配置 | 成功率 |
|---|---:|
| Sim-only | 0.0% |
| Real-only | 36.6% |
| Sim+Real | 75.0% |
| RetinaGAN+Real | 76.7% |
| Ensemble-RetinaGAN（3 个不同 seed/权重的 GAN）+Real | 93.3% |
| Ensemble-RetinaGAN（IL 完全不用真实演示） | **96.6%** |

关键读数：低数据 + 域外检测器时，**单个 RetinaGAN 输出在光照/颜色语义上振荡**，须靠 3-GAN ensemble 添加多样性才达 93–97%；而裸 CycleGAN 在此域会**扭曲房间结构与门的位置**，作者以安全风险为由拒绝上真机评估（附录 Fig. 8 给了扭曲示例）——裸 GAN 语义漂移在低数据域的失败模式被官方坐实。

## 局限性

- **检测标注不免费**：44k+37k 张真实标注图是前置成本；论文以"检测是机器人通用能力、标注可摊销"辩护，并指出用 off-the-shelf 检测器是方向但未验证（纯真实数据训的检测器在 sim/real 两域预测不平衡可能训崩 GAN）。
- **约束粒度受限于检测器**：box 级一致性不管背景结构；域外场景靠低置信区域一致性支撑属于"假设"而非机制。分割级（Mask R-CNN/ShapeMask）更强但标注更贵，列为 future work。
- **GAN 固有训练方差**：ensemble 补救本身就是"单次训练不可靠"的证据；对抗 min-max + cycle 的组合在低数据下尤其不稳。
- 评估规模：grasping 90 次/配置、pushing 10 次、door 30 次——按今天标准（COLOSSEUM 等）偏小，std 估计基于独立 Bernoulli 假设。

## 与 SB 路线的本质区别

1. **约束机制**：RetinaGAN 的感知一致性是"外部冻结感知模型 + 损失惩罚"，与 GSBM（`reports/2310.02233`）把 task-aware cost 写进路径状态代价、以及 SB-Render-Lite 计划中的 depth/DINO 一致性项（\(\lambda_g\)）在动机上**同构**——都是"分布对齐目标之外锚定语义"。区别在 SB 框架里约束进入 transport 的变分目标，与熵正则最小位移原则共存；GAN 里约束与对抗项直接相加，无位移控制，权重 \(\lambda_{prcp}\) 全靠试。**写论文时必须把 RetinaGAN 列为语义保持设计的直接前驱**，并论证 SB 版本的增量（原则性 \(\varepsilon\)、无对抗不稳定性、cost 可插拔）。
2. **训练动力学**：CycleGAN 系是 min-max 对抗（模式坍缩、方差大、需 ensemble）；SB Flow/IMF 是迭代回归匹配（无判别器）。这是 SB 可主张的稳定性叙事，但必须用 B2/B2b 的多 seed 方差数据实证，不能只讲理论。
3. **cycle 一致性 vs 熵正则耦合**：cycle 假设双向 bijection（CUT 论文已指出过强）；SB 的耦合由熵正则 OT 决定，不假设 bijection，且 \(\varepsilon\rightarrow 0\) 时收敛到确定性 OT map。
4. **复用性主张的参照**：RetinaGAN 用"同一 GAN 复用到 pushing"证明翻译器摊销价值。SB-Render-Lite 若主张跨任务/跨场景摊销，应设计对应实验（同一 transport 服务两个下游任务），否则该项优势叙事让位。

---

# 收录条目（导航级）

## 条目 1：Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World（Tobin et al.）

- 作者：Josh Tobin, Rachel Fong, Alex Ray, Jonas Schneider, Wojciech Zaremba, Pieter Abbeel（OpenAI / UC Berkeley）
- 发表：**IROS 2017**，pp. 23–30，DOI 10.1109/IROS.2017.8202133（web 复核 2026-08-14：Google Scholar + ResearchGate DOI 确认）；arXiv 1703.06907
- 链接：https://arxiv.org/abs/1703.06907
- 内容：视觉 DR 的奠基论文。用 MuJoCo 内置低保真渲染器 + **纯非真实感随机纹理**（随机 RGB/渐变/棋盘格）训练 VGG-16 改造的物体定位器（回归物体桌面坐标），随机化维度：纹理（所有表面）、distractor 数量与形状、相机位姿与 FOV、光源数量/位置/镜面参数、图像噪声。**零真实图像**达到真机 1.5 cm 定位精度，抗干扰物与部分遮挡；接经典运动规划在 Fetch 上完成杂乱场景抓取 38/40。
- 关键消融（对协议设计直接有用）：(i) 纹理种类 <1,000 时性能显著劣化——**随机化多样性有硬下限**；(ii) 训练场景含 distractor 是抗真实杂乱的必要条件；(iii) ImageNet 预训练在大数据量下**非必需**（推翻作者自己的假设）；(iv) 5k 样本可用、50k 饱和；(v) 相机随机化有小幅稳定收益。
- 与 SB-Render-Lite 的关系：B1（DR 对照组）的思想源头，但注意 Tobin 版是"检测器 + 经典规划"的模块化管线而非端到端 policy；现代 DR 基线应按 RCAN 中"DR 直训 policy"的 mild/heavy 配置实现。DR 的本质是**以覆盖换鲁棒、不产生翻译图像**，因此 B1 没有图像域指标，只有 policy success——协议中要明确这一不对称。

## 条目 2：Using Simulation and Domain Adaptation to Improve Efficiency of Deep Robotic Grasping（GraspGAN）

- 作者：Konstantinos Bousmalis*, Alex Irpan*, Paul Wohlhart*, Yunfei Bai, Matthew Kelcey, Mrinal Kalakrishnan, Laura Downs, Julian Ibarz, Peter Pastor, Kurt Konolige, Sergey Levine, Vincent Vanhoucke（Google Brain / X）
- 发表：**ICRA 2018**（web 复核 2026-08-14：Google Research 出版页"ICRA (2018)"确认）；arXiv 1709.07857
- 链接：https://arxiv.org/abs/1709.07857
- 内容：机器人 sim2real GAN 翻译的开山系统。pixel-level（GraspGAN：U-Net 生成器 + **多尺度 472/236/118 三档 70×70 patch 判别器** + LSGAN）与 feature-level（DANN + domain-specific batch norm）混合适应。语义锚定三件套：PMSE content loss + 生成器**兼预测分割 mask** 的辅助任务 + 任务网络末层激活一致性——这套"语义锚定"设计是 RCAN（aux seg/depth）与 RetinaGAN（检测一致性）的共同先声。数据：真实 9.4M 帧（~1M 次抓取，6 台 Kuka，**只用图不用标签**训 GAN）+ 仿真 8M 样本（PyBullet）。
- 关键数字：25,704 次真机测试抓取、36 个 unseen 物体、612 次/评估。**零真实标签：sim-only 23.5% → DR 36.0% → GraspGAN 63.4%，追平用 939,777 个真实标签训练的 62.75%**；2% 真实标签（188k）时 68.5% vs real-only 35.5%；总体真实样本需求缩减 **up to 50×**。附带重要发现：**1,000 个程序化随机物体 > 51,300 个 ShapeNet 真实感物体**（所有随机化档位下一致）——资产真实感不是瓶颈。
- 与 SB-Render-Lite 的关系：确立了"unpaired 真实 RGB（无标签）+ 仿真数据 → 翻译 → 下游 policy"的完整评估范式，与 SB-Render-Lite 的数据假设完全同构，是数据效率叙事的锚点引用；其多尺度 patch 判别器被 RCAN/RetinaGAN 沿用，B2 实现时保持 CycleGAN/CUT 官方判别器即可，不必复刻。

## 条目 3：Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks（CycleGAN）

- 作者：Jun-Yan Zhu*, Taesung Park*, Phillip Isola, Alexei A. Efros（BAIR, UC Berkeley）
- 发表：**ICCV 2017**，pp. 2223–2232（web 复核 2026-08-14：CVF Open Access 确认）；arXiv 1703.10593
- 链接：https://arxiv.org/abs/1703.10593
- 内容：unpaired 翻译的事实标准。双向生成器 \(G/F\) + 双判别器 + cycle 一致性 L1（\(\lambda=10\)），可选 identity loss（\(0.5\lambda\)，防无谓改色）。官方配置（B2 复现依据）：ResNet 生成器（256×256 用 9 个 residual block，~11.4M 参数）、70×70 PatchGAN 判别器（C64-C128-C256-C512）、LSGAN 损失、InstanceNorm、Adam batch=1、lr 2e-4、100 epochs 恒定 + 100 epochs 线性衰减、权重 \(\mathcal{N}(0,0.02)\) 初始化。
- 已知失败模式（写 related work 可引）：bijection 假设过强；分布对齐不保证个体对应（论文自己承认"infinitely many mappings 诱导同一分布"）；对抗训练 mode collapse；几何/结构改变类翻译失败。RetinaGAN 附录展示了其在低数据机器人域扭曲房间结构的实例。
- 与 SB-Render-Lite 的关系：B2 官方基线之一；`reports/2409.09347_schrodinger_bridge_flow_unpaired_translation.md`（SB Flow）中 DSBM/SB 系方法的标准图像域对照即是它。

## 条目 4：Contrastive Learning for Unpaired Image-to-Image Translation（CUT）

- 作者：Taesung Park, Alexei A. Efros（UC Berkeley）, Richard Zhang, Jun-Yan Zhu（Adobe）
- 发表：**ECCV 2020**，LNCS vol. 12354, pp. 319–345，DOI 10.1007/978-3-030-58545-7_19（web 复核 2026-08-14：Springer + ECVA 官方 PDF 确认）；arXiv 2007.15651
- 链接：https://arxiv.org/abs/2007.15651
- 内容：用 **PatchNCE**（多层、patch 级 InfoNCE，τ=0.07）替换 cycle 一致性：翻译后 patch 应与输入图同位置 patch 互信息最大，**负样本取自同一图像内部**（外部负样本反而有害）。单向翻译，复用生成器 encoder 前半段取 5 层特征 + 2 层 256 维 MLP 投影。两个官方配置：**CUT**（\(\lambda_X=\lambda_Y=1\)，含 identity NCE 正则，质量最优）与 **FastCUT**（\(\lambda_X=10,\lambda_Y=0\)，更快更省）。
- 关键数字（Table 1，B2 选型依据）：horse→zebra FID **CycleGAN 77.2 vs CUT 45.5** vs FastCUT 73.4；Cityscapes FID 76.3 vs 56.4，且语义对应指标（mAP 20.4→24.7）更高；训练开销 sec/iter 0.40 vs 0.24 vs 0.15、显存 4.81 vs 3.33 vs 2.25 GB（GTX 1080Ti 实测）——**CUT 比 CycleGAN 快 40% 省 31% 显存且全面更优**。数据规模参考：horse2zebra 训练集共 2,403 张。
- 与 SB-Render-Lite 的关系：实验计划 B2 已指定 CUT 官方实现；PatchNCE 的"同位置 patch 互信息"与 SB-Render-Lite 的 DINO patch 一致性项在功能上重叠，消融叙事时应说明二者关系（前者学习相似度、后者用冻结特征）。CUT 是"经典 GAN 线的最强性价比代表"，若 SB 在几何指标上赢不了 CUT，方向判停条款（实验计划 §4.3）触发。

---

# SB-Render-Lite 经典基线协议规格清单

> 对接 `generative_policy/sb/reports/sb_render_lite_experiment_plan.md` 的对照组表（B0–B5）。以下规格把 B1/B2 钉死到可复现粒度，新增 B2b，并给出统一指标与公平性条款。所有对照组共享：同一批 sim 数据（ManiSkill2 五元组 50k 帧）、同一批 real RGB（50k 帧，只取 marginal）、同一下游 BC policy（Diffusion Policy 或 ACT 固定一个不调参）、同一评估域（real-proxy，100 episodes × 3 seeds）。

## B1：Domain Randomization（Tobin/RCAN 式）

| 项 | 规格 |
|---|---|
| 训练数据需求 | 仅 sim；**0 真实数据**（这是 DR 的核心卖点，报告时明示） |
| 随机化维度 | 按 Tobin 清单实现：全表面纹理（≥1,000 种非真实感纹理，低于此数有消融证据劣化）、光照位置/颜色/强度、相机位姿 ±(10,5,10)cm 级扰动 + FOV ±5%、distractor 物体 0–10 个、图像噪声 |
| 档位 | **两档必做**：mild（纹理+光照）与 heavy（全维度）。依据：RCAN 显示档位间 zero-shot 差 4 pp、微调后走向不同（77–85%），单档报告会被质疑 cherry-pick |
| 网络/训练 | 与 B0 完全相同的 policy 架构与超参；DR 只改数据管线（ManiSkill2 渲染 wrapper），帧数与 B4/B5 transported 帧数对齐（50k） |
| 评估指标 | 仅 policy success（DR 不产生翻译图像，无图像域指标——协议中显式标注该不对称） |
| 注意事项 | (i) DR 是强预训练：若后续加"少量真实微调"扩展实验，DR 组必须同样微调（RCAN：DR+5k 达 77–85%，差距会大幅收窄）；(ii) 记录 DR 采样种子使数据可复现；(iii) 可选组合组 B1+B5（sim 先 DR 再 SB transport），检验 RCAN"随机化喂翻译器更优"的结论在 SB 上是否成立 |

## B2：CycleGAN / CUT（裸 unpaired GAN 翻译）

| 项 | 规格 |
|---|---|
| 训练数据需求 | unpaired：sim 50k 帧 + real 50k 帧，**与 SB 组完全同一批数据、同一预处理**；禁止使用任何配对/标注信息 |
| 实现 | 官方代码库（junyanz/pytorch-CycleGAN-and-pix2pix 与 taesungp/contrastive-unpaired-translation），官方默认超参：CycleGAN = ResNet-9blocks + 70×70 PatchGAN + LSGAN + \(\lambda_{cycle}=10\)（identity loss 开/关各跑一次并报告）、batch 1、lr 2e-4、100+100 epochs；CUT = 同架构 + PatchNCE（\(\lambda_X=\lambda_Y=1\)），FastCUT 作为低成本参考点 |
| 网络规模 | 生成器 ~11.4M 参数 vs SB UNet ~60M——**参数量差 5 倍必须在论文表格中明示**，并加一档"容量对齐"消融（SB 缩到 ~15M 或 CUT backbone 加宽）堵 reviewer 的容量混淆质疑 |
| 分辨率/空间 | 官方像素级 256×256（保持"经典基线最强官方形态"）；另跑一个 SD-VAE latent 版 CUT 作为与 SB 同空间的公平对照（latent 版为非官方改装，结果分开报告） |
| 训练预算 | 参考 CUT 实测 0.24–0.40 sec/iter（1080Ti），256×256 50k 帧规模约 1–2 GPU·day/seed |
| 多 seed | **≥3 seeds 硬性要求**，报 per-seed 与均值±std。依据：RetinaGAN 的 ensemble 补救证明 GAN 单 seed 方差大，单 seed 对比对双方都不可信 |
| 评估指标 | 图像域 FID/KID（到 real marginal，参考用）+ 几何保持全套（depth-L1、SuperPoint/LightGlue 重投影、seg IoU、DINO cosine）+ 下游 policy success。**预期失败模式**（写入预注册假设）：FID 可以很好但几何指标出现语义漂移（物体挪位/纹理吞噬），这正是 CycleGAN 系的文献级已知弱点 |
| 推理成本 | GAN 生成 1 NFE vs SB Flow N 步——在成本表中如实报告；SB 若 NFE>4 需说明离线数据生成场景下该成本可接受（部署期零开销叙事，见 RCAN 精读 §区别 1 的反向对照） |

## B2b（新增建议）：感知一致性增强 GAN（RetinaGAN 式）

> 动机：只打裸 CycleGAN/CUT 会被 reviewer 视为 strawman——RetinaGAN 已证明加感知一致性 +12 pp。SB-Render-Lite 主打语义/几何保持，公平的"最强 GAN 对手"必须带同类约束。

| 项 | 规格 |
|---|---|
| 实现 | CUT（或 CycleGAN）+ 冻结感知模型一致性损失。**低成本复现路径**：不训 EfficientDet（原版需 44k+37k 真实标注，不可行），改用 (a) 冻结 DINO v2 patch feature 一致性（与 B5 的 \(\lambda_g\) 项同一冻结模型，保证约束强度可比），或 (b) 开放词汇检测器（Grounding DINO）对 sim 类别做框一致性（Huber on boxes）。权重参考 RetinaGAN：\(\lambda_{prcp}\in[0.1, 1.0]\) 稳定区间，取 0.1 起步 |
| 训练数据需求 | 同 B2（unpaired 50k+50k）；感知模型零标注成本（这是相对原版 RetinaGAN 的协议差异，须注明） |
| 意义 | 该组与 B5 的对比是论文核心论点的直接检验："同样的语义约束，装在对抗翻译器上 vs 装在 SB transport 上，哪个下游更好、哪个更稳（seed 方差）"。若 B2b ≈ B5，SB 的增量主张只剩熵正则旋钮与训练稳定性，论文定位需相应调整 |

## B-RCAN（可选，降级为讨论项亦可）

| 项 | 规格 |
|---|---|
| 实现 | 在 ManiSkill2 中同 seed 冻结场景双渲染（randomized/canonical）生成配对数据，pix2pix 式训练 G，policy 在 canonical 域训练，评估时 real-proxy 帧经 G 拉回 canonical |
| 价值 | 提供"**0 真实数据**"参照点：SB 用了 50k real RGB，RCAN 一张不用——该对照直接量化"real marginal 信息值多少个百分点"。若资源紧张，可只在论文 related work 引用 RCAN 原始数字（70% vs DR 33–37%）并说明协议不可比 |
| 注意 | RCAN 方向是 real→canonical，与 B2/B5 的 sim→real 方向相反，**图像域指标不可直接横比**，只比下游 success；且需要 canonical 域人工设计（记录设计决策） |

## 统一指标体系与公平比较条款

**指标（全组统一）**：
1. 主指标：real-proxy 域 policy success（100 eps × 3 seeds，均值±std，报相对 B0 的 Δpp）；
2. 几何/语义：depth-L1、keypoint 重投影、seg IoU、DINO cosine——SB vs GAN 的主战场；
3. 分布：FID/KID 仅参考，不进结论（GraspGAN/RCAN/RetinaGAN 三篇都没用 FID 说话，机器人社区认 success rate）；
4. 成本表：训练 GPU·h、参数量、推理 NFE 与 ms/帧、真实数据需求（帧数/标注量）；
5. 数据效率曲线（若预算允许）：real marginal 取 {2k, 10k, 50k} 三档重训 B2/B2b/B5——对应 RetinaGAN 10k 实验与 GraspGAN 1%/2% 实验的现代版，SB 若主打低数据稳定性，这条曲线是关键证据。

**公平比较硬性条款**：
1. 所有翻译类方法（B2/B2b/B4/B5）用**同一批 unpaired 数据、同一下游 policy 配置**；
2. GAN 组 ≥3 seeds、SB 组同样 ≥3 seeds，方差进表；
3. 参数量与容量对齐消融至少一档；
4. 翻译方向不同的方法（RCAN 式）只比下游指标；
5. 引用原论文数字时必须注明协议差异：RCAN/RetinaGAN/GraspGAN 均为 **QT-Opt/Q2-Opt RL + 真机在线评估**，SB-Render-Lite 是**离线 BC + sim proxy 评估**，数字不可直接搬运对比，库内实验以自复现为准；
6. DR 组档位、GAN 组 identity loss 开关、SB 组 \(\varepsilon\) —— 各方法的"自由旋钮"都要报告扫描结果或固定依据，不允许单点最优对单点默认。

**六篇原始论文关键事实速查表**（引用时的数字锚点，检索/核验日期 2026-08-14）：

| 方法 | venue（已核验） | 真实数据需求 | 真机评估规模 | 关键数字 |
|---|---|---|---|---|
| Tobin DR | IROS 2017 | 0 | Fetch 40 次抓取 | 定位 1.5cm；抓取 38/40；纹理需 ≥1k 种 |
| GraspGAN | ICRA 2018 | 9.4M 无标签帧 | 25,704 次测试抓取 | 无标签 63.4% ≈ 940k 标签的 62.75%；50× 缩减 |
| RCAN | CVPR 2019 | 0（+可选 5k on-policy） | 102 次/机器人 | zero-shot 70% vs DR 33–37%；+5k→91% > 580k 真实的 87% |
| RetinaGAN | ICRA 2021 | GAN 135k eps（低配 10k）+ 检测标注 81k 帧 | 90 次/评估 | 纯 GAN 数据 80.0% vs CycleGAN 67.8%（+12pp）；跨任务 0%→90% |
| CycleGAN | ICCV 2017 | unpaired 两域各 ~1–1.3k 图（视觉基准） | — | horse→zebra FID 77.2；~11.4M 参数生成器 |
| CUT | ECCV 2020 | 同上 | — | FID 45.5；比 CycleGAN 快 40% 省 31% 显存 |

---

# 并入主库建议

1. **INDEX.md 归类**：建议新增小节"经典基线：DR 与 GAN 翻译（sim2real 对照组）"，置于"重要对照 / SB 方法支撑"之后，收录本报告（两篇精读 + 四条导航）。RCAN 与 RetinaGAN 按库内惯例可拆为独立 `reports/1812.07252_rcan_randomized_to_canonical.md` 与 `reports/2011.03148_retinagan_object_aware_sim2real.md`（内容直接取本报告对应章节），Tobin/GraspGAN/CycleGAN/CUT 维持导航条目。
2. **papers.tsv 增补**：六行新条目（id、title、year、category 建议：`domain_randomization_baseline`、`gan_pixel_da_grasping`、`randomized_to_canonical_translation`、`object_aware_gan_sim2real`、`unpaired_translation_cyclegan`、`unpaired_translation_contrastive`）。注意 RCAN venue 写 CVPR 2019（缺口分析 R09 已写对，任务分工单中的"CoRL?"为误记，勿带入主库）。
3. **synthesis.md §4.2 基线表升级**：现有"必比 baseline"清单可直接替换为本报告的 B1/B2/B2b 协议规格（含数据需求、seeds、容量对齐条款）；建议把"B2b 感知一致性增强 GAN"写进实验计划正式对照组，它是审稿人视角下真正的强基线。
4. **交叉链接**：RetinaGAN 精读 §区别 1 应与 `reports/2310.02233_generalized_schrodinger_bridge_matching.md`（task-aware cost）互链；RCAN §区别 4 的 DR+SB 组合消融建议回写进实验计划 §3 的可选加测行。RL-CycleGAN（CVPR 2020，arXiv 2006.09001）本报告仅间接覆盖（经 RetinaGAN 对照数据 68.9%），若后续需要"任务耦合约束 GAN"的完整谱系可作为小型增补选题。
5. **与 G1（E09/E10 的重建路线）的合流点**：DR/GAN 线与重建/渲染线共同构成"SB 的两面对照"——前者是分布级但无理论最小位移，后者是逐场景高保真但不可摊销。建议最终 related work 用"逐场景重建 ↔ 分布级 transport ↔ 覆盖式随机化"三极图组织，本报告与 E09/E10 各占一极的事实基础。
