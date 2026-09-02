# E09：SplatSim + RialTo 精读 —— 重建/渲染 real2sim 竞品路线（上篇）

> 扩充研究员：E09 ｜ 日期：2026-08-14 ｜ 选题来源：内部审查 R09 缺口 **G1**（机器人 sim2real 渲染/重建竞品系统线，P0 优先级）
> 任务：SplatSim（arXiv 2409.10161）与 RialTo（arXiv 2403.03949）全文精读，聚焦**输入假设（需要什么真实数据/扫描）**与**真机评估协议**；产出"重建/渲染路线(real2sim) vs SB transport 路线"对照表中本文两篇的行（LucidSim/X-Sim 两行由选题 E10 补齐）。
> 全文获取方式：两篇均经 arXiv 官方 HTML 全文（含附录）精读；RialTo 另经 RSS 官方 proceedings PDF 核对。非二手转述。

## 选题定位

`SB-Render-Lite` 的核心主张是"用 SB/生成式 transport 缩小 sim→real 视觉域差，从而提升真机策略"。机器人社区当前的主流替代范式是 **real2sim 重建/渲染路线**：先从真实世界采集扫描，重建 photorealistic 渲染资产（SplatSim 的 3DGS）或几何数字孪生（RialTo 的 mesh + 关节标注），再在"贴近真实的仿真"里训练策略。reviewer 的第一个问题必然是"为什么用 SB transport 而不是直接重建渲染"。本报告精读这条路线的两个代表系统，把它们的**输入假设、人力成本、真机协议、保真上限**钉死成可引用的事实，为 SB 路线的差异化论证提供参照系。

## TL;DR

1. **SplatSim（ICRA 2025）**：在 PyBullet 里用 3D Gaussian Splatting 替换 mesh 渲染，仅凭一次静态场景扫描（含机器人）+ 每物体扫描，就能渲染 photorealistic 的机器人-物体交互轨迹，训练 Diffusion Policy 后 **zero-shot 真机部署达 86.25%**（Real2Real 上限 97.5%）。零真实演示、零真实深度。但**关键消融**：即便渲染已如此逼真，去掉图像增广后成功率从 86.25% 暴跌至 21%——photorealistic 重建之后仍存在致命的残余外观 gap（阴影/反射/线缆等），这恰是学习式 transport 的目标对象。
2. **RialTo（RSS 2024）**：用手机扫描 + 15 分钟 GUI 人工标注构建**几何数字孪生**（USD，含关节），提出 inverse distillation 把 ~15 条真实演示转成带 privileged state 的仿真演示，再用 PPO+BC 稀疏奖励微调 + teacher-student 蒸馏（与真实数据 co-training）回真机。8 任务下对 BC 基线平均提升 67%（扰动最强档 75% vs 5%）。它用**点云模态绕开视觉 gap**而非解决它；动力学 gap 不做系统辨识，靠"演示先验 + 真实 co-training"补偿。
3. **对 SB-Render-Lite 的核心启示**：real2sim 路线是"逐场景专用化"范式——RialTo 的 Objaverse 消融（通用资产训练 10% vs 目标场景孪生 90%）直接证明其不可摊销性；SB transport 的差异化主张应是**分布级、免重建、跨场景摊销**，且可作为 GS 渲染之上的 residual bridge（组合而非互斥）。

---

# 精读一：SplatSim — Zero-Shot Sim2Real Transfer of RGB Manipulation Policies Using Gaussian Splatting

## 基本信息

- 论文：SplatSim: Zero-Shot Sim2Real Transfer of RGB Manipulation Policies Using Gaussian Splatting
- 作者：M. Nomaan Qureshi, Sparsh Garg, Francisco Yandun, David Held, George Kantor, Abhisesh Silwal（全部 CMU）
- 发表：**ICRA 2025**（任务给定已核验；检索日期 2026-08-14）；arXiv 2409.10161
- 链接：https://arxiv.org/abs/2409.10161 ｜ 项目页：https://splatsim.github.io
- 归类：real2sim 渲染路线；3DGS photorealistic rendering；zero-shot RGB policy transfer；SB-Render-Lite 直接竞品。

## 一句话总结

把仿真器（PyBullet）的 mesh 渲染管线替换成对齐到仿真坐标系的 3D Gaussian Splatting 渲染：物理仍由仿真器算，图像由真实场景的 splat 资产按仿真状态摆位后渲染，从而让"仿真里收集的演示"自带 photorealistic 外观，训练出的 RGB Diffusion Policy 可 zero-shot 上真机。

## 解决的问题

RGB 是操作任务中信息最丰富的模态（颜色/纹理/光照/反射，例如判断水果成熟度只能靠 RGB），但视觉 sim2real 本质是 out-of-domain generalization：仿真渲染图像分布与真实图像分布差异过大，导致 RGB 策略几乎无法 zero-shot 迁移。已有成功案例（locomotion、in-hand rotation）全部依赖 depth/点云/触觉这些"天然 gap 小"的模态。SplatSim 要做的是把 RGB 的渲染分布直接拉到真实分布附近，而不是像 domain randomization 那样靠覆盖换鲁棒。

## 方法核心：3DGS 渲染管线

**总体结构**：专家演示在 PyBullet 中收集（人类 Gello 遥操作，或用 privileged 状态的运动规划器自动生成）→ 仿真器输出每个时刻的状态 \(s_t=(q_t, x^1_t,\dots,x^n_t)\)（关节角 + 各物体位姿）→ 把状态喂给"与仿真器对齐的 splat 模型"渲染 photorealistic 图像 \(I^{sim}\) → 得到 \(\{(I^{sim}_t, a_t)\}\) 训练 Diffusion Policy；动作是末端执行器位姿。部署时策略只吃真实 RGB。

关键前提：只要能把真实场景 splat 中的每个刚体分割出来、并求出它相对仿真坐标系的齐次变换，就能在任意新位姿渲染该刚体。3DGS 的显式点云式结构使刚体变换只需 \(\mu' = R\mu + t\)、\(\Sigma' = R\Sigma R^{T}\)，渲染质量不塌。

1. **Robot Splat Model（机器人渲染）**：
   - 对含机器人的静态场景 splat \(\mathcal{S}_{real}\) **手动分割**出机器人的 3D Gaussians；
   - 用其均值点云与仿真器 ground-truth 点云做 **ICP**，得 splat 系→机器人系变换 \(T^{\mathcal{F}_{splat}}_{\mathcal{F}_{robot}}\)；
   - 用 **CAD 模型提供的各 link 轴对齐包围盒**把 Gaussians 分到各 link；
   - 由 PyBullet 正运动学得每个 link 的 \(T^l_{fk}\)，按 \(T=(T^{\mathcal{F}_{splat}}_{\mathcal{F}_{robot}})^{-1} T^l_{fk} T^{\mathcal{F}_{splat}}_{\mathcal{F}_{robot}}\) 变换 Gaussians，即可渲染任意关节角下的机器人。
2. **Object Splat Model（物体渲染）**：每个物体单独多视角扫描成 \(\mathcal{S}^k_{obj}\)，ICP 对齐到该物体在仿真中的 ground-truth 点云；随后按仿真状态中的物体位姿变换渲染。
3. **铰接体/夹爪**：平行夹爪的 link 不与坐标轴对齐，包围盒切不开，改用**在 URDF 标注的仿真点云上训练 KNN 分类器**给每个 Gaussian 判 link 归属。
4. **策略训练**：Diffusion Policy + 图像增广（高斯噪声、random erasing、亮度/对比度、color jitter）。作者明确说明增广是为覆盖 splat 渲染不了的因素：**无阴影、无动态反射、线缆等非刚体不渲染**。

方法论上值得注意：这不是学习式方法——整条渲染管线没有任何可训练的 domain 适配模块，全部是几何对齐（ICP + FK + 刚体变换）。视觉保真完全由 3DGS 重建质量决定。

## 输入假设与人力成本

需要什么才能启动（按论文原文逐项核对）：

- **一段含机器人（home 位）的静态真实场景多视角 RGB 视频/扫描**——这是唯一的场景级真实数据；
- **每个被操作物体的多视角扫描**（单独的 object splat）；
- **机器人 CAD/URDF 模型**（link 包围盒分割、KNN 训练、FK 全依赖它）；
- 手动分割机器人 Gaussians + 每资产一次 ICP 对齐的人工/工程成本；
- **零真实演示、零真实深度、零真实交互数据**（论文明确："eliminates the need for the real-world data collection to learn these interactions, and relies solely on an initial video of the static scene with the robot"）。

人力成本论文只量化了演示收集端：仿真 3.0 小时（仅 T-Push 需人类遥操作，其余 3 任务由运动规划器全自动生成 400 条/任务）vs 真实演示 20.5 小时。**扫描与资产制作时长未报告**——这是与 RialTo（GUI 用户研究给了分钟级计时）相比的一个透明度缺口，引用时应注意。

## 真机实验协议与结果

- **硬件**：UR5 + Robotiq 2F-85，2× RealSense D455（仅用 RGB），部署推理 RTX 3080Ti。
- **协议**：4 个任务 × 40 trials；三元对照 **Sim2Sim / Real2Real / Sim2Real**，用于隔离"迁移损失"（Sim2Sim 95.62% → Sim2Real 86.25% 的差值才是残余 gap，而非任务本身难度）。
- **结果**（成功率，40 trials）：T-Push 90%（演示：160 条 Gello 遥操作）；Pick-Up-Apple 95%（400 条规划器）；Orange-On-Plate 90%（400 条）；Assembly 70%（精确放置最难）。平均 **86.25%**，对照 Real2Real **97.5%**、Sim2Sim **95.62%**。
- **渲染保真量化**：300 个关节构型下渲染图 vs 真实图，PSNR 22.62 / SSIM 0.7845。
- **关键消融——增广**：无图像增广时 4 任务平均只有 **21%**，加增广后 86.25%。作者归因于 splat 渲染无法覆盖的动态因素（反射、阴影变化）。

## 局限性

论文自认 + 精读补充：

1. **只支持刚体**：布料、液体、植物、线缆全部出界（结论章明确）；机器人本体线缆也不渲染。
2. **光照烘焙**：splat 是静态场景一次扫描，阴影/反射不随交互变化；渲染图无阴影。
3. **逐场景逐物体线性成本**：每个新部署场景要重扫，每个新物体要单独扫描 + ICP；策略视觉分布绑定被扫描的那个场景。
4. **依赖 CAD/URDF 与手动分割**：机器人 Gaussians 手动分割、link 靠 CAD 包围盒，换机器人平台要重做。
5. **残余 gap 由增广硬扛**：21%→86.25% 说明"重建保真"并不闭环，最后一段 gap 交给了无结构的通用增广——这段 gap 恰好是可学习 transport 的用武之地（见对照行）。
6. 评估任务均为桌面短程操作，无遮挡重、光照剧烈变化的场景压力测试。

## 与 SB transport 路线的对照（本篇的行，详述版）

- **数据需求**：场景扫描×1 + 每物体扫描 + CAD；零演示零深度。SB transport 需要的是 unpaired 真实域图像 marginal（可以是随手采的相机流，无需扫描运动轨迹覆盖多视角），但**通常需要仿真侧渲染与真实侧图像两个 marginal 的样本量都足够**。
- **资产成本**：SplatSim 成本在资产制作（扫描/分割/ICP，逐场景线性）；SB 成本在一次性 transport 模型训练，逐场景边际成本≈采一批真实图像。
- **视觉保真上限**：SplatSim 上限=3DGS 重建质量（PSNR 22.62/SSIM 0.78），**几何一致性由构造保证**（刚体变换不会画错物体位置），但光照/阴影/非刚体封顶；SB 上限=生成模型容量，能建模阴影反射，但**无几何硬约束**，有语义/几何漂移风险（需 GSBM 类 task-aware cost 约束，见库内 `2310.02233`）。
- **动力学一致性**：SplatSim 完全依赖 PyBullet 原生物理，渲染与物理解耦、物理 gap 不处理；SB 视觉 transport 同样不处理动力学——两者在这个维度打平，都需要与 policy 侧方法（如库内 `2509.18631` 的 joint feature-action OT co-training）组合。
- **可扩展性**：SplatSim 逐场景专用化、只刚体；SB 可跨场景摊销（一个 transport 模型伺候多场景/多物体），且对非刚体外观（布料反光等）没有表示层障碍。
- **真机证据强度**：SplatSim 有 4 任务×40 trial 的 zero-shot 真机数字且有 Real2Real 上限对照——SB-Render-Lite 论文的真机协议至少要对齐这个规格。

---

# 精读二：RialTo — Reconciling Reality through Simulation: A Real-to-Sim-to-Real Approach for Robust Manipulation

## 基本信息

- 论文：Reconciling Reality through Simulation: A Real-to-Sim-to-Real Approach for Robust Manipulation（系统名 RialTo）
- 作者：Marcel Torne, Anthony Simeonov, Zechu Li, April Chan, Tao Chen, Abhishek Gupta*, Pulkit Agrawal*（MIT Improbable AI / UW，*equal advising）
- 发表：**RSS 2024**（web 复核于 2026-08-14：官方 proceedings https://roboticsproceedings.org/rss20/p015.pdf，DOI 10.15607/RSS.2024.XX.015，RSS 第 20 届，Delft，2024-07）；arXiv 2403.03949
- 链接：https://arxiv.org/abs/2403.03949 ｜ 项目页：https://real-to-sim-to-real.github.io/RialTo/
- 归类：real→sim 数字孪生 + teacher-student RL；robustify 真实 IL 策略；SB-Render-Lite 竞品（几何路线）。

## 一句话总结

用手机扫描 + 15 分钟 GUI 标注把目标真实场景变成带关节的几何数字孪生（USD），提出 inverse distillation 把 ~15 条真实演示"带着 privileged state"搬进仿真，在孪生里用 PPO+BC 稀疏奖励做大规模 RL 鲁棒化，最后 teacher-student 蒸馏（混合真实演示 co-training）回点云策略部署真机——8 任务上把 BC 基线在强扰动下的 5% 拉到 75%。

## 解决的问题

IL 策略对物体位姿变化、视觉干扰物、执行中物理扰动脆弱，除非演示量暴涨；真机 RL 不安全且慢；纯手工建仿真场景又太贵。RialTo 的定位与 SplatSim 相反而互补：SplatSim 解决"仿真图像不像真"，RialTo 解决"真实策略不鲁棒"——它不追求视觉保真（用点云绕开外观 gap），追求的是**几何/运动学保真**（数字孪生里能物理交互、能开抽屉），从而让 RL 能在"目标场景的复刻"里安全地学出恢复行为（重抓、重对准、抗推搡）。

## 方法核心：real→sim 数字孪生 + teacher-student RL

四步管线：

1. **Real-to-sim 场景重建**：手机/现成工具扫描（Polycam 适合大场景、AR Code 适合单物体、NeRFStudio nerfacto + Poisson 重建适合薄结构如碗架——附录 XII 给了工具选型表）→ 得到单块纹理 mesh → **自研 GUI 人工加工**：切分物体、加关节（抽屉/柜门/烤箱）、摆放组织 → 导出 USD 进 IsaacSim。物理参数**不辨识**，全场景统一默认值（质量 0.41 kg、摩擦 0.5、关节摩擦 0.1；碰撞体用 64 顶点/32 凸包分解，碗架等薄结构用 SDF 256 分辨率）。
2. **Inverse distillation（真→仿演示迁移，核心创新）**：真实演示只有 (点云观测, 动作)，没有 Lagrangian state（物体位姿），无法直接在仿真里做 state-based RL。做法：在 ~15 条真实演示上 BC 训练一个点云策略 \(\pi_{real}(a|o)\) → 把它**放进仿真**在渲染点云上 rollout → 收集成功轨迹，此时仿真天然知道 privileged state → 得到带 state 的仿真演示集 \(\mathcal{D}_{sim}\)。隐含假设：点云的 real→sim 视觉 gap 足够小，真实训练的策略在仿真里能偶尔成功（RGB 模态下这一步大概率不成立——这是 RialTo 选点云的深层原因之一）。
3. **RL 微调（privileged state）**：PPO + BC loss（对 \(\mathcal{D}_{sim}\) 的 log-likelihood 项，权重 0.1），**稀疏奖励**（每任务一个目标状态判据，如 drawer_joint>0.1，附录 VIII 全部列出），初始物体/机器人位姿域随机化。策略是 256×256 MLP；动作空间为 14 维离散化 delta 末端位姿（±3 cm × 6、±0.2 rad × 6、开/合夹爪）。演示的双重作用：(a) 解决稀疏奖励探索；(b) **把策略偏向物理可信、可迁移的行为**——从零 PPO 会利用仿真缺陷（如利用烤箱关节装配误差从底部顶开），这类行为真机必挂。
4. **Teacher-student 蒸馏 + 真实 co-training**：state teacher → 点云 student（ConvONet 式体素编码器：local PointNet + 3D U-Net + max/avg 池化 → 128 维，拼 9 维机器人状态 → MLP）。蒸馏数据配方（附录 IX-B）：15000 条全视角点云轨迹 + 5000 条真实相机视角轨迹 + 2000 条加干扰物轨迹 + **15 条真实演示**，各 1/4 采样；再做一轮 DAgger（1/3 各）。真实 co-training 是补 sim-to-real 残余 gap 的关键（见实验）。

## 输入假设与人力成本

- **一次场景扫描**（手机即可）+ **GUI 人工标注**：6 人用户研究（5 名无仿真经验）实测**人均 active 14 分 40 秒 / 总 25 分 12 秒**建成一个含切分物体+关节的场景；附录 XIII-A 给出线性 scaling law：total_active_time = 扫场景 3:14 + 4:50×N_物体 + 3:40×N_切分 + 2:54×N_关节。
- **~15 条真实演示**（单标定深度相机点云 + delta-EE 动作，键盘接口采集，30 分钟）。演示量是硬门槛：book-on-shelf 任务 0/5/10 条真实演示时 RL 全部 0%（inverse distillation 的真实策略进仿真不成功→无演示可收集→退化为从零 RL），15 条才到 90%；简单的开抽屉 5 条即可 89%。
- **传感假设**：标定深度相机（D455/D435 各一台/setup）；Franka Panda ×2 台。
- **计算成本**：RTX 2080/3090 单卡，端到端 **约 2 天 3 小时/任务**（inverse distillation 7h + RL 20h + 蒸馏 24h）——论文自认这使 continual learning 不可行。
- **不需要**：物理参数辨识、密集奖励工程、大规模真实数据采集。

## 真机实验协议与结果

- **协议**（值得 SB-Render-Lite 直接复制的口径）：8 任务（书上架、盘上碗架、杯上架、开抽屉、开柜、厨房烤箱、杯入垃圾桶、水槽取盘上架——后三个是 in-the-wild 场景）× **三级扰动**：L1 仅随机化物体/机器人位姿；L2 加视觉干扰物（杂乱摆放）；L3 执行中施加物理扰动（移动被操作物/目标位、推回抽屉、移机器人底座）。每方法**取最优 checkpoint，≥10 次 rollout，报 bootstrapped 标准差**。
- **主结果**：RialTo 三级平均 **91% / 77% / 75%**；BC(15 demos) 为 25% / 11% / 5%。总体表述为对基线 **>67%** 平均提升；in-the-wild 三任务对 IL 平均 +57%。
- **数据效率对照**（book-on-shelf）：BC 50 条演示（人力 1h45m）只到 40/30/20%；RialTo 15 条演示+15 分钟 GUI（人力约 45 分钟）达 90/70/60%——**约 2.5× 成功率、<1/2 人力**。
- **真实 co-training 消融**：难任务上（book/plate 带扰动）比仅用仿真演示 co-training 高 3.5×/2×；定性上真实 co-training 的策略更保守安全（抓书前留更大间隙）。说明残余的点云视觉 gap + 动力学 gap 由这 15 条真实数据兜底。
- **Real-to-sim 必要性消融（对 SB 叙事最重要）**：用 4 个 Objaverse 抽屉训练的多任务策略在真实目标抽屉上只有 **10%**，目标场景孪生训练达 **90%**——"通用资产 + 指望泛化"在场景级操作上远逊于逐场景重建。
- **RL from scratch**：3/5 任务 0%，烤箱 62% 但靠利用仿真关节误差、行为不可迁移。
- **RL from vision 不可行**：state-based 12h 到 96%，vision-based 35h 才 1%（点云渲染慢 10×、显存限制 batch 小 100×）——这解释了为什么必须发明 inverse distillation 走 state 路线。
- **仿真-真机相关性**：最终点云策略 sim 与 real 成功率接近（如 mug 72% sim vs 100% real，作者刻意把仿真调得比真实更难）。

## 局限性

论文自认 + 精读补充：

1. **依赖精确深度**：薄壁、透明、反光物体点云失效；RGB/RGBD 策略"未来工作"（框架不禁止但全文无 RGB 实验）。
2. **只支持可仿真、可资产化的任务/物体**：铰接刚体为主，deformable 出界；准静态任务、慢控制器（快动力学下默认物理参数会破功）。
3. **物理不辨识**：默认质量/摩擦 + 演示先验补偿是工程折中，复杂接触任务下会成为上限。
4. **每任务 ~2 天训练 + 每场景 15 分钟人工 GUI**：比 SplatSim 多了逐任务的 RL 大头；GUI 虽轻但仍是人在环。
5. 点云策略牺牲了 RGB 语义（颜色/纹理相关任务如判断成熟度不可做）——与 SplatSim 正好互补。

## 与 SB transport 路线的对照（本篇的行，详述版）

- **数据需求**：扫描×1 + 15 真实演示 + 深度相机。比 SplatSim 多演示、比 SB 多扫描与标注；SB 的 unpaired 图像采集无需演示成功与否的标签。
- **资产成本**：15 分钟/场景 GUI（有 scaling law，人力随物体/关节数线性涨）+ 2 天/任务训练。SB 无逐场景 3D 资产，但每个任务/域仍要真实图像样本与（可能的）transport 微调。
- **视觉保真上限**：不适用/刻意回避——RialTo 的答案是"换模态"。这提示 SB-Render-Lite 论文必须回答："当深度可用且任务不需要 RGB 语义时，为什么不直接学点云策略？" 合理回答：RGB-only 部署硬件更便宜、RGB 语义任务（颜色/纹理条件）点云做不了、透明/反光物体深度本身失效。
- **动力学一致性**：这是 RialTo 相对 SplatSim 和 SB 视觉 transport 的**独有增量**——它通过"在几何孪生里物理交互 + RL 扰动训练 + 真实演示先验"获得了对动力学扰动的鲁棒性（L3 75%），尽管没做系统辨识。SB 纯视觉路线在此维度为零，正面对比时必须把比较限定在视觉 gap 轴，或引入 `2602.23737`（BDGxRL，DSB 对齐 transition dynamics）作为 SB 家族在动力学轴的回应。
- **可扩展性**：逐场景专用化（Objaverse 消融是铁证），multi-task 版只是把多个单任务 teacher 蒸到一个 student，不产生跨场景泛化；SB 的摊销叙事在此有真实差异化空间。
- **真机证据强度**：8 任务 × 3 扰动级 × ≥10 rollouts + bootstrapped std + in-the-wild 场景，是本对照集中最完整的鲁棒性协议。

---

# 对照表：重建/渲染路线（real2sim）vs SB transport 路线

> 本表为 E09/E10 共建表的 E09 部分：给出 SplatSim、RialTo 两行 + SB-Render-Lite 参照行；LucidSim、X-Sim 两行由 E10 按同列口径补齐后合并。列口径 = 任务书五轴：数据需求 / 资产成本 / 视觉保真上限 / 动力学一致性 / 可扩展性，另加输入假设一句话与真机证据两列以便审稿对照。

| 系统（venue） | 输入假设一句话 | 真实数据需求 | 资产/人力成本 | 视觉保真上限 | 动力学一致性 | 可扩展性 | 真机证据 |
|---|---|---|---|---|---|---|---|
| **SplatSim**（ICRA 2025） | "给我一段含机器人的场景扫描 + 每物体扫描 + CAD，我还你 photorealistic 仿真渲染" | 静态场景多视角 RGB 扫描×1（含机器人 home 位）+ 每物体多视角扫描；**零演示、零深度、零真实交互** | 逐场景/逐物体扫描 + 手动分割机器人 Gaussians + ICP + 依赖 CAD/URDF；扫描时长未报告；演示端 3h(sim) vs 20.5h(real) | 高（PSNR 22.62/SSIM 0.78），几何一致性由刚体变换构造保证；但光照烘焙、无阴影/动态反射、非刚体（线缆/布料/液体）不可渲染；**残余 gap 需增广硬扛（无增广 21% vs 有 86.25%）** | 不处理：物理全交 PyBullet，无辨识无修正；渲染与物理解耦 | 逐场景逐物体线性成本；只刚体；策略视觉分布绑定被扫描场景 | 4 任务×40 trials，zero-shot 86.25%（Real2Real 97.5% / Sim2Sim 95.62% 三元对照）；UR5+2×D455，仅 RGB |
| **RialTo**（RSS 2024） | "给我一次手机扫描 + 15 分钟 GUI 标注 + 15 条真实演示，我还你抗扰动的点云策略" | 场景扫描×1（Polycam/ARCode/NeRFStudio）+ **~15 条真实演示**（点云+动作；难任务的硬门槛）+ 单标定深度相机 | GUI 标注 14m40s active/场景（6 人用户研究实测，线性 scaling law）+ 每任务 **~2 天 3 小时**训练（1×RTX 2080/3090）；物理参数用默认值不辨识 | 刻意回避：点云模态绕开外观 gap；纹理 mesh 只用于渲染仿真点云；RGB 策略未验证 | **部分处理（本表独有）**：孪生内物理交互 + RL 位姿随机化 + 物理扰动训练 + 真实演示 co-training 偏置可迁移行为；但无系统辨识，准静态/慢控制器限定 | 逐场景专用化（Objaverse 4 抽屉泛化仅 10% vs 目标孪生 90%）；multi-task 版不产生跨场景泛化；铰接刚体为主 | 8 任务×3 扰动级×≥10 rollouts+bootstrapped std：91/77/75% vs BC15 25/11/5%（>67% 提升）；含 3 个 in-the-wild 任务；Franka+单深度相机 |
| **SB transport / SB-Render-Lite（参照行，我方设想）** | "给我 sim 渲染样本 + unpaired 真实图像样本，我还你 real-style 训练数据" | 仿真渲染 marginal + **unpaired 真实域图像 marginal**（无需扫描/配对/标注；演示可选）；对新场景边际成本≈采一批真实图像 | 无逐场景 3D 资产；一次性 transport 模型训练 + 部署推理 NFE 成本（需 latent 化/蒸馏，见库内 G9/G12 缺口） | 上限=生成模型容量：可建模阴影/反射/非刚体外观等重建路线烘焙不了的效应；但**无几何硬约束**，存在语义/几何/action-label 漂移风险，需 GSBM 式 task-aware cost（`2310.02233`）或 keypoint/深度一致性正则 | 不处理（纯视觉 transport）；动力学轴需组合 policy 侧 OT co-training（`2509.18631`）或 dynamics bridge（`2602.23737`） | **跨场景/跨物体摊销**：一个 transport 模型可服务多场景；非刚体外观无表示层障碍；但每个新真实域仍需图像样本 | 待建：协议应对齐本表两行（≥3 元对照 + 多扰动级 + ≥10 rollouts + 效率/人力列） |
| LucidSim（CoRL 2024） | ——由 E10 补齐—— | | | | | | |
| X-Sim（CoRL 2025） | ——由 E10 补齐—— | | | | | | |

---

# 交叉洞察：这两篇对 SB-Render-Lite 意味着什么

1. **"逐场景专用化 vs 分布级摊销"是与 real2sim 路线正面交锋的主轴**。两篇的证据链一致：SplatSim 的策略绑定被扫描场景；RialTo 的 Objaverse 消融（10% vs 90%）从反面证明泛化式训练干不过目标场景重建。SB transport 若能证明"一个 transport 模型 + 若干真实图像样本"在多个场景上接近逐场景重建的成功率，就是清晰的差异化贡献；反之若 SB 也要逐场景微调，则叙事崩塌为"更贵的增广"。实验设计必须包含**跨场景摊销性**的直接测量（N 场景共享 transport vs N 次重建的成本-成功率曲线）。
2. **SplatSim 的增广消融（21%→86.25%）是 SB 路线最有力的外部论据**：photorealistic 重建之后仍有致命残余 gap（阴影/反射/非刚体），目前由无结构的通用增广硬扛。SB 可以主张两种站位：(a) **替代**——直接从 unpaired marginal 学 transport，免扫描免资产；(b) **组合（residual bridge）**——以 GS 渲染为 source marginal、真实图像为 target marginal 学一个"残余修饰"bridge（I²SB 式 informative prior 启动，见库内 `2302.05872`），把 21%→86.25% 那段增广换成有结构的 transport。组合站位审稿风险更低，且与 SplatSim 非互斥。
3. **RialTo 提醒：视觉 gap 可以被模态选择整个绕开**。SB-Render-Lite 论文必须显式回答"为什么不用点云"：合法场景是 RGB 语义任务（颜色/纹理条件，SplatSim 的水果成熟度例子）、无深度传感器的低成本部署、深度失效的透明/反光物体。同时 RialTo 的 inverse distillation 隐含依赖"点云 real→sim gap 小到真实策略能在仿真成功"——RGB 模态下这个前提不成立，**而 SB transport 恰好能把 RGB 的 real→sim 方向也桥起来**（把真实演示图像翻成 sim 风格以便在仿真里 relabel/评估），这是一个未被占据的接口点。
4. **真机评估协议直接抄作业**：SplatSim 的 Sim2Sim/Real2Real/Sim2Real 三元对照隔离迁移损失 + RialTo 的三级扰动 × ≥10 rollouts × bootstrapped std × in-the-wild 场景 + 两篇共有的人力小时数对照表。SB-Render-Lite 的实验表应至少含：三元对照、扰动分级、人力/资产成本列（我们的"免扫描"优势要量化成小时数才有说服力）、以及视觉指标（PSNR/SSIM/FID）与 policy success 的相关性报告（视觉指标服从 success，与库内 `synthesis.md` §4.3 口径一致）。
5. **动力学轴上三条路线全部不完整**：SplatSim 不碰（纯渲染）、RialTo 半碰（RL 鲁棒化但不辨识）、SB 视觉 transport 不碰。诚实的对照讨论应把"视觉 gap"与"动力学 gap"两轴分开画，SB 家族在动力学轴的回应是 BDGxRL（`2602.23737`）而非 SB-Render-Lite 本身——不要让 reviewer 误以为我们宣称视觉 transport 解决全部 sim2real。

---

# 并入主库建议

1. **INDEX.md**：建议新增分区"竞品系统：重建/渲染 real2sim 路线"，收录本文两篇精读行（SplatSim、RialTo），与 E10 的 LucidSim、X-Sim 合并为四条；分区导语一句话："SB transport 的主要竞争范式，related work 与实验对照必须正面回应"。
2. **synthesis.md**：§4.2 baseline 清单应增补"real2sim 重建渲染（SplatSim 式 GS 渲染）与数字孪生 RL（RialTo 式）"为**系统级对照**（区别于 CycleGAN/DR 等方法级 baseline）；§5 的立项表述可引用 Objaverse 消融与增广消融作为"分布级摊销 + residual bridge"主张的外部证据。
3. **对照表合并**：本文对照表与 E10 的 LucidSim/X-Sim 行按同列口径合并成完整五系统表后，可直接作为论文 related work 的底稿；建议合并版落在 `topics/` 下独立文件或并入 synthesis。
4. **后续精读优先级**（本文精读中发现、库内尚缺）：Maniwhere（CoRL 2024，SplatSim 引用的 RGB+深度大规模 RL 泛化对照）、NeRF2Real（ICRA 2023，SplatSim 增广方案来源、NeRF 渲染路线前驱）、Embodied Gaussians / Robo-GS（GS+物理的另两条支线）——均为 G1 线的二级补充，优先级低于 E10 两篇。
5. **档案口径**：两篇 venue 均已核验（SplatSim=ICRA 2025 任务给定已核验；RialTo=RSS 2024 经官方 proceedings + DOI 复核，检索日期 2026-08-14）；全文均经 arXiv 官方 HTML 精读（含 RialTo 附录 VIII–XIV 的任务/实现/硬件/用户研究细节），无二手转述。
