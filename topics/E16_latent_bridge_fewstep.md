# E16 扩充报告：Latent Bridge 与少步部署

> 扩充选题（对应 R09 缺口分析 G10/G16，检索与 venue 复核日期 2026-08-14）
> 精读：LBM（arXiv 2503.07535）、CDBM 方法章（arXiv 2410.22637）；收录：LSB（arXiv 2411.14863）、DBIM（arXiv 2405.15885）；决策笔记：ASBM / LBM / CDBM 三条加速路线对照。

## 选题定位

`SB-Render-Lite` 的部署形态受两个硬约束支配：真机相机流 10–50 Hz 的推理预算（NFE）与 224²–1024² 的分辨率（显存 / 吞吐）。库内已精读的 I²SB（`../reports/2302.05872_i2sb.md`）与 SB Flow（`../reports/2409.09347_schrodinger_bridge_flow_unpaired_translation.md`）解决了"transport 学什么"，但都默认几十到上百 NFE 的像素级采样；库中此前完全没有"latent 空间 bridge + 少步 / 一步推理"的条目（DBIM 仅有导航行）。本报告补上这条部署链：**latent 化（LBM）→ 免训练加速（DBIM）→ consistency 蒸馏（CDBM）→ 对抗式单阶段少步（ASBM）→ 零训练对照（LSB）**，并给出三条加速路线的裁决。

## TL;DR

1. **LBM（ICCV 2025 Highlight）证明：在 VAE latent 空间做 bridge matching，配合"离散 4 时间步训练 + 解码后 LPIPS 像素损失 + SDXL 初始化"，1 NFE 即可在 object removal / depth / normal / relighting 上打平或超过 50 NFE 的扩散基线**，训练成本仅 2×H100、20k 迭代。代价：训练需要预先给定的图像 coupling（配对 / 伪配对），且它只做一次 Markovian projection，不逼近 SB 最优 coupling。
2. **CDBM（NeurIPS 2024）给出把 consistency models 嫁接到任意 DDBM 式 bridge 的完整配方**：统一噪声调度设计空间（显式覆盖 Brownian bridge 与 I²SB 调度）、一阶 Exponential-Integrator bridge ODE solver、绕开 t=T 奇点的两步采样（最少 2 NFE），蒸馏（CBD）与微调（CBT）两范式，实测 4–50× 加速且 CBT 更稳更优。它是"先训好 bridge 再压缩"路线的标准工具，且**不改变 teacher 的 coupling**——与 SB 最优性的偏离由 teacher 决定。
3. **对 SB-Render-Lite 的裁决**：有伪配对（digital twin / 同 scene 双渲染）→ 直接 latent bridge（LBM 配方，1–4 NFE，最短路径）；纯 unpaired → 先用库内 SB Flow 在 latent 空间学 coupling 与 bridge，再依预算逐级压缩（DBIM 免费 10–25× → CDBM-CBT 到 2–4 NFE → 用 teacher 伪配对做 LBM 式一步化）。ASBM 的对抗式 D-IMF 理论上最忠实于 SB 且 unpaired 单阶段 4 NFE，但 GAN 稳定性与 adversarial shortcut 风险（呼应库内 R10-T7 警示）使其只宜作 NFE 对齐的对照基线。

---

## 一、精读：LBM — Latent Bridge Matching for Fast Image-to-Image Translation

### 基本信息

- 论文：LBM: Latent Bridge Matching for Fast Image-to-Image Translation
- 作者：Clément Chadebec, Onur Tasar, Sanjeev Sreetharan, Benjamin Aubin（Jasper Research）
- 会议：**ICCV 2025 Highlight**（venue 沿用库内 R09 于 2026-08-14 的 web 核验结论）
- 链接：https://arxiv.org/abs/2503.07535 （全文经 ar5iv HTML 精读）
- 代码：https://github.com/gojasper/LBM （官方开源）
- 归类：latent-space bridge matching；one-step image-to-image translation；蒸馏式训练技巧（非蒸馏架构）。

### 一句话总结

LBM 把 bridge matching 从像素空间搬进预训练 VAE 的 latent 空间，用"离散少数时间步训练 + 解码后 LPIPS 像素损失 + SDXL U-Net 初始化"三件套，使单次网络评估（1 NFE）的图像翻译在六类任务上达到或超过多步扩散 / flow matching 基线。

### 动机

- 条件扩散模型做图像翻译要几十上百步，蒸馏加速方法又多为 text-to-image 定制、难以做到满意的单步生成。
- Flow matching 已被广泛移植到翻译类任务，但其随机对应物 bridge matching 此前主要停留在低分辨率像素空间（作者点名 I²SB 系的局限），可扩展性与任务泛化未被验证。
- 目标：一个"任何成对翻译任务都能套、可上高分辨率、天生一步推理"的通用配方。

### 方法核心

**（1）bridge matching 底座。** 给定端点对 \((x_0, x_1)\)，构造 Brownian bridge 插值 \(x_t=(1-t)x_0+t x_1+\sigma\sqrt{t(1-t)}\,\epsilon\)，对应 SDE \(\mathrm{d}x_t=\frac{x_1-x_t}{1-t}\mathrm{d}t+\sigma \mathrm{d}B_t\)。训练即 Markovian projection：回归 drift，\(\mathbb{E}\|(x_1-x_t)/(1-t)-v_\theta(x_t,t)\|^2\)。\(\sigma=0\) 时退化为 flow matching（零噪声极限）。

**（2）latent 化。** 端点先经预训练 VAE 编码为 \(z_0,z_1\)，bridge 与 drift 回归全部在 latent 空间进行（式 4–5），latent 可离线预计算。这是可扩展到 1024² 的关键：网络在 \(\sim\!8\times\) 下采样的 latent 上运算。drift 网络 \(v_\theta\) 用 SDXL 预训练 text-to-image U-Net 初始化。条件版把条件（如光照图）在 latent 通道维拼接。

**（3）一步推理如何实现。** 这是本文最值得吃透的机制，由三个设计共同达成：

- *参数化角度*：由 drift 定义可反解目标 latent 预测 \(\widehat{z}_1=(1-t)\,v_\theta(z_t,t)+z_t\)（式 6）。在 \(t=0\) 处，\(v_\theta(z_0,0)\) 的回归目标期望是 \(\mathbb{E}[z_1|z_0]-z_0\)，因此**一步 Euler 从 0 积到 1 等价于网络直接输出条件期望 \(\mathbb{E}[z_1|z_0]\)**——1 NFE 的 LBM 本质上是一个带 bridge-matching 训练信号的回归器。这也解释了它为什么需要紧的 coupling：配对越紧、条件分布越单峰，一步回归越不糊。
- *时间步分布*：训练只在 4 个等距离散时间步采样 \(t\)（部分任务进一步偏置，如 depth 任务 \(\pi(t)=0.9\,\delta_{t=0}+\dots\) 把 90% 的容量压在一步推理的操作点上），并保证推理恰好用这些步。消融显示离散分布在对应步数上优于 uniform 训练，但把 NFE 上限锁死在 4（超过即性能骤降；uniform 训练则可继续加步提质）。
- *像素损失*：把 \(\widehat{z}_1\) 解码回图像与真值比较（LPIPS，权重 \(\lambda\approx5\text{–}10\) 最优），显著加速 domain shift 并锐化一步输出——作用类似蒸馏方法中的感知 / 判别损失，但无对抗组件。

**（4）随机性消融（对 SB 视角最有信息量）。** 小而非零的 \(\sigma\)（0.005–0.1，按任务）在 FID 上一致优于 \(\sigma=0\) 的 flow matching：随机 bridge 能覆盖更广的样本多样性；\(\sigma\) 过大则破坏源图信息、性能下降。这与库内 I²SB 报告"stochastic bridge 与 OT-ODE 是同一框架两个极限、按任务不确定性选择"的结论互为印证，也支持 R10-T8"SB 随机性可作数据增广"的评估设计建议。

### 实验（任务覆盖与速度）

- **任务覆盖**：object removal、monocular depth、surface normal、背景驱动 relighting（harmonization）、可控 relighting、可控 shadow generation 六类 + 附录 image restoration。所有任务同一配方，仅换数据与少数超参。
- **object removal**（RORD 52k 验证集，coarse mask）：1 NFE 的 LBM FID 26.29 / fMSE 1314.6 / PSNR 22.38，优于 50 NFE 的 SDXL-Inpainting（39.30）、PowerPaint（29.83）、Attentive Eraser（29.70）与 1 NFE 的 LAMA（30.03）；fine mask 下 FID 15.50 vs 次优 18.43。能连带移除阴影。
- **normal / depth（zero-shot）**：normal 在 NYUv2/ScanNet/iBims/Sintel 上平均排名 1.4（次优 Lotus-D 2.4、Diff-E2E-FT 2.8）；depth 五数据集平均排名 3.5，为全表最佳。**注意：这两类任务与机器人几何一致性直接相关**——同一配方既能做 real 化翻译，也能估计几何，为"翻译前后 depth/normal 一致性"检查提供了同源工具。
- **relighting**：自建 10k 真实测试集上 FID 12.79 优于 Harmonizer/PCT-Net/PIH/INR/IC-Light；可控版（光照图条件）仅用 Blender 合成数据训练即可泛化。
- **速度与训练成本**：所有主结果均为 **1 NFE**（1 次 U-Net + VAE 编解码）；训练 2×H100、20k–25k 迭代，bucketing 支持到 1024²。
- **合成数据比例消融（与 sim2real 直接相关）**：relighting 训练集中 Blender 合成数据占比升高时 FID 先降后升——合成数据帮助模型在受控场景学光照规律，但过多会损害输出真实感。这等于一个现成的 sim/real 数据配比实验模板。

### 局限性

- **需要预先存在的 coupling**（作者在结论中自述的首要局限）：所有任务的端点对都是构造出的语义配对（masked↔clean、image↔depth、composite↔relit）。纯 unpaired 的 sim2real 不能直接套用；形式上虽写作 \(\pi_0\times\pi_1\)，实际全部实验依赖配对。
- **不解 SB**：固定数据 coupling 上做一次 Markovian projection，无 IMF 迭代、无熵正则 OT 最优性主张；\(\sigma\) 只是超参而非熵正则旋钮的系统化使用。
- 离散时间步训练锁死 NFE 上限（≤4）；VAE 重建质量是上限；失败案例包括残留阴影 / 玻璃反射、合成数据导致的色偏与"塑料感"。
- 全部指标是视觉指标，无下游任务（对我们而言需补 policy success / geometry preservation 评估）。

### 对 SB-Render-Lite 的可借鉴点

- "离散时间步 + LPIPS 解码损失 + 强预训练初始化"三件套可直接移植到任何 latent bridge（包括 SB Flow 学出的 bridge 的一步化学生模型）。
- 1-NFE 输出 ≈ \(\mathbb{E}[z_1|z_0]\) 的视角给出清晰的适用判据：伪配对质量决定一步化是否可行。
- depth/normal 任务用同一配方训练，可作为翻译几何一致性的"同源探针"。

---

## 二、精读（方法章）：CDBM — Consistency Diffusion Bridge Models

### 基本信息

- 论文：Consistency Diffusion Bridge Models
- 作者：Guande He*, Kaiwen Zheng*, Jianfei Chen, Fan Bao, Jun Zhu（清华 / 生数科技；与 DBIM 同一团队）
- 会议：**NeurIPS 2024**（venue 沿用库内 R09 于 2026-08-14 的 web 核验结论）
- 链接：https://arxiv.org/abs/2410.22637 （方法章经 ar5iv HTML 精读）
- 代码：https://github.com/thu-ml/DiffusionBridge （与 DBIM 共用同一代码库）
- 归类：diffusion bridge 的 consistency 蒸馏 / 训练；few-step sampling。

### 一句话总结

CDBM 学习 DDBM 概率流 ODE 的 consistency 函数 \(h_\theta:(x_t,t,y)\mapsto x_\epsilon\)，通过统一 bridge 设计空间与专用一阶 ODE solver，把 consistency distillation / training 完整嫁接到任意 DDBM 式 bridge 上，用 2–4 NFE 达到 teacher 上百 NFE 的质量（4–50× 加速）。

### consistency 目标如何嫁接到 bridge（核心问题）

Consistency models 原生定义在"噪声→数据"的扩散 PF-ODE 上。搬到 bridge 有三个结构性障碍，CDBM 逐一给出解法：

**障碍 1：t=T 端点奇点。** bridge 的终端 \(x_T=y\) 是固定点，PF-ODE 在 \(t=T\) 处奇异、仅在 \(t\in[\epsilon, T-\gamma]\) 良定义。CDBM 用一个 Brownian bridge 显式例子证明：先从 \(T\) 用**一步随机 posterior sampling** 跳到 \(T-\gamma\)（该步需先用一次网络评估粗估 \(x_0\)，再从 \(q_{T-\gamma|0T}\) 采样），之后走 ODE 即可保持边缘分布。推论：**CDBM 的采样天然最少 2 NFE**（1 次粗估 + 1 次 consistency 评估），这是它与 LBM"真 1 NFE"的结构性差别；多轮"加噪-回跳"交替可进一步换质量。（更一般 bridge 的边缘保持严格论证由同组 DBIM 的非马尔可夫构造给出。）

**障碍 2：设计空间碎片化。** 各家 bridge（I²SB、DDBM-VP/VE、Bridge-TTS、Brownian bridge）调度、预测目标、precondition 各不相同，而原版 CM 推导绑死在 VE 调度 + Euler solver 上。CDBM 用四元组 \((\alpha_t,\bar\alpha_t,\rho_t^2,\bar\rho_t^2)\) 统一表述所有线性 drift bridge 的解析条件分布 \(q_{t|0T}=\mathcal N(a_t x_T+b_t x_0, c_t^2 I)\)，统一以 \(x_0\) 为预测目标、以 precondition（\(c_{\text{skip}}/c_{\text{out}}\)，在 \(t=\epsilon\) 处取 1/0）满足边界条件 \(h_\theta(x_\epsilon,\epsilon,y)=x_\epsilon\)。**论文 Table 1 显式覆盖 Brownian bridge 与 I²SB 调度**——意味着库内 I²SB 模型、以及 SB Flow 产出的 Brownian-bridge 型模型，形式上都在可嫁接范围内。

**障碍 3：solver。** consistency 目标的偏差由 ODE solver 的局部误差控制。CDBM 推导了通用调度下的一阶 Exponential-Integrator 型 bridge solver（Prop 3.1）：
\[x_r=\tfrac{\alpha_r\rho_r\bar\rho_r}{\alpha_t\rho_t\bar\rho_t}x_t+\tfrac{\alpha_r}{\rho_T^2}\big[(\bar\rho_r^2-\tfrac{\bar\rho_t\rho_r\bar\rho_r}{\rho_t})x_\theta+(\rho_r^2-\tfrac{\rho_t\rho_r\bar\rho_r}{\bar\rho_t})\tfrac{y}{\alpha_T}\big].\]
该 solver 本身就强过 DDBM 原 hybrid sampler（见实验），同时是连接两种训练范式的桥梁。

**两种范式：**

- **CBD（consistency bridge distillation）**：用预训练 bridge score \(s_\phi\) 走一步 ODE 得 \(\hat x_r\)，最小化 \(d(h_\theta(x_t,t,y),\,h_{\theta^-}(\hat x_r,r,y))\)。Prop 3.2 给出蒸馏误差上界 \(O((\Delta t_{\max})^p)\)（p 为 solver 阶数）。
- **CBT（consistency bridge training / fine-tuning）**：利用无偏 score 估计 \(\nabla\log q_{t|T}=\mathbb E[\nabla \log q_{t|0T}\,|\,x_t,x_T]\)，代入一阶 solver 后 \(\hat x_r\) 有闭式 \(\hat x_r=a_r y+b_r x+c_r z\)（与 \(x_t\) 共享同一 \(z\)），完全摆脱 teacher。Prop 3.3：\(\mathcal L_{\rm CBD}=\mathcal L_{\rm CBT}+o(\Delta t_{\max})\)。
- 训练 schedule：恒定 \(\Delta t\)（须精调，易不稳）或渐缩 \(t-r(t)\) 配 \(\lambda=1/(t-r)\)（更稳）。实测 **CBT 全面优于 CBD**、超参更皮实（呼应 ECT 的"微调优于蒸馏"结论），且省去存 teacher 的显存。

### 实验要点（作为方法章佐证，从简）

- Edges→Handbags 64²：CBT **NFE=2 FID 0.80**，优于 teacher（DDBM + 本文 ODE-1 solver）NFE=100 的 0.89 → ~50×。
- DIODE-Outdoor 256²：CBT NFE=2 FID 2.93 vs teacher NFE=100 的 2.57（轻微退化换 50×）。
- ImageNet 256² center-mask inpainting（I²SB 调度）：CBT NFE=4 FID 4.77 ≈ teacher NFE=10 的 4.81 → 此处仅 ~4×，因为一阶 EI solver 本身在 10 NFE 已很强。
- 保持生成多样性与语义插值能力（追踪采样噪声序列做插值）。

### 局限性

- consistency 类目标固有的数值不稳定性（作者自述）；CBD 对 \(\Delta t\) 极敏感。
- 实验止于 256² 像素空间，**latent 空间验证缺失**——"latent bridge + consistency 压缩"的组合是文献空白（对我们是机会，见决策笔记）。
- 蒸馏不改变 coupling：teacher 是 paired bridge（I²SB/DDBM）则学生也是；与 SB 最优性的差距完全由 teacher 决定，蒸馏本身再叠加 \(O(\Delta t^p)\) 偏差。
- 最少 2 NFE 的结构性下限；若硬性要求 1 NFE 需转 LBM 式配方或 CTM 类扩展。

---

## 三、收录条目

### 3.1 LSB — Latent Schrödinger Bridge: Prompting Latent Diffusion for Fast Unpaired Image-to-Image Translation

- arXiv 2411.14863（v1 2024-11-22）；作者 Jeongsol Kim, Beomsu Kim, Jong Chul Ye（KAIST）。
- **venue（2026-08-14 检索复核）：仍为 arXiv preprint**。web 检索与 Google Scholar 条目均未见主会收录记录（引用格式为 AAAI 模板但无接收信息），按库内纪律**标注 preprint**，引用时注明。
- 方法一段话：把 SB 概率流 ODE 的速度场解析分解为三项线性组合——noise predictor、target predictor、source predictor：\(v(x_t,t)=\tfrac{(1/2-t)\sqrt\tau}{\sqrt{t(1-t)}}\hat\epsilon+\hat x_1-\hat x_0\)（"目标域吸引 + 源域排斥 + 前半程加噪后半程去噪"）。Proposition 1 证明用**任意** coupling \(\Gamma_{01}\) 的预测器替代 EOT coupling，ODE 仍连接两个边缘（但中间边缘不再是 SB 边缘——即显式承认与 SB 最优性的偏离，换取可学习性）。实现上用单个 Stable Diffusion 1.5 在 latent 空间近似三个预测器：textual inversion 为每个域优化文本嵌入（每域 <1k 张无标注图即可）、SNR 换元 \(\bar\alpha_s=1/(\sigma_t^2+1)\)、\(y_s=x_t/\sqrt{\sigma_t^2+1}\) 对齐 bridge 状态与 VP 扩散训练输入、\(t<0.5\) 用源域 / 之后用目标域噪声预测器、CFG 等价于放大 \((\hat x_1-\hat x_0)\)、末尾补一步 denoising。
- 实验一句话：8 NFE 下 AFHQ Cat→Dog FID 113.2、Horse→Zebra 96.18、Dog→Wild 94.79，均优于 DDIB / PnP / SDEdit 同类零训练方法。
- 对 SB-Render-Lite 的定位：**几乎零训练的 unpaired latent 翻译对照组**（与 SDEdit/DDIB 同组、但 NFE 低一个量级），适合作为"不训练任何 bridge 时能到什么水平"的下界；其 SB-ODE 分解也是很好的教学式参照。局限：质量上限受 SD 先验对机器人工况（工业相机、机械臂特写）覆盖度的限制；Prop 1 表明它不比库内 SB Flow 更接近 SB 最优。

### 3.2 DBIM — Diffusion Bridge Implicit Models

- arXiv 2405.15885；作者 Kaiwen Zheng*, Guande He*, Jianfei Chen, Fan Bao, Jun Zhu（清华 / 生数；CDBM 同组前作）。
- **venue（2026-08-14 web 复核）：ICLR 2025 Poster 确认**——OpenReview `forum?id=eghAocvqBk` 标注 "ICLR 2025 Poster"，ICLR 2025 官方 proceedings PDF 在册；与库内 R05 的 DBLP 记录（conf/iclr/ZhengH0B025a）一致。库内 `deep_research_learning_resources.md` 已有导航行（A 级），本条升级为收录条目。
- 方法一段话：把 DDBM 前向过程在采样时间格点上推广为一族**非马尔可夫 bridge**（方差参数 \(\rho\) 控制），保持边缘分布与训练目标不变，因此**免训练**重用任何已训 bridge score；\(\rho\to0\) 得到确定性隐式采样，正是 DDIM 对 DDPM 之推广在 bridge 上的对应物；连续极限诱导一个比 DDBM PF-ODE 更简洁的新 ODE 并支持高阶 solver。因 \(t=T\) 奇点，首步强制注入 booting noise——将其视作 latent 变量即可做忠实编码 / 重建 / 语义插值。
- 数字：较 DDBM vanilla sampler 最高 **25×**；ImageNet 256² inpainting：DDBM 100 NFE FID 6.46 vs DBIM 10 NFE 4.51（三阶 solver 4.34）。
- 对 SB-Render-Lite 的定位：**任何已训 bridge 的零成本加速层**——在做任何蒸馏之前先套 DBIM（NFE 降到 ~10），离线批量翻译往往就够；同时其非马尔可夫构造是 CDBM 边缘保持论证的理论依托，二者构成"免训练 → 再训练"的同源两级。库内 R07 已核验其"最高 25× 加速"描述无误。

---

## 四、决策笔记：三条少步部署路线（ASBM 对抗蒸馏 / LBM latent 化 / CDBM consistency 蒸馏）

ASBM（arXiv 2405.14449，NeurIPS 2024，venue 沿用 R09 2026-08-14 核验；本次经 arXiv abs 页核对方法与数字）：D-IMF 把 IMF 的连续时间 Markovian projection 替换为**少数离散时刻的转移核学习**，用 DD-GAN 对抗式实现，从 minibatch OT coupling 起步迭代；理论上 D-IMF 不动点是 SB 在离散时刻的解。unpaired、单阶段、4 NFE。

### 4.1 对照表

| 维度 | ASBM（对抗 D-IMF） | LBM（latent 化一步桥） | CDBM（consistency 蒸馏 / 微调） |
|---|---|---|---|
| 训练复杂度 | **高**：D-IMF 外层迭代 × 内层 DD-GAN（G 42M + D 27M）对抗训练；CelebA 级 1M 步；GAN 调参与稳定性风险 | **低**：单阶段回归式训练，无对抗、无迭代；2×H100 20k it；需预训练 VAE（+可选 SDXL 初始化） | **中**：两阶段——先训 bridge teacher（DDBM/I²SB/SB Flow），再 CBD/CBT；CBT 是微调、成本低但 consistency 目标有数值不稳定性 |
| 最终 NFE | 4（实测最优操作点；NFE=8 反而退化到 FID 55.72，伸缩性差） | **1**（离散步训练可扩到 2–4；超过 4 性能骤降） | 2–4（结构性下限 2：首步须随机跳过 t=T 奇点） |
| 质量保持 | CelebA 128² unpaired（female→male，ε=1）：4 NFE FID 16.86 vs DSBM ~百步 24.06——**超过** teacher 级 IMF（male→female 方向 NFE=4 为 16.62，量级一致）；但依赖 FID 类指标，对抗输出的多模态覆盖存疑 | 多任务 1 NFE 打平 / 超过 50 NFE 扩散基线（RORD FID 26.29 vs 29.7–39.3；normal 平均 rank 1.4）；1024² 已验证 | NFE=2 逼近 teacher NFE=100（E2H 0.80 vs 0.89，DIODE 2.93 vs 2.57）；ImageNet 级 4×；256² 像素空间已验证、latent 未验证 |
| 与 SB 最优性的偏离 | **理论最小**：目标即 SB（离散时间投影）；实际偏离来自 GAN 近似误差与有限迭代，难以量化 | **最大且在 coupling 层面**：固定给定 coupling 的单次 Markovian projection，不逼近 EOT coupling；若任务本有语义配对，这是利用监督而非缺陷 | **保序压缩**：不改变 teacher 的 coupling / 边缘，仅加 \(O(\Delta t^p)\) 蒸馏误差；与 SB 的距离 = teacher 与 SB 的距离 |
| 数据要求 | unpaired 即可（两域样本） | **需要 coupling**（paired / 伪配对） | 继承 teacher：I²SB teacher 需 paired；SB Flow teacher 可 unpaired（蒸馏对由 teacher 采样生成） |
| 主要风险 | GAN 不稳定 + adversarial shortcut（R10-T7 已警示视觉 realness 判别器可能牺牲几何 / 语义正确性）；分辨率验证止于 128² | 一步输出 ≈ 条件期望，多模态被平均化（LPIPS 缓解不消除）；SB 随机增广优势丢失 | CBD 超参敏感；两阶段误差叠加；latent 空间组合未经文献验证 |

### 4.2 三条路线的结构性理解

一条统一的主线可以把三者串起来：**少步化的本质是"把 coupling 的决定与 transport 的压缩分离"**。

- ASBM 把两件事耦合在一次对抗训练里（边求 SB coupling 边少步化），理论最优雅、工程最脆弱；
- LBM 假设 coupling 已给定（数据配对），只做压缩，因此最简单最快，但对 unpaired 无解；
- CDBM 显式两阶段：teacher 决定 coupling（可以是 SB 的近似解），学生只负责压缩且保序。
- 补充两级：DBIM 是"零训练压缩"（10–25×，NFE~10），LSB 是"零训练 + 零 bridge"下界（8 NFE，借 SD 先验）。

### 4.3 给 SB-Render-Lite 的部署建议（含与库内 I²SB / SB Flow 的衔接）

**判据先行：有没有（伪）配对？**

**情形 A：可构造伪配对**（digital twin 同 scene 双渲染、SplatSim 式重建-渲染、或 I²SB 场景的合成退化）→ **直接 latent bridge，跳过 SB 迭代**。

- 用 LBM 配方在 VAE latent 上训练 sim→real 翻译：离散 4 时间步（质量偏置在 t=0）、LPIPS 解码损失 \(\lambda\approx10\)、\(\sigma\in[0.005,0.1]\) 扫描、预训练 U-Net 初始化；1–4 NFE 部署。
- 库内 I²SB 保留为像素级 paired 基线（LBM 可视为 I²SB 思路的 latent + 一步化现代版）；对照实验应含"I²SB + DBIM（免训练 10 NFE）"与"I²SB + CDBM-CBT（2–4 NFE）"两条平行压缩线，以分离"latent 化收益"与"少步化收益"。
- 理由：1 NFE 打平 50 NFE 的实证 + 最低训练成本 + 1024² 可扩展性，是到部署的最短路径；SB 最优性在此情形没有额外价值——配对本身就是比 EOT coupling 更强的监督。

**情形 B：纯 unpaired**（真实机器人帧无对应 sim 帧）→ **先训好 bridge，再逐级压缩**（不建议"直接 latent bridge"，因 LBM 的一步回归在独立耦合下会把多个 real 模式平均化）。

1. **Coupling / bridge 阶段**：用库内 SB Flow（α-DSBM）在 VAE latent 空间训 sim↔real bridge——它决定 coupling、最接近 SB 最优、保留熵正则 \(\epsilon\) 作为 realism vs 结构保持的旋钮（呼应 I²SB 报告的建议）。
2. **免费加速**：直接套 DBIM 隐式采样 + 高阶 solver（无需再训练，10–25×，NFE≈10）。离线批量翻译（为 policy 训练生成 real 化数据集）到这一步通常已够——这是**优先级最高、风险最低的动作**。
3. **在线闭环需要 2–4 NFE**：CDBM-CBT 微调。CDBM 的统一设计空间显式含 Brownian bridge 调度，形式上可直接嫁接 SB Flow 产物；蒸馏数据用 teacher coupling 采样的 (sim, real-like) 合成对。注意这是文献空白区（CDBM 只在像素空间验证过），需预留调试预算，也是可发表的方法贡献点。
4. **硬性 1 NFE**：用 teacher 采样构造伪配对，再走情形 A 的 LBM 配方一步化（bridge 版的"先采样后回归"蒸馏）。接受额外一层近似。
5. **ASBM 定位为对照基线而非主线**：它是唯一 unpaired 单阶段 4 NFE 方案，理应进入 NFE 对齐的对比表；但 GAN 稳定性、NFE=8 退化暴露的伸缩性问题、以及 R10-T7 对 adversarial shortcut 破坏 geometry/action 保持的警示，使其不适合承载主结果。

**通用注意事项：**

- **蒸馏后必须复测结构指标**：keypoint/depth/inverse-dynamics 一致性与下游 policy success（库内评估协议），任何一步化都可能以"平均化"或"shortcut"的方式伤害 action label 有效性，而 FID 看不出来。
- **随机增广的取舍**：R10-T8 指出 SB 一对多随机翻译可做 K-变体增广；确定性 1-NFE 学生丢失该能力。可采用双模式：**训练期**（离线生成 policy 训练数据）用 teacher 或 \(\sigma>0\) 多步 SDE 采样保留多样性，**部署期**（在线闭环）用 1–4 NFE 学生。LBM 的 \(\sigma\) 消融正好支持小随机性有益的判断。
- **延迟预算 sanity check**：LBM 用 SDXL 级 U-Net 在 1024² 下 1 NFE；机器人相机流 224–448² 对应 latent 28²–56²，1 次中型 U-Net + VAE 编解码在单卡上稳入 10–50 Hz 预算，还有换小 backbone 的余量。NFE 之外，VAE 编解码占比在小分辨率下不可忽略，工程上应计入端到端延迟。
- **数据配比**：LBM 的合成数据比例消融（过多合成损害 realism）可直接移植为 sim/real 混合训练的配比扫描模板。

---

## 并入主库建议

1. **文件拆分**：合并入主库时建议把本报告拆为 `reports/2503.07535_lbm.md`（逐篇精读）与 `reports/2410.22637_cdbm.md`（方法章精读）两个独立页面，沿用 I²SB 报告的模板；LSB、DBIM 两条收录条目并入 `deep_research_learning_resources.md` 的对应表格（DBIM 已有导航行，补"收录条目见本报告"指针即可，勿重复建页）。
2. **INDEX 更新**：`reports/INDEX.md` 建议新增小节「部署效率：latent bridge 与少步采样」，收 LBM / CDBM 两页并在导语注明与 I²SB（paired 基线）、SB Flow（unpaired 底座）的衔接关系；「对当前 SB-Render-Lite 的直接启发」段落可补一句"部署链条：SB Flow → DBIM（免训练）→ CDBM-CBT（2–4 NFE）→ LBM 式一步化"。
3. **venue 记录**（写入 metadata 时采用）：LBM = ICCV 2025 Highlight、CDBM = NeurIPS 2024、ASBM = NeurIPS 2024（三者沿用 R09 2026-08-14 核验）；DBIM = ICLR 2025 Poster（本次 2026-08-14 OpenReview + ICLR proceedings 复核）；LSB = arXiv preprint（本次 2026-08-14 复核未见主会收录，引用需标 preprint 并注明检索日期）。
4. **勿重复立项**：ASBM 与 LightSB 的方法章精读属 G1 求解器家族任务（R09 建议清单第 3 条），本报告仅在决策笔记维度核对了 ASBM 的摘要与关键数字，不替代其精读。
5. **建议新增实验项**（写入实验计划时）：(a) "SB Flow + DBIM"vs"SB Flow + CDBM-CBT"vs"伪配对 + LBM"的 NFE–质量–policy success 三方曲线；(b) 一步化前后 keypoint/depth/inverse-dynamics 一致性回归测试；(c) \(\sigma\) 与 K-变体增广对下游成功率的边际收益（呼应 R10-T8）。
6. **开放机会点**：latent 空间的 consistency bridge（LBM 的 latent 化 × CDBM 的保序压缩）在文献中尚无验证，若 SB-Render-Lite 走情形 B 路线，该组合本身即可作为论文的 efficiency 章节贡献。
