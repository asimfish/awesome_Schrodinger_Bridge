# E17：Zero-shot 翻译基线（DDIB 精读 + SDEdit 收录）

> 扩充研究员：E17｜日期：2026-08-14｜对应缺口：内部审查 R09 缺口 G11（P1）
> 精读 1 篇：DDIB（arXiv 2203.08382，ICLR 2023）；收录 1 篇：SDEdit（arXiv 2108.01073，ICLR 2022）。
> 全文获取方式：两篇均经 arXiv 官方 HTML 全文读取（正文 + 全部附录），无缺页；venue 经 OpenReview / 项目主页 / GitHub 三方 web 复核，检索日期 2026-08-14。

## 选题定位

DDIB 与 SDEdit 是 **不训练任何跨域模型** 的扩散域翻译方法：DDIB 用两个各自独立训练的单域扩散模型经 Gaussian latent 中转（确定性双段 ODE），SDEdit 只用一个目标域扩散模型做"加噪—去噪"（随机单段 SDE）。对 SB-Render-Lite 而言，它们是实验矩阵里 **成本最低的零训练（零 bridge / 零配对）对照组**，也是"为什么需要真正的 SB 耦合"这一论证的 **反面参照**：DDIB 论文自己就把方法解释为"两段退化 Schrödinger Bridge 的拼接"，因此 SB-Render-Lite 相对 DDIB 的增益，恰好就是"**跨域耦合优化本身**"的净价值——两者共享全部其余组件（扩散骨干、latent 空间、数据）。

## TL;DR

1. **DDIB = 两段退化 SB 的 Gaussian-pivot 拼接**：每段是"数据↔N(0,I)"的 SB 在参考过程恰好满足终端边缘时的退化极限（KL 最小值为 0，最优路径测度＝参考过程本身，前向 policy 为零）；此时 SB 的 probability-flow ODE 与 SGM 的 PF ODE 逐点相等（DDIB Prop 3.2，理论基础为库内已覆盖的 SB-FBSDE，ICLR 2022）。
2. **DDIB 的最优性只到段内、且只到边缘层面**：(i) 它走的是 PF-ODE 确定性 flow map，而退化 SB 的最优耦合是随机的（两者只共享边缘路径）；(ii) 即便问"flow map 是不是段内 OT map"，一般也是否定的（Lavenant–Santambrogio 2022 反例；高斯情形成立，Khrulkov et al. ICLR 2023）；(iii) 最关键的：即使每段都是精确 OT，**复合 sim→N(0,I)→real 一般也不是 sim→real 的 OT**（d≥2 时两个 Brenier map 的复合一般不再是凸势梯度；d=1 是唯一例外）。DDIB 的目标里从未出现任何跨域 cost——sim-real 对应完全是两个单域模型 latent 组织方式的副产品。
3. **Exact cycle consistency ≠ 对齐**：DDIB 的往返恒等来自 ODE 可逆性，与耦合质量无关；"可逆但语义漂移"是其对机器人数据最危险的失败模式（物体几何、接触状态、任务阶段被 OT mass-moving"搬走"），论文附录 C 自认这类失败案例。
4. **SDEdit 是更便宜的"半段"基线**：只需真实域一个模型（可直接用预训练底模，即 Stable Diffusion img2img 的原型），faithfulness 只有概率上界（Prop 1），由 t0∈[0.3,0.6] 调节 realism–faithfulness 权衡；无 cycle consistency，语义漂移风险更高。
5. **对照组规格**（§4）：B0=identity、B1=SDEdit、B2=DDIB（含 partial-depth 变体，恰为 B1↔B2 插值），与 SB 方法共享 VAE/骨干/数据/NFE 预算；主指标沿库内口径用 real-domain policy success + 语义漂移率，不以 FID 为主。若零训练基线已收回大部分 sim2real 差距，SB 耦合必须以剩余差距与漂移率改善来论证其训练成本。

---

## 1. 精读：Dual Diffusion Implicit Bridges（DDIB）

### 1.1 元信息

- 标题：Dual Diffusion Implicit Bridges for Image-to-Image Translation
- 作者：Xuan Su（Stanford）、Jiaming Song（NVIDIA）、Chenlin Meng（Stanford）、Stefano Ermon（Stanford / CZ Biohub）
- arXiv：https://arxiv.org/abs/2203.08382 （v1 2022-03-16）
- **venue：ICLR 2023**（核验：OpenReview `5HLoTvVGDe` PDF 页眉 "Published as a conference paper at ICLR 2023"；项目页 suxuann.github.io/ddib 标 "In ICLR 2023"；GitHub `suxuann/ddib` README 同；检索日期 2026-08-14。注意官方 bibtex key 为 `su2022dual` 但 `year={2023}`，引用时以 2023 为准）
- 代码：https://github.com/suxuann/ddib
- 归类建议：`zero_shot_translation_baseline`（零训练扩散域翻译 / 退化 SB）

### 1.2 动机

无配对图像翻译的主流方法（CycleGAN 系、AlignFlow 等 flow 系）依赖 **源-目标联合训练**，带来三个结构性问题：

1. **域对专属**：模型绑定特定 (source, target) 对，N 个域两两翻译需要 O(N²) 个模型；StarGAN 式共享域又会引入信息瓶颈。
2. **隐私/数据隔离**：训练需同时访问两域数据（论文以跨医院医学影像为例）。
3. **不可增量扩展**：新域对必须重训。

DDIB 的回答：每个域独立训练一个扩散模型；翻译时 source 模型把图像沿 PF ODE 正向积分到 Gaussian latent，target 模型再反向积分回图像。训练完全解耦（O(N) 个模型，甚至一个条件模型即可），翻译时只传 latent。对 sim2real 语境，"隐私"卖点无关紧要，但 **模块复用** 直接相关：真实域模型只训一次，任意新的仿真渲染变体（换纹理/光照/渲染器）都无需重训真实侧。

### 1.3 方法核心

#### 1.3.1 两段 ODE

记 \(v_\theta\) 为由 score 网络定义的 PF-ODE 速度场（\(\mathrm{d}\mathbf{x} = [\mathbf{f} - \tfrac{1}{2}g^2 \nabla_x \log p_t]\,\mathrm{d}t\)），`ODESolve(x; v, t0, t1)` 为其数值积分（实现用 DDIM，一阶；作者明确可换 DPM-Solver/Heun 等高阶求解器）。整个算法只有两行：

```
x_latent = ODESolve(x_sim;  v_sim,  0, 1)   # 源域确定性编码到 N(0,I)
x_real   = ODESolve(x_latent; v_real, 1, 0)  # 目标域确定性解码
```

无任何跨域训练、无配对数据、推理确定性（同一输入恒得同一输出）。

#### 1.3.2 Exact cycle consistency（Prop 3.1）

由于两段都是 ODE，翻译 sim→real→sim 在零离散误差假设下 **精确恒等**；实践中误差仅来自 ODE 求解器离散化。2D 合成实验（Moons/Checkerboards/Concentric & Parallel Rings/Squares 六个数据集，标准化到单位方差）中往返 L2 误差 0.0065–0.0143，几乎可忽略。对照：CycleGAN 需要显式 cycle loss 且只能近似满足。**注意这是"映射可逆性"的结果，与翻译质量/耦合最优性逻辑上独立**（见 1.3.3 L3 的推论）。

#### 1.3.3 与 SB 的关系推导：DDIB 在什么极限下是 SB，耦合最优性差在哪

这是本条目对 SB-Render-Lite 最重要的部分，论文只给了骨架（Prop 3.2 + 引用 SB-FBSDE），下面把推导链补全，并指出三层"最优性缺口"。

**(a) 背景：SBP 与熵正则 OT。** SBP 在给定两端边缘 \(p_0, p_1\) 与参考路径测度 \(W\)（此处取 VE/VP 扩散过程）下求 \(P^\star = \arg\min\{ \mathrm{KL}(P\|W) : P \in \mathcal{D}(p_0,p_1)\}\)。其静态投影等价于以参考转移核为 cost 的熵正则 OT（Brownian 参考、扩散强度 ε 时即 quadratic cost + ε·熵；ε→0 收敛到 Monge OT）。这与库内 `2409.09347`（SB Flow）报告的口径一致。

**(b) 退化极限：SGM 是 SB 的特例。** 取 \(p_0 = p_\text{data}\)，\(p_1 = \mathcal{N}(0,I)\)，\(W\) = VP/VE 前向加噪过程。SB-FBSDE（Chen–Liu–Theodorou，ICLR 2022，arXiv 2110.11291；库内 Liu 专题已覆盖）证明：当前向/后向 policy 取
\[(\mathbf{z}_t, \hat{\mathbf{z}}_t) = (0,\; g(t)\nabla_x \log p_t(x))\]
时 SB 的对数似然与 SGM 的完全相等，而这组 policy 恰在 **参考过程的终端边缘本身就等于 prior** 时达到。直观解释：若加噪过程在 t=1 已经把 \(p_\text{data}\) 送到 \(\mathcal{N}(0,I)\)（VE 需 σ_max→∞、VP 需 ∫β dt→∞；有限步长下近似成立），则 \(W\) 本身就落在可行集 \(\mathcal{D}(p_0,p_1)\) 里，KL 最小值为 0，**SB 最优路径测度＝参考过程本身**，前向不需要任何额外 drift（z=0）——SB "退化"成普通 score-based diffusion。DDIB Prop 3.2 进一步验证：把这组 policy 代入 SB 的 PF ODE（\(\mathrm{d}\mathbf{x} = [\mathbf{f} + g\mathbf{z} - \tfrac{1}{2}g(\mathbf{z}+\hat{\mathbf{z}})]\mathrm{d}t\)）后逐项化简恰为 SGM 的 PF ODE（论文附录 D，一行代入即得）。

**(c) 结论（论文声明）**：DDIB = "source↔latent" 与 "latent↔target" **两个退化 SB 的拼接**，拼接之所以良定义，是因为两段共享同一个 pivot 边缘 \(\mathcal{N}(0,I)\)。论文据此称 DDIB "intrinsically entropy-regularized optimal transport"、"the most OT-efficient translation procedure"。

**(d) 批判：耦合最优性到底差在哪（三层）。**

- **L1（段内：路径测度 vs 耦合）**。退化 SB 的最优 *耦合* 是参考过程的随机联合分布 \((x_0,\, x_1 = \sqrt{\bar\alpha}\,x_0 + \sigma z)\)；而 DDIB 走的是 PF-ODE 的 **确定性 flow map**。两者共享全部边缘 \(\{p_t\}\)，但联合分布不同。所以严格说，DDIB 每段"沿退化 SB 的概率流走"，而 **不是** 实现了该段的熵正则 OT plan。
- **L2（段内：flow map ≠ Brenier map）**。退而求其次问：这个确定性 flow map 是否至少是"data↔Gaussian"的（ε→0 意义下）OT map？答案：**高斯分布时精确成立**（Khrulkov, Ryzhakov, Chertkov, Oseledets, *Understanding DDPM Latent Codes Through Optimal Transport*, ICLR 2023, arXiv 2202.07477，并给出一般情形的猜想与高精度数值支持），但 **一般情形被证伪**——Lavenant & Santambrogio（*The flow map of the Fokker–Planck equation does not provide optimal transport*, Applied Mathematics Letters 133:108225, 2022）构造了显式反例（沿袭 Kim–Milman 2012 的怀疑与 Tanana 2021 的部分反例）。实用注脚：反例中两映射差异只在个别点、数值上 flow map "几乎最优"，所以 DDIB 的 OT 解释作为 **近似直觉** 仍可用，但"most OT-efficient"不能当定理引用。
- **L3（跨域：复合不最优——本质缺口）**。即便把 L1/L2 都让掉、假设每段是精确 OT map \(T_s: p_\text{sim}\to\gamma\)、\(T_r: p_\text{real}\to\gamma\)，DDIB 的跨域映射 \(T_r^{-1}\circ T_s\) **一般也不是** \(p_\text{sim}\to p_\text{real}\) 的 OT map：
  - d=1 是唯一的幸运情形：单调映射的复合仍单调，而 1D 中把 μ 推到 ν 的单调传输映射唯一且恰为 quadratic-cost OT map，故 pivot 复合＝OT。
  - d≥2：Brenier 定理下 OT map 必须是凸势梯度 \(\nabla\varphi\)；\(T_r^{-1}\circ T_s = \nabla\varphi_r^{*}\circ\nabla\varphi_s\)（* 为凸共轭）一般 **不再是** 任何凸函数的梯度，因此不解 sim→real 的 Monge 问题。
  - 更根本地：DDIB 的构造中 **从未出现任何跨域代价** \(c(x_\text{sim}, x_\text{real})\) 或跨域 KL。sim-real 的对应关系完全由两个单域模型各自把数据"编排"进 \(\mathcal{N}(0,I)\) 的方式隐式决定——没有任何变分原理约束这个对应。真 SB 方法（DSB / SB Flow `2409.09347` / paired 情形的 I²SB `2302.05872`）直接解 SBP\((p_\text{sim}, p_\text{real})\)，最优性以跨域熵正则 cost 表达，这正是 DDIB 缺失的东西。
  - **推论**：exact cycle consistency 与耦合最优性正交。任何可逆映射（复合两个可逆 flow）都自动 cycle-consistent；可逆保证"不丢信息"，不保证"对齐语义"。

**(e) 何时 pivot 复合近似可靠。** 若两域共享同一骨干（DDIB 的 ImageNet 实验实际用 **同一个** 预训练 ADM + classifier guidance，把 1000 个类当 1000 个"域"），latent 组织天然共享，语义对应良好（狮子→各种动物仍保留吼叫姿态）；若两域 **独立训练** 且分布差异大（论文附录 C：鸟→狗），posture 等语义即漂移。对 SB-Render-Lite 的直接启示：**sim/real 用 domain-label 条件化的单模型实现 DDIB，是稳住这条基线的关键设计**（也让"SB 增益"的结论更保守可信，见 §4）。

### 1.4 实验

- **2D 合成**（六个 2D 数据集两两翻译）：颜色拓扑平滑保持；cycle 往返 L2 0.0065–0.0143（单位方差数据），验证 Prop 3.1。
- **Example-guided color transfer**（逐图训模型，在归一化 RGB 空间）：DDIB 结果与经典 OT 求解器逐像素 MSE 十分接近——EMD 0.0337/0.0293、Sinkhorn 0.0281/0.0326、linear/Gaussian mapping 0.035–0.075（两张目标图）。这是"DDIB 行为近似 OT"的最直接实证（也符合 L2 层"数值上几乎最优"的判断）。
- **Paired 定量评测**（Facades、Maps，MSE，归一化 [-1,1]）：Facades A→B 上 DDIB 0.5312 优于 CycleGAN 0.7129 / AlignFlow 0.5801；Maps A→B 0.0194 亦最优；但两个 B→A 方向（分割→照片）都劣于基线（0.3946 vs 0.2512、0.1302 vs 0.0897）。注意：为跑通该评测，作者先用两域各 ~1000 像素算了一个 Sinkhorn 色彩对应，对分割图做色彩预变换（附录 E.1）——**这实际上注入了少量跨域信息**，纯零耦合声明在此实验上要打折扣；同时说明当两域色彩/统计差异大（传输成本高）时裸 DDIB 会退化。
- **Class-conditional ImageNet 256×256**：用 Dhariwal & Nichol 预训练 ADM（每类约 1000 张训练图）+ classifier guidance，单模型即当 1000 个域模型；跨类翻译保留姿态/表情；多域翻译无需任何微调。

### 1.5 局限（论文自认 + 本报告补充）

1. **语义漂移＝无耦合约束的直接后果**（附录 C 自认"feature and limitation"）：OT mass-moving 行为把源图内容"搬运"为目标域中传输距离最近的内容；域差大时（鸟→狗）姿态不保。机器人语境下更致命——被搬走的可能是 **物体几何、接触状态、任务阶段**，视觉更真实但动作标签失效。库内口径（`2409.09347` 报告、INDEX）早已指出这类翻译需附加 DINO/CLIP 语义、depth/keypoint、temporal、inverse-dynamics 约束；DDIB 连跨域 cost 都没有，属于该谱系的最弱端。
2. **两个 B→A 方向失败** 提示：当目标域信息量大于源域（分割→照片）时，无耦合的 Gaussian pivot 无法补出正确细节。sim→real 恰可能属于这种"上采样语义"的方向（真实域纹理/光照信息更丰富），需在对照组实验里重点观察。
3. **计算成本**：encode+decode 两次全 ODE 积分（DDIM 数十到数百步 ×2）；color transfer 应用甚至逐图训模型。逐帧独立处理，无时序一致性——对视频/轨迹数据是硬伤。
4. **确定性是双刃剑**：无随机性 → 不能给出多样翻译候选，也无法表达 sim→real 的本征多义性（同一渲染可对应多种真实光照）。
5. 论文未提供 FID 级别的大规模无配对定量对比（主要靠 paired MSE 与定性图），基线覆盖（CycleGAN/AlignFlow）按 2026 标准偏旧；作为"对照组"使用时应自行补齐评测。

### 1.6 对 SB-Render-Lite 的意义

DDIB 与 SB Flow/I²SB 共享扩散骨干与 latent 空间，唯一差别是"是否优化跨域耦合"。因此它是隔离"耦合价值"的最干净对照；§1.3.3 的推导可直接改写进 SB-Render-Lite 论文的 related work / method motivation（"为什么 pivot 复合不够、需要真 SB"）。

---

## 2. 收录条目：SDEdit（arXiv 2108.01073）

- 标题：SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations
- 作者：Chenlin Meng、Yutong He、Yang Song、Jiaming Song、Jiajun Wu、Jun-Yan Zhu、Stefano Ermon（Stanford + CMU）
- **venue：ICLR 2022 Poster**（核验：OpenReview `aBsCjcPu_tE` 标 "ICLR 2022 Poster"；两个项目主页均标 "In ICLR 2022"；检索日期 2026-08-14）
- 归类建议：`zero_shot_translation_baseline`（单模型加噪-去噪翻译/编辑）

**一句话**：把 guide 图（笔画/拼贴/——对我们而言是 **仿真渲染图**）加噪到中间时刻 t0，再用 **仅在目标域（真实图）上预训练** 的扩散模型反向去噪到 t=0，即得"投影到真实图像流形"的输出；不训练任何新模型、不需要源域模型、不需要配对。

**方法要点**：

- 核心超参 t0 控制 **realism–faithfulness 权衡**：t0→0 输出≈guide 本身（保真但假），t0→1 退化为无条件采样（真实但无关）；作者经验推荐 t0∈[0.3, 0.6]，可对单张图二分搜索后全任务固定。
- Prop 1 给出概率上界：\(\|x^{(g)} - \text{SDEdit}(x^{(g)}; t_0)\|_2^2 \le \sigma^2(t_0)\,(C\sigma^2(t_0) + d + 2\sqrt{-d\log\delta} - 2\log\delta)\)（置信 1−δ，C 为 score 范数界，d 为维数）——faithfulness **只有上界、没有下界或结构保证**，与 DDIB 的 exact cycle consistency 形成对照。
- 实操配置（论文附录 C/D）：VE（σ_min=0.01，σ_max=348–1348 按数据集）与 VP（β_min=0.1，β_max=20）两版；stroke 合成 t0=0.5、N=500、K=1（VP）；编辑 t0=0.45–0.5；compositing t0=0.35、N=700；带 mask 变体可冻结不可编辑区域；可叠加 classifier guidance 做类条件 SDEdit（附录 E.3）。256² 单图 29.1 s（2080Ti，N=500）。
- 实验：LSUN bedroom/church、CelebA-HQ 上 stroke 合成/编辑/compositing，对 GAN inversion 基线（StyleGAN2-ADA 投影、in-domain GAN、e4e）人评 realism 最高 98.09%、总满意度最高 91.72% 胜率；全部用公开预训练 checkpoint，**未训练任何模型**。
- 影响力：即 Stable Diffusion `img2img` 的机制原型（项目主页明确此关联），是当代所有"渲染图→真实感重绘"工作流的祖型。

**与 DDIB / SB 的关系**：SDEdit 可视为 DDIB 的"半段+随机化"退化——把 DDIB 的确定性源域编码（完整 PF-ODE 到 t=1）替换为 **随机加噪到 t0<1**，且干脆不需要源域模型。层级：SDEdit（1 个模型，随机，部分深度）⊂ DDIB（2 个模型，确定性，全深度）⊂ 真 SB（跨域耦合优化）。它比 DDIB 更便宜，但漂移控制更弱：t0 是唯一旋钮，且对 contact-rich 帧（密集接触、遮挡）尤其危险。

**sim2real 用法**：sim 渲染帧当 guide，真实域模型去噪。这正是机器人文献里各类 "diffusion-based visual augmentation" 的原型，作为 B1 基线成本最低（若直接用开源底模连真实域训练都可省去，但那会引入域先验失配，规格里作为 B1' 变体单独标注）。

---

## 3. 关键事实核验清单（检索日期 2026-08-14）

| 事项 | 结论 | 依据 |
| --- | --- | --- |
| DDIB venue | ICLR 2023 ✅ | OpenReview `5HLoTvVGDe`（PDF 页眉）、项目页、GitHub README |
| SDEdit venue | ICLR 2022 Poster ✅ | OpenReview `aBsCjcPu_tE`、两个项目主页 |
| SB-FBSDE（DDIB 理论基座）venue | ICLR 2022 Poster ✅ | OpenReview `nioAdKCEdXB`、iclr.cc poster 页；库内 `guan_horng_liu_research_roadmap.md` 口径一致 |
| "DDPM encoder ≈ OT map" 之争 | 高斯情形成立、一般情形有反例、数值近似 | Khrulkov et al. arXiv 2202.07477（ICLR 2023）；Lavenant & Santambrogio, Appl. Math. Letters 133:108225 (2022) |
| 全文获取 | DDIB、SDEdit 均获 arXiv HTML 全文（含附录），无缺页 | 本次抓取 |

---

## 4. 零训练对照组规格（SB-Render-Lite）

### 4.1 分组与命名

| 组 | 方法 | 需要训练什么 | 翻译时用什么 |
| --- | --- | --- | --- |
| B0 | identity | 无 | sim 帧原样用（sim2real 差距下界的锚点） |
| B1 | SDEdit | 仅真实域扩散模型（1 个） | sim 帧加噪到 t0 → 真实域模型去噪 |
| B1' | SDEdit-底模 | 无（直接用开源预训练底模 img2img） | 同上；标注域先验失配风险 |
| B2 | DDIB | sim、real 两个单域扩散模型（推荐：1 个 domain-label 条件模型） | sim 模型 DDIM 全编码 → real 模型解码 |
| B2' | partial-depth DDIB | 同 B2 | 编码只到 t_enc<1 再换 real 模型解码（B1↔B2 插值） |
| 上方对照 | SB-coupled | SB Flow（unpaired，`2409.09347`）/ I²SB（paired 上限，`2302.05872`）/ GSBM（可加结构代价，`2310.02233`） | 学到的跨域 bridge |

术语纪律：**"零训练"专指零 bridge 训练、零配对训练**。B2 仍需训单域扩散模型（这部分成本与 SB 方法的骨干预训练共享，可复用）；B1 只需真实域一侧。

### 4.2 数据量与训练配置

- **域定义**：source = 仿真渲染 RGB，target = 真实机器人相机 RGB；统一在同一 VAE latent 空间操作（SD VAE 或自训小 VAE），分辨率与 SB 实验对齐（建议 256×256）。
- **数据量**：sim 侧 20k–100k 帧（渲染便宜，覆盖任务分布）；real 侧全量使用，并做 {1k, 5k, 20k} 三档消融——真实数据量是 sim2real 的真实约束，**DDIB 解码质量对 real 数据量的敏感度本身就是重要结论**（real 模型欠拟合时 B2 解码会产生幻觉细节）。与 SB 方法使用完全相同的数据切分。
- **模型**：UNet 50–100M 或 DiT-S/B 量级；**默认用 domain-label 条件化单模型**同时覆盖两域（DDIB ImageNet 实验的做法）：共享 latent 组织、稳住 B2、省一半训练。须在论文中注明这是 **有利于基线** 的设计选择，使"SB 净增益"结论保守可信。记录 GPU·h。
- **B1 配置**：t0 ∈ {0.3, 0.4, 0.5, 0.6} 扫描（论文推荐区间），K=1，DDIM 50–100 步；报告完整 t0–权衡曲线而非单点。
- **B2 配置**：DDIM 确定性 encode/decode 各 ≥100 步（离散误差直接决定 cycle 质量）；**必报 cycle 误差**（sim→real→sim 的 latent L2）作为实现 sanity check（参照论文 2D 量级：单位方差下 ~0.01）；B2' 取 t_enc ∈ {0.5, 0.75, 1.0}。
- **公平性**：所有方法同 VAE、同数据、同（或至少如实报告）推理 NFE；SB 方法的 NFE 优势/劣势要单列（真 SB 可少步采样，DDIB 恒需两段全程）。

### 4.3 评估口径（与 SB 方法逐项对齐，继承库内口径）

1. **视觉分布**：FID/KID（翻译结果 vs real 验证集）——次要指标，防止"翻译=美化"误导。
2. **语义/几何保持**（零训练基线预期的失分处，也是 SB 应赢的地方）：
   - DINO/CLIP 特征相似度（翻译前后）；
   - depth / keypoint 一致率：利用 sim 侧 GT depth 与物体 keypoint，测翻译前后投影一致性；
   - 物体 mask IoU（sim GT mask vs 翻译后重检测 mask）；
   - **语义漂移率**：keypoint 位移超阈值（如 >5 px @256²）样本占比——作为单一诊断数字进入主表。
3. **时序**：相邻帧独立翻译的 warp error（光流一致性）——B1/B2 逐帧独立，预期显著差于带时序耦合的方案。
4. **主指标（库内纪律）**：翻译后的 sim 数据按 `2509.18631` 的 co-training 协议参与策略训练，报告 **real-domain policy success**；辅以 inverse dynamics consistency（翻译前后帧对的动作可预测性）。
5. **决策规则**：定义 gap = success(oracle real 数据) − success(B0)。若 max(B1, B2) 已收回 ≥70% 的 gap，则 SB-Render-Lite 的贡献表述必须转为"剩余差距 + 漂移率 + 时序一致性"的改善，并给出"SB 相对最优零训练基线的收益–成本曲线"（R09 G11 的原始要求）。

### 4.4 预期失败模式（写进实验假设，供证伪）

- **B2/DDIB**：独立训练两模型时布局漂移、物体身份被 OT"搬运"替换；real 数据不足时解码幻觉；B→A 型方向性退化（§1.5-2）。
- **B1/SDEdit**：t0 小 → 保留 sim 质感（假），t0 大 → 场景重排（漂移）；contact-rich 帧最危险。
- **共同**：无时序一致性；无 action-aware 约束（对照 EgoBridge/GOT 线的 joint feature-action cost）。

---

## 5. 并入主库建议

1. **报告与索引**：将本文件（或拆出的 DDIB 精读部分）以 `reports/2203.08382_ddib_dual_diffusion_implicit_bridges.md` 体例并入；`INDEX.md` 新增小节「零训练翻译基线（对照组）」，收 DDIB（精读）与 SDEdit（收录条目），并与「重要对照 / SB 方法支撑」小节互链。
2. **papers.tsv**：追加两行，category 建议 `zero_shot_translation_baseline`：`2203.08382`（year 2023）、`2108.01073`（year 2022）；PDF 按库内惯例补抓。
3. **synthesis.md**：在实验矩阵中并入 §4 的 B0/B1/B1'/B2/B2' 分组与决策规则；「为什么需要 SB 耦合」一节可直接引用 §1.3.3 的三层推导（L1 路径 vs 耦合、L2 flow map ≠ Brenier、L3 复合不最优——d=1 例外、d≥2 反例、无跨域 cost）。
4. **交叉引用**：DDIB 理论基座 SB-FBSDE（2110.11291）已在 `guan_horng_liu_research_roadmap.md` 覆盖，精读版无需重复，链接即可；R09 提到的 Latent Schrödinger Bridge（arXiv 2411.14863，声称以远低于 SDEdit/DDIB 的 NFE 逼近 SB ODE）属 G9 范围，建议后续扩充时与本条目做效率对比联动。
5. **论文写作素材**：related work 中 DDIB/SDEdit 应作为"零训练下界"出现，引用口径——DDIB=ICLR 2023、SDEdit=ICLR 2022（均已 web 复核，检索日期 2026-08-14）；"most OT-efficient" 声明引用时须附 Lavenant–Santambrogio 2022 的限定。
