# E19 扩充笔记：OT 理论工具箱 —— UOT / GW 系变体 / Neural OT 求解器与评估陷阱

- 撰写：文献扩充研究员 E19（选题来自 R09 缺口分析："OT 理论工具箱"）
- 日期 / 检索日期：2026-08-14（所有发表状态均于当日经 web 复核）
- 阅读深度：**半精读 1 篇**（Neural OT, arXiv 2201.12220）；**综述式收录 5 篇**（UOT 1607.05816、entropic GW ICML 2016、fused GW 1805.09114、UGW 2009.04266、W2 benchmark 2106.01954）
- 资料来源：arXiv abs 页返回的 HTML 全文（1607.05816 / 2009.04266 / 1805.09114 / 2201.12220 / 2106.01954 均获取到含公式的正文）＋ PMLR 官方 PDF 文本（entropic GW，该篇**不在 arXiv**，以 HAL hal-01322992 与 PMLR v48 为准）。无"无法获取全文"情形。

## 选题定位

主库 25 篇精读集中在**应用侧**：OT/UOT/GW/Sinkhorn 在 imitation reward、sim-real co-training、SB 翻译中的用法（见 `reports/synthesis.md` 第 2 节方法谱系）。但库内没有一份统一的**理论工具笔记**回答三个反复出现的问题：

1. sim 与 real 数据**质量不守恒**（real 长尾、sim 可控均匀；类别/模式比例失衡）时，什么情况下必须放弃 balanced OT 改用 UOT？松弛参数怎么定？
2. 跨视角/跨 embodiment 时特征不可直接比，GW 系（entropic GW / fused GW / unbalanced GW）各自的准确定义、成本和边界在哪？（GWIL 报告 `2110.03684` 已覆盖 GW 的 IL 应用侧，本笔记只补理论口径，不重复。）
3. 神经网络"解 OT"的方法实际解出来的是什么？W2 benchmark 揭示的"neural OT 求解器解的可能不是 OT"对 SB-Render-Lite 的 coupling 模块选型和消融设计意味着什么？

落点：SB-Render-Lite 的 **coupling 设计**（sim latent ↔ real latent 如何配对）与**消融实验设计**（如何验证 coupling 质量）。

## TL;DR

1. **UOT 的准确口径**是把边缘约束换成 φ-divergence 罚项：`inf_γ ∫c dγ + D_φ1(γ₁|μ) + D_φ2(γ₂|ν)`。KL 情形下 Sinkhorn 只需把边缘投影改成幂次 `λ/(λ+ε)` 的软投影，代价与 balanced 相同。ρ（=λ）有几何含义：控制"搬运 vs 生灭"的预算，WF/HK 情形有显式截断半径——超过该距离的质量只生灭不搬运。
2. **sim/real 类别比例失衡时 balanced coupling 数学上必然错配**：总质量守恒强迫 sim 过剩模式的质量流向 real 的错误模式。UOT 是即插即用替换，且已在库内 `2509.18631`（sim-real co-training）中实际使用；本笔记补齐其选参与诊断协议。
3. **GW 是二次指派问题（QAP、NP-hard）**，entropic GW 的镜像下降每步恰是一次 Sinkhorn，配合损失分解技巧每迭代 O(n²m+nm²)；**FGW** 用 α 在"特征 W 距离"与"结构 GW 距离"间插值，q=1 时是真度量；**UGW** 必须用 quadratic divergence `D_φ(π⊗π|μ⊗μ)` 松弛边缘才能保持 2-齐次性与定性。三者都只保证收敛到**稳定点**，且对齐只到等距同构（反射模糊）。
4. **W2 benchmark 的核心教训**：以 FID 论优劣的 neural OT 求解器可能根本没在解 OT——QC 求解器在 D=256 时 L2-UVP 88.2%（比"把两域当高斯"的线性基线 67.4% 还差），FID 却很好；作者结论"bad OT solvers can yield good generative performance"。SB-Render-Lite 的 coupling 质量**必须用独立于视觉质量的指标验证**。
5. **NOT（半精读）**给出 weak OT 的 maximin 算法：`sup_f inf_T`，T(x,z) 随机映射表示 plan。两个对 SB-Render-Lite 直接相关的现象：strong cost 下 z 输入必然被忽略（conditional collapse）；maximin 的 arginf 集合可能含非 OT 的"假解"（cost 对 μ 严格凸时才排除）。
6. SB = **entropic** OT 的动态形式，NOT/W2-benchmark 家族解的是 **unregularized** OT——是不同对象。SB-Render-Lite 的 ε 是建模选择而非 nuisance，必须显式报告并纳入消融。

## 收录清单（venue 于 2026-08-14 复核）

| # | 工具 | 论文 | venue（复核结果） | 深度 |
|---|---|---|---|---|
| 1 | Unbalanced OT + scaling 算法 | Chizat, Peyré, Schmitzer, Vialard, *Scaling Algorithms for Unbalanced Optimal Transport Problems*, [arXiv:1607.05816](https://arxiv.org/abs/1607.05816) | **Mathematics of Computation** 87(314):2563–2609, 2018, DOI 10.1090/mcom/3303 | 综述式 |
| 2 | Entropic GW | Peyré, Cuturi, Solomon, *Gromov-Wasserstein Averaging of Kernel and Distance Matrices* | **ICML 2016**, PMLR v48:2664–2672；不在 arXiv（HAL hal-01322992） | 综述式 |
| 3 | Fused GW | Vayer, Chapel, Flamary, Tavenard, Courty, *Optimal Transport for structured data with application on graphs*, [arXiv:1805.09114](https://arxiv.org/abs/1805.09114) | **ICML 2019**, PMLR v97:6275–6284（titouan19a） | 综述式 |
| 4 | Unbalanced GW | Séjourné, Vialard, Peyré, *The Unbalanced Gromov Wasserstein Distance: Conic Formulation and Relaxation*, [arXiv:2009.04266](https://arxiv.org/abs/2009.04266) | **NeurIPS 2021**, pp. 8766–8779 | 综述式 |
| 5 | Neural OT | Korotin, Selikhanovych, Burnaev, *Neural Optimal Transport*, [arXiv:2201.12220](https://arxiv.org/abs/2201.12220) | **ICLR 2023**（notable top-25% / spotlight，OpenReview d8CBRlWNkqH） | **半精读** |
| 6 | W2 benchmark | Korotin, Li, Genevay, Solomon, Filippov, Burnaev, *Do Neural Optimal Transport Solvers Work? A Continuous Wasserstein-2 Benchmark*, [arXiv:2106.01954](https://arxiv.org/abs/2106.01954) | **NeurIPS 2021**（proceedings.neurips.cc, hash 7a6a6127） | 综述式 |

---

## 第一节 UOT：何时必须用

### 1.1 数学口径

**通用定义**（Chizat et al. 2018, Def. 2.11；形式源自 Liero–Mielke–Savaré 2015）。给定成本 c 与两个 φ-divergence（Csiszár divergence，由熵函数 φ 生成，含 KL、TV、等式约束、区间约束等特例）：

$$\mathrm{UOT}(\mu,\nu)\;=\;\inf_{\gamma\in\mathcal{M}_+(X\times Y)}\ \int_{X\times Y} c\,\mathrm{d}\gamma\;+\;\mathcal{D}_{\varphi_1}\!\big(P^X_\#\gamma\,\big|\,\mu\big)\;+\;\mathcal{D}_{\varphi_2}\!\big(P^Y_\#\gamma\,\big|\,\nu\big)$$

取 φ = ι_{{1}}（等式约束）即退回 balanced OT；取 D_φ = λ·KL 得最常用的 KL-UOT，λ→∞ 时恢复 balanced。**关键点：γ 在全体非负测度上优化，总质量不守恒；γ 的边缘不再等于 μ、ν，只被罚项拉近。**

**熵正则形式与算法**（同文 eq. 3.2 与 Prop. 5.1 附近）。加 ε 熵项后问题变为

$$\min_{\gamma}\ \varepsilon\,\mathrm{KL}\big(\gamma\,\big|\,e^{-c/\varepsilon}\mathrm{d}x\mathrm{d}y\big)+\lambda_1\mathrm{KL}(\gamma_1|\mu)+\lambda_2\mathrm{KL}(\gamma_2|\nu),$$

由通用 scaling 算法（Dykstra 迭代特例）解出，KL 情形有闭式：

$$a^{(\ell+1)}=\Big(\frac{\mu}{K b^{(\ell)}}\Big)^{\frac{\lambda_1}{\lambda_1+\varepsilon}},\qquad b^{(\ell+1)}=\Big(\frac{\nu}{K^{\top} a^{(\ell+1)}}\Big)^{\frac{\lambda_2}{\lambda_2+\varepsilon}},\qquad K=e^{-c/\varepsilon}.$$

与标准 Sinkhorn 唯一的区别是指数从 1 变成 λ/(λ+ε)（λ→∞ 还原 Sinkhorn）。文中证明该迭代在 Thompson 度量下线性收敛，并给出 log-domain absorption 稳定化（Algorithm 2）与 ε-scaling 技巧。

**两个有几何意义的特例**（帮助给 ρ/λ 赋物理含义）：

- **Wasserstein–Fisher–Rao / Hellinger–Kantorovich**（WF/HK）：c(x,y) = −log cos²₊(d(x,y))，cos₊(z)=cos(z∧π/2)，D = λKL。此时**存在显式截断半径（cut locus）**：距离超过阈值的质量对搬运成本为 +∞，只会被生灭处理（论文 Fig. 3(h) 展示了 cut locus=0.2 的最优 plan）。
- **Optimal partial transport**：D = λTV，等价于只搬运一部分质量、其余按每单位 2λ 的价格丢弃——λ 是质量约束的拉格朗日乘子。

### 1.2 解决什么问题（四种"必须用"的场景）

1. **总质量本身不等/归一化无意义**：如密度估计的中间量、计数数据。
2. **离群点鲁棒性**：balanced OT 强迫离群质量被搬运，代价可以任意大；KL-UOT 把离群质量的影响封顶为"生灭价"。
3. **模式/类别比例失衡（对本库最重要）**：源域某模式质量 30%、目标域同模式只有 10% 时，balanced coupling 必须把多出的 20% 塞进目标域的**其它**模式（错配是约束的必然后果，不是优化没做好）；UOT 允许按价销毁多余质量。
4. **部分匹配**：只有子集可对应（遮挡、视野差异），TV 型 UOT = partial OT。

### 1.3 计算成本

- 每迭代与 Sinkhorn 同阶：一次 K·b 矩阵向量乘 O(nm)，存 K 需 O(nm) 显存；网格上平方欧氏成本可用可分离高斯卷积免存 K。
- KL 情形 proxdiv 闭式、TV/区间约束情形也有闭式（论文 Table 1）；小 ε 需 log-domain 稳定化（与稳定化不兼容卷积技巧）。
- 收敛速度经验上对 λ 减小而加快（罚项越松越快），对维度不敏感。

### 1.4 常见误用

1. **把 UOT 的 γ 当概率耦合用**。γ 总质量 ≠ 1 且边缘 ≠ 输入分布；下游当 pairing 权重前必须明确归一化方案，且要决定"被销毁的质量"（不配对样本）如何处理。
2. **ρ/λ 不校准、不消融**。UGW 论文原文明确写"for supervised tasks, the value of ρ should be cross-validated"。λ 过小→几乎全生灭、没有搬运（对齐失效）；λ 过大→退回 balanced（失衡问题回归）。
3. **忽略 λ 与 ε 的耦合**。有效软化指数是 λ/(λ+ε)：固定 λ 扫 ε 时，边缘松弛强度也在变。消融要么固定 λ/(λ+ε)，要么同时报告两者。
4. **跨 (λ, ε) 配置直接比较 UOT 数值**。罚项量纲随 λ 变，UOT 值不可比；要比就比固定配置下的相对量或下游指标。
5. **KL 与 TV 混用不加区分**。KL = 按比例软折价（每单位质量的边际生灭价随偏离度增长）；TV = 硬阈值（价格恒定 2λ，超过就整块丢）。长尾"稀释"场景宜 KL；遮挡/视野"截断"场景宜 TV。

### 1.5 对 SB-Render-Lite 的落点

SB-Render-Lite 的训练里 coupling 出现在两处：(i) minibatch 内 sim/real 样本配对（SB Flow / I²SB 风格的 bridge 训练都要先定端点耦合）；(ii) latent-action 联合对齐损失（synthesis 4.2 基线清单里的"OT/UOT joint feature-action alignment"，库内 `2509.18631` 已实践）。落点四条：

1. **默认把 minibatch 配对从 balanced Sinkhorn 换成 KL-UOT**：实现代价只是指数上加 λ/(λ+ε)（POT `ot.unbalanced` / OTT-JAX 均现成）。real 长尾 + sim 均匀的类别失衡正是 1.2-3 场景。
2. **λ 的校准协议**：在共享 encoder 的 latent 空间里，统计"同类 sim-real 对"与"异类对"的距离分布，把生灭价（KL 情形约 2λ 量级，WF 情形是显式 cut 半径）定在两分布之间——使异类搬运比生灭贵、同类搬运比生灭便宜。然后消融 λ ∈ {校准值/4, 校准值, 校准值×4, ∞(=balanced)}。
3. **诊断指标**：按类别/任务阶段报告边缘保留率 m(γ₁ 在该类)/m(μ 在该类)。健康状态 = 被打折的 sim 类别恰是 real 缺失或稀缺的类别；若打折发生在 real 充足的类别，说明 λ 错或 encoder 距离不含语义。
4. **与 balanced 的对照必须保留**：λ=∞ 一档是消融锚点，可直接量化"质量不守恒处理"带来的下游 policy success 增益——这是"UOT 相关性"主张的实验证据位。

---

## 第二节 GW 系变体与结构对齐

三个工具共同的对象：**metric-measure space（mm-space）** X=(X, d_X, μ)——空间 + 域内距离/相似度 + 权重。GW 系不比较跨域点对，只比较**域内两两关系结构**，因此天然适配"两个域的特征不可直接比"的设定。

### 2.1 Entropic GW（Peyré–Cuturi–Solomon, ICML 2016）

**口径**。对离散 mm-space (C, p) 与 (C̄, q)（C 为任意相似度矩阵，不必是距离阵），GW discrepancy 为

$$\mathrm{GW}(C,\bar C,p,q)=\min_{T\in\mathcal{C}_{p,q}}\ \sum_{i,j,k,\ell} L\big(C_{ik},\,\bar C_{j\ell}\big)\,T_{ij}\,T_{k\ell},$$

L 为逐元素损失（平方损失或 KL）。这是二次指派问题（QAP）的松弛，**非凸**；L=L2 且 C 为距离阵时 GW^{1/2} 是 mm-space 商等距同构后的距离（Mémoli 2011）。熵正则版 GW_ε 减去 εH(T) 后用 KL 镜像下降求解，步长 τ=1/ε 时每步恰为一次 Sinkhorn 投影：

$$T\ \leftarrow\ \mathcal{T}_{\varepsilon}\big(\mathcal{L}(C,\bar C)\otimes T,\ p,\ q\big),$$

即"用当前 T 合成线性成本 → 解一次 entropic OT"。

**关键计算技巧**（Prop. 1）：若 L(a,b)=f₁(a)+f₂(b)−h₁(a)h₂(b)（平方损失、KL 均满足），则 L(C,C̄)⊗T = c_{C,C̄} − h₁(C) T h₂(C̄)^⊤，把每次成本合成从朴素 O(n²m²) 降到 **O(n²m + nm²)** 的矩阵乘。这是后续所有 GW 数值方法（含 POT 实现、UGW、FGW line-search）的基石。

**注意**：τ=1/ε 不在收敛理论覆盖范围内（理论要求 τ 足够小），实践中稳定；只保证收敛到**稳定点**，且这是软指派版 quadratic assignment（softassign）的推广。

### 2.2 Fused GW（Vayer et al., ICML 2019）

**口径**。结构化对象 = 每个节点带特征 a_i（跨域可比）+ 域内结构矩阵 C（域内两两关系），表示为乘积空间上的测度 μ=Σ h_i δ_{(x_i,a_i)}。FGW 用 α∈[0,1] 融合两种成本：

$$\mathrm{FGW}_{q,\alpha}(\mu,\nu)=\min_{\pi\in\Pi(h,g)}\ \sum_{i,j,k,\ell}\Big[(1-\alpha)\,d(a_i,b_j)^q+\alpha\,\big|C_1(i,k)-C_2(j,\ell)\big|^q\Big]\,\pi_{ij}\,\pi_{k\ell}.$$

**插值定理**（Thm 3.1）：α→0 退化为特征上的 Wasserstein，α→1 退化为结构上的 GW。**度量性质**（Thm 3.2）：q=1 时是真度量（商掉"保权重、保特征、保结构"的等距同构）；q>1 时三角不等式松弛 2^{q-1} 倍，是 semi-metric。

**求解与成本**：Conditional Gradient（Frank–Wolfe）——每步 (i) 算梯度（q=2 时用 2.1 的分解技巧，O(n²m+nm²)）；(ii) 以梯度为成本解一个**线性** OT（网络流或 Sinkhorn）；(iii) 闭式 line-search（二次多项式约束极小）。非凸，收敛到局部稳定点（Lacoste-Julien 2016 的非凸 FW 结果）。POT 有 `fused_gromov_wasserstein` 与 entropic/BAPG 变体。

### 2.3 Unbalanced GW（Séjourné–Vialard–Peyré, NeurIPS 2021）

**口径**。把 GW 推到任意正测度时，**不能**直接套 UOT 的一次divergence罚项——那样的泛函在质量缩放 θμ, θν 下行为不一致（θ→0 与 θ→∞ 极限分别是 GW 与 Hellinger 型，量纲错乱）。正确做法是引入 **quadratic divergence**：

$$\mathrm{D}_\varphi^{\otimes}(\rho\,|\,\nu)\ \triangleq\ \mathrm{D}_\varphi\big(\rho\otimes\rho\,\big|\,\nu\otimes\nu\big),$$

$$\mathrm{UGW}(\mathcal X,\mathcal Y)=\inf_{\pi\in\mathcal{M}_+(X\times Y)}\ \iint \lambda\big(|d_X(x,x')-d_Y(y,y')|\big)\,\mathrm{d}\pi\,\mathrm{d}\pi\;+\;\mathrm{D}_\varphi^{\otimes}(\pi_1|\mu)+\mathrm{D}_\varphi^{\otimes}(\pi_2|\nu).$$

这保证 UGW 是 **2-齐次**的（θ⁻²UGW(θμ,θν)=UGW(μ,ν)），且正定（Prop. 2：UGW=0 ⟺ 两 mm-space 等距同构）。φ=ι_{{1}} 或 ρ→∞ 退回 balanced GW。同文还给出 conic 提升的 **CGW**——真正满足三角不等式的距离，且 UGW ≥ CGW（Thm 1）；CGW 需在锥空间离散化、不可扩展，实际计算都用 UGW。

**算法**：熵正则 + **bi-convex 松弛**——把 π 复制成 (π, γ) 交替最小化，固定一个时对另一个是凸的 entropic UOT（每步即第一节的幂次 Sinkhorn），GPU 友好（官方 `pip install unbalancedgw`）。**tightness 注意**：ε=0 的 balanced GW 情形有 Konno 定理保证交替解满足 π=γ（当 |d_X−d_Y|² 条件负定，如树度量/欧氏/球面/双曲距离）；**熵正则 UGW_ε 情形 π=γ 是否成立是开放问题**，实验上总是成立。图上的测地距离不满足条件负定，理论保证失效。

### 2.4 GW 系共同的常见误用

1. **把稳定点当全局最优**。GW/FGW/UGW 全部非凸；不同初始化（乘积耦合、随机、由特征 OT warm-start）可得不同耦合。严肃使用需 multi-restart + 用能量值筛选。
2. **忽略等距模糊（尤其反射）**。GW 只识别到等距同构：欧氏嵌入下反射/对称解与真解能量相同。对机器人数据这意味着**左右手性、旋转对称物体的朝向可能被翻转**——下游动作映射会系统性出错，必须用外部信息（action 标签、重力方向）消歧。
3. **FGW 的 α 不做量纲归一**。特征成本 d(a,b)^q 与结构成本 |C₁−C₂|^q 单位不同，α=0.5 无先验意义；应先把两项归一（如各除以其均值）再扫 α，且 α∈{0,1} 两端必须做对照。
4. **域内距离不含任务语义**。GW 对齐的是 C 的结构；若 C 用原始像素距离而非任务相关表征，"结构相似"≠"任务对应"（GWIL 报告 `2110.03684` 的局限一节已指出，此处从理论侧确认：GW 的不变量是 C 的等距类，仅此而已）。
5. **长尾/离群数据用 balanced GW**。与第一节同理，模式比例失衡时结构错配是必然；且注意 UGW 的松弛项是 quadratic divergence，**其 ρ 与 UOT 的 ρ 量纲不同**（罚的是 π⊗π），调参不能直接搬。
6. **数值不可比**。GW/FGW 值随 C 的尺度二次变化；跨数据集或跨归一化方案比较绝对值无意义。

### 2.5 对 SB-Render-Lite 的落点（跨视角结构对齐的可行性）

**场景分层**（呼应 GWIL 报告的判断）：sim/real 若共享相机配置与状态空间结构，普通 OT/UOT 在共享 encoder latent 上即可，GW 系是杀鸡用牛刀；GW 系的适用区是**跨视角**（第三人称 sim ↔ 腕部相机 real）与**跨 embodiment**（后续 human→robot 阶段）。可行性评估：

1. **形式选择：fused GW 优先**。特征项放跨域近似可比的部分（DINO/CLIP latent 距离），结构项放域内可靠的部分：同一 episode 内的**时序邻接距离**（|t−t'| 或 phase 差）、**keypoint 两两距离阵**、或 (s,a) 图。α 扫描 {0, 0.25, 0.5, 0.75, 1}，α=0 即普通特征 OT 对照。q=2 便于计算（分解技巧），报告指标时注明是 semi-metric。
2. **长尾处理**：real 侧长尾时用 UGW 型松弛。注意"fused + unbalanced"的组合超出了 UGW 论文的定理覆盖（POT 有工程实现路径），只当启发式用，不引其理论性质。
3. **规模可行**：minibatch 级 n,m ≤ 512 帧时每迭代 O(n²m+nm²) 在 GPU 上毫秒级，可以放进 dataloader 在线算；episode 级 n~10³ 可离线预算；全数据集级 n≥10⁵ 不可行——只做分层抽样后的 batch 级耦合。
4. **角色定位（重要）**：第一阶段**不要**用 GW 系替换 SB 的核心 objective；先用作 (i) coupling 正则（把 FGW 耦合当 SB 端点配对的 prior），(ii) 评估指标——"structure preservation score"：transport 前后 keypoint 两两距离阵的 GW 残差。原因：非凸稳定点 + 反射模糊直接进主 objective 风险高。
5. **验证协议**：用 sim 引擎造"同场景双视角"合成对（真值对应已知），测 FGW 耦合恢复真值对应的 top-1 准确率；用 action 标签一致性检测反射错配（左右翻转的耦合会让 gripper 开合/移动方向标签冲突）。

---

## 第三节 Neural OT 求解器与评估陷阱

先立一个对本库关键的概念区分：**SB = entropic OT 的动态形式**（ε>0，路径测度视角），而本节两篇处理的是 **unregularized OT**（ε=0，map/plan 视角）。两者极限相通（ε→0），但学到的耦合对象不同。SB-Render-Lite 若声称"学到了 OT coupling"，要先说清是哪一个。

### 3.1 Neural OT（半精读；Korotin et al., ICLR 2023 notable top-25%）

**问题与口径**。目标是直接学 OT plan（而非只把 OT cost 当 GAN 损失）。采用 **weak OT**（Gozlan et al. 2017）：成本函数吃进整个条件分布，

$$\mathrm{Cost}(\mathbb P,\mathbb Q)=\inf_{\pi\in\Pi(\mathbb P,\mathbb Q)}\int_{\mathcal X} C\big(x,\ \pi(\cdot|x)\big)\,\mathrm{d}\pi(x),$$

strong OT 是 C(x,μ)=∫c(x,y)dμ(y) 的特例。实验用 **γ-weak quadratic cost**：C(x,μ)=∫½‖x−y‖²dμ(y) − (γ/2)Var(μ)（γ≤1 时下有界、对 μ 凸）——方差项**奖励**条件分布的多样性。

**核心构造**（Lemma 2–4, Corollary 1）。把对偶式里的 C-transform `f^C(x)=inf_μ {C(x,μ)−∫f dμ}` 中对分布 μ 的优化，改写为对**随机映射** t: Z→Y 的优化（noise outsourcing：任意 plan 都可由 T(x,z), z∼S 表示），再用 Rockafellar 互换定理把逐点 inf 提升为对 T(x,z) 的整体 inf，得到 maximin 鞍点问题：

$$\mathrm{Cost}(\mathbb P,\mathbb Q)=\sup_f\ \inf_T\ \Big[\int f\,\mathrm{d}\mathbb Q+\int_{\mathcal X}\Big(C\big(x,T(x,\cdot)_\#\mathbb S\big)-\int_{\mathcal Z} f\big(T(x,z)\big)\mathrm{d}\mathbb S(z)\Big)\mathrm{d}\mathbb P(x)\Big].$$

Lemma 4：最优 potential f* 下，任何实现最优 plan 的 T* 都在内层 arginf 中。**但反向不保证**：arginf 可能还含**不是** OT map 的"假解"；附录 F 证明当 C(x,·) 对 μ 严格凸时假解不存在——strong cost 恰好**不**严格凸。

**算法**（Algorithm 1）：两个网络 T_θ(x,z)（UNet 类）与 f_ω（ResNet 类），SGAD 交替；**T 内层多步（K_T>1）、f 外层单步**——与 GAN 的"判别器多步"惯例相反；f 无需 Lipschitz 约束（对比 WGAN）。理论支撑：Theorem 1 证明神经网络是随机传输映射的 L² 普适逼近器。

**关键实验现象**：
- **Conditional collapse**：strong cost（γ=0）时 T(x,z) 自发忽略 z、退化为确定性映射（因为 strong cost 的 f^C 的 arginf 可由退化分布达成）。此时 NOT 退化为 W2 benchmark 里表现最好的 ⌈MM:R⌉ 求解器（同一作者线；Rout et al. 2022、Fan et al. 2022 是其 strong-cost 推广）。
- γ>0 时得到真正的 one-to-many 随机映射；γ 是"多样性 vs 与输入相似性"的单旋钮（γ=1 时多样但风格保持变差）。
- 无监督翻译实验（celeba↔anime、handbags↔shoes、outdoor→church，64–128px）：确定性 OT map 以 L² 像素成本"最小改动"翻译，天然保颜色/姿态/构图——这正是 sim2real 想要的"保任务结构"倾向的朴素版本。
- **局限（原文自述）**：鞍点解集含非 OT 解的问题未解决；maximin 训练稳定性依赖超参。

### 3.2 W2 benchmark（Korotin et al., NeurIPS 2021）

**方法论**。利用 Brenier 定理（二次成本下 OT map = 凸函数梯度）：取 ICNN ψ（input-convex neural network），则 (P, ∇ψ#P) 是一对**真值已知**的 benchmark 测度对。构造高维高斯混合对（D=2…256）与 CelebA 64×64 图像对。指标：

$$\mathcal{L}^2\text{-UVP}(\hat T)=100\cdot\frac{\|\hat T-T^*\|^2_{\mathcal L^2(\mathbb P)}}{\mathrm{Var}(\mathbb Q)}\%,\qquad \cos\big(\hat T-\mathrm{id},\,T^*-\mathrm{id}\big)\in[-1,1],$$

前者度量 map 误差（常数基线=100%），后者度量用于生成模型梯度更新的方向对不对。基线：identity、常数、**线性**（把两域当高斯的闭式 OT map）。

**被测求解器**：⌈LS⌉（熵/二次正则对偶，Seguy et al.）、⌈MM⌉（maximin+amortized H）、⌈MM-B⌉（batch 内解 inner min）、⌈MMv1/MMv2⌉（ICNN 参数化）、⌈W2⌉（ICNN+cycle 回归，非 maximin）、⌈QC⌉（batch 离散 OT 对偶回归，WGAN-QC）及 reversed 版本。

**结果要点**：
- D=2 时人人都好；维度升高后只有 ⌈MM⌉/⌈MMv1⌉/⌈MMv2⌉/⌈W2⌉ 及其 reversed 版保持 UVP ≤10%。
- **⌈QC⌉ 在 D=256 时 UVP=88.2%，比线性基线（67.4%）还差**；⌈LS⌉ 54.7%（正则化偏差）；⌈MM-B⌉ 22.5%（batch 内 min 高估 f^c，系统性偏差）。
- CelebA 上 ⌈QC⌉/⌈MM-B⌉ 的 cos≈0——给生成器的梯度方向与真实 W2 梯度**近乎正交**，"这些求解器没有在提取 W2"。
- 生成建模实验的反转：⌈QC⌉ FID 最好但学到的 map 远离 identity（收敛后理应≈identity），作者判断它实际在优化某个"非 W2 的 dissimilarity"；⌈MM⌉ 受 **gradient deviation** 之害（‖f−f*‖ 小不代表 ‖∇f−∇f*‖ 小，仅 ICNN 系方法有理论控制），在低维流形支撑的生成设定下产出模糊图像；maximin 求解器还出现"先收敛到近优鞍点、后发散"的训练轨迹。
- 结论原文："**bad OT solvers can yield good generative performance**"、"increased OT accuracy does not necessarily correlate to better results downstream"。

### 3.3 评估陷阱清单（从两篇提炼，按对本库的危险度排序）

1. **视觉指标遮蔽耦合错误**。FID/LPIPS 好 ⇏ coupling 接近最优/保结构（⌈QC⌉ 教训）。危险形态：SB-Render-Lite 的翻译图看着很真，但 sim 帧被映到了语义错误的 real 模式上，policy 学到错误关联。
2. **minibatch 离散 OT 的偏差**。batch 内解 OT/取 min 是总体问题的有偏估计（⌈MM-B⌉ 的 inner-min 高估、⌈QC⌉ 的 batch 对偶变量偏差；比较 Bellemare et al. 2017 的偏梯度论证）。SB 训练里 batch 级端点配对同样受影响，batch size 是隐藏超参。
3. **优化到的不是想要的解**。maximin 的 arginf 假解（NOT Lemma 4 反向不成立）、训练后期发散（W2 benchmark 的 ↬ 现象）、conditional collapse（z 被忽略）。
4. **正则化偏差被无视**。熵正则 plan 系统性偏离 unregularized plan（⌈LS⌉ 教训）；SB 的 ε 同理——它改变耦合本身（ε 大→耦合趋独立、翻译趋"平均脸"），不是免费的数值技巧。
5. **potential 准 ⇏ map 准**（gradient deviation）。任何"学 potential/score、用其梯度当 map"的模块都要直接验 map，不能只看 potential 损失下降。

### 3.4 对 SB-Render-Lite 的落点（coupling 质量验证协议 + 消融设计)

1. **主指标不动摇**：downstream real-domain policy success（与 synthesis §4.3 口径一致）；本节补的是"**coupling 质量的独立证据链**"，防止"FID 好所以 coupling 好"的推理漏洞。
2. **移植 W2 benchmark 方法论造已知真值对**（消融套件的第一层）：在 latent 空间取 ICNN ψ，构造 (P_sim-latent, ∇ψ#P_sim-latent) 对，用 SB-Render-Lite 的 coupling 模块去学，报告 L²-UVP 与 cos。这是唯一能给出"绝对误差"的检查；任何 coupling 模块改动（UOT 化、FGW 正则、ε 调整）先过这一关。
3. **sim→real 无真值时的三件套**：(a) 边缘拟合——transported sim latent 分布与 real latent 分布的 Sinkhorn divergence；(b) 逆一致性——训练反向 real→sim bridge 做 cycle 检查（对应 W2 benchmark 的 reversed solver 敏感性发现：正反向不对称本身就是诊断信号）；(c) 不变量保持率——物体身份 / keypoint 几何 / action label 经 transport 的保持率（与 synthesis §4.3 辅助指标衔接）。
4. **必做消融**：batch size ∈ {64, 256, 1024}（陷阱 2）；ε 扫描并报告耦合熵（陷阱 4）；训练曲线上记录 held-out transport cost 与边缘残差，采用"最优 checkpoint"而非"最终 iterate"（陷阱 3）。
5. **随机翻译器设计**：需要 one-to-many 的 sim→real 渲染增广时，学 NOT 的 γ-weak 设计——显式方差奖励 + 单旋钮 γ 控制多样性，并做 **z-sensitivity 检查**（∂T/∂z 的范数或不同 z 下输出的 LPIPS 散度）防 conditional collapse；用 strong cost + z 输入的结构是已知无效组合。
6. **选型备忘**：要确定性"最小改动"翻译 → NOT 确定性版 / MM:R 系（W2 benchmark 认证的可靠区）；要带熵的动态耦合与生成路径 → SB 系（库内 SB Flow `2409.09347`、I²SB `2302.05872`）；两者杂交（如 DDIB≈SB 连接，NOT 论文相关工作里已引）时 ε 语义要写清楚。

---

## 与库内已有 25 篇的衔接（不重复精读）

- `2110.03684` GWIL：GW 在 cross-domain IL 的应用与局限（本笔记 §2 只补理论口径与误用清单，应用判断以 GWIL 报告为准）。
- `2509.18631` Guided OT sim-real co-training：库内 UOT 的实际用例；§1.5 的 λ 校准与边缘保留率诊断可直接用于其复现/消融。
- `2409.09347` SB Flow / `2302.05872` I²SB：unpaired / paired 翻译基线；§3 的 coupling 验证协议适用于两者的端点配对环节。
- `2509.19626` EgoBridge：joint feature-action OT 对齐；若升级到跨视角设定，§2.5 的 FGW 方案是候选。
- `2410.21795` TemporalOT / `2409.06615` RHyME：sequence-level 对齐——FGW 的结构项取时序距离阵时与其思想相通，可互为对照。
- `reports/synthesis.md` §4.2/§4.3：本笔记的消融协议与指标是其 baseline / 辅助指标清单的理论展开，无口径冲突。

## 并入主库建议

1. **papers.tsv 新增 6 行**（category 建议按三节拆：`uot_theory`、`gw_theory`、`neural_ot_evaluation`）：
   - `1607.05816` | Scaling Algorithms for Unbalanced Optimal Transport Problems | 2018 | uot_theory | https://arxiv.org/abs/1607.05816
   - `pmlr-v48-peyre16` | Gromov-Wasserstein Averaging of Kernel and Distance Matrices | 2016 | gw_theory | https://proceedings.mlr.press/v48/peyre16.html （**无 arXiv id**，建议用 PMLR id 并在备注标 HAL hal-01322992）
   - `1805.09114` | Optimal Transport for structured data with application on graphs (FGW) | 2019 | gw_theory | https://arxiv.org/abs/1805.09114
   - `2009.04266` | The Unbalanced Gromov Wasserstein Distance | 2021 | gw_theory | https://arxiv.org/abs/2009.04266
   - `2201.12220` | Neural Optimal Transport | 2023 | neural_ot_evaluation | https://arxiv.org/abs/2201.12220
   - `2106.01954` | Do Neural Optimal Transport Solvers Work? A Continuous Wasserstein-2 Benchmark | 2021 | neural_ot_evaluation | https://arxiv.org/abs/2106.01954
2. **INDEX.md** 增设小节"方法基础：OT 理论工具箱"，链接本笔记；建议置于"方法基础：跨状态空间…"之后。
3. **synthesis.md** 的 §4.2（baseline）与 §4.3（指标）可在下次修订时引用本笔记 §1.5 / §3.4 的消融协议（本次按硬性约束未改动任何现有文件，由主库维护者决定）。
4. **NOT（2201.12220）建议后续升级为全精读**并入 reports/：它与 SB Flow 是"同一 unpaired 翻译问题的 ε=0 与 ε>0 两条路线"，作为方法对照价值高；本笔记的半精读已覆盖其主定理与算法，全精读需补附录 F（假解排除条件）与 Appendix D（与翻译基线的定量对比）。
5. **后续扩充候选**（本次未收录，按优先级）：Séjourné et al. 2019 *Sinkhorn Divergences for Unbalanced OT*（arXiv:1910.12958，UOT 的去偏 divergence，修 §1.4-4 的可比性问题）；Fatras et al. *Minibatch OT* 系列（arXiv:2101.01792，陷阱 2 的系统分析）；Chapel et al. 2020 *Partial GW*（arXiv:2002.08276，UGW 的 TV 型对照）；Uscidda & Cuturi 2023 *Monge Gap*（arXiv:2302.04953，免 ICNN 的 map 正则，coupling 验证的替代工具）。
