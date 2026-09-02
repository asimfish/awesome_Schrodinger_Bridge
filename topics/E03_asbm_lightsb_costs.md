# E03 文献扩充：ASBM + LightSB 轻量 SB 求解器对照（NFE / 训练成本 / 适用维度）

## 选题定位

- 来源：缺口分析条目「ASBM + LightSB 轻量求解器对照」。主库现有 25 篇中，SB 求解器一线只覆盖了 I²SB（paired）、SB Flow（unpaired、连续时间 IMF 加速）、GSBM/DeepGSB/3MSBM（generalized / multi-marginal）与 adjoint 系列，**缺"低 NFE / 低训练成本"这一维度的系统对照**。
- 本文精读 2 篇：**ASBM**（Adversarial Schrödinger Bridge Matching，arXiv 2405.14449，NeurIPS 2024，已核验）与 **LightSB**（Light Schrödinger Bridge，arXiv 2310.01174，ICLR 2024，已核验）；并给出 **LightSB-M**（Light and Optimal Schrödinger Bridge Matching，arXiv 2402.03207，ICML 2024，已核验）的导航条目。
- 核心交付：ASBM / LightSB / LightSB-M / I²SB / DSBM 的 NFE、训练成本、可扩展维度、数据情形对照表，以及对 `SB-Render-Lite`（约 224–256px 机器人视觉图像、低推理延迟）的选型结论。
- 精读依据：两篇均通过 arXiv 官方 HTML 全文精读（含附录）；LightSB-M 读了 HTML 全文的方法与实验章。DSBM venue 经 papers.nips.cc 复核为 NeurIPS 2023（检索日期 2026-08-14）。

## TL;DR

1. **ASBM 把 IMF 从连续时间搬到离散时间（D-IMF）**：只学 N+1 个转移概率而非整条 SDE，用 DD-GAN 实现，推理只要 **NFE=4** 就在 CelebA 128×128 unpaired 翻译上把 DSBM（NFE=100）的 FID 从 37.8 降到 16.08（ε=1）；代价是约 7 天 × 1×A100 的对抗训练。这是目前**像素空间 unpaired SB + 个位数 NFE** 的最直接证据。
2. **LightSB 把 Schrödinger potential 参数化为高斯混合**，得到无模拟、无 minimax、单目标 KL 最小化的闭式求解器：**训练分钟级（4 CPU 核）、推理零神经网络前向**（条件计划是解析高斯混合），且有 SB 泛函逼近的 universal approximation 定理；但**只适用于中低维向量数据（实验 ≤1000 维 PCA / 512 维 ALAE latent），不能直接进像素空间**。
3. 对 `SB-Render-Lite` 的结论是**双轨制**：像素空间主线用 ASBM 型 D-IMF（唯一同时满足 unpaired + 低 NFE + 高维图像可行的路线，224–256px 需自行扩展验证）；LightSB/LightSB-M 作为**冻结视觉编码器 latent 上的零成本探针与 ε 扫描工具**——先用分钟级 CPU 实验确定 sim/real latent 可桥接性与 ε 区间，再投入 GPU 训 ASBM。DSBM 仅作 IMF 参照基线，且在大 ε 高维下明显退化，应避免作为主力。

---

## 精读一：Adversarial Schrödinger Bridge Matching（ASBM / D-IMF）

### 元信息

- 论文：Adversarial Schrödinger Bridge Matching
- 方法名：D-IMF（procedure）/ ASBM（算法实现）
- 作者：Nikita Gushchin*、Daniil Selikhanovych*、Sergei Kholkin、Evgeny Burnaev、Alexander Korotin（Skoltech / AIRI）
- 会议：NeurIPS 2024（任务给定已核验，采用；论文页脚与 OpenReview 一致）
- 链接：https://arxiv.org/abs/2405.14449 （arXiv HTML 全文精读，检索日期 2026-08-14）
- 代码：https://github.com/Daniil-Selikhanovych/ASBM
- 归类：unpaired translation；discrete-time IMF；adversarial（DD-GAN）实现；低 NFE SB。

### 动机

IMF（DSBM、Peluchetti 2023）交替做 Markovian / reciprocal 投影，收敛到 SB，但它学的是**连续时间 SDE**，推理必须走上百步数值积分（DSBM 典型 NFE≈100）。作者的问题是：能否保留 IMF 的理论保证，但把"学一条 SDE"换成"学有限个转移概率"，让推理天然只需几步？

### 方法核心

1. **离散化的理论基础（Theorem 3.1）**：固定 N 个中间时刻，定义离散 reciprocal 过程（端点联合分布 × 离散 Brownian bridge 内插）与离散 Markov 过程（链式转移分解）。证明：**既是离散 reciprocal 又是离散 Markov、且两端边缘为 p₀/p₁ 的离散过程，恰是 SB 在这些时刻的有限维投影**——把 Léonard 的连续时间刻画完整搬到了离散时间。
2. **D-IMF 程序（Theorem 3.6）**：交替做离散 reciprocal 投影（重插 Brownian bridge，只需能采样端点对）与离散 Markovian 投影（估计相邻时刻转移概率 q(x_{t_n}|x_{t_{n-1}})），KL 意义下收敛到 SB。注意离散 Markovian 投影**不是**连续投影的近似——两者一般不同，但 D-IMF 自身即收敛到同一 SB；理论上 N=1 也成立。
3. **高斯情形闭式更新（Theorems 3.7/3.8）**：任意维 D 下两种投影都有解析式（连续 IMF 的对应结果需解矩阵 ODE 且仅 D=1 有解析解）。据此做收敛数值分析：**经验上指数收敛**；收敛速度随 N 很快饱和（N=5 已足够），但随 ε 增大显著变慢（ε=1→10 约需 10 倍迭代）。
4. **ASBM 实现**：Markovian 投影 = 学转移概率，天然接 DD-GAN（时间条件生成器 G_θ(x_{t_{n-1}}, z, t_{n-1}) + 条件判别器，非饱和 GAN 损失）。为处理 Markovian 投影定义的方向不对称性，正/反两个方向交替训练（同 DSBM 实践）。初始 coupling 用独立耦合或 minibatch OT。图像实验 N=3（t=1/4,2/4,3/4）→ **推理 NFE=4**，外层迭代 K=5。作者特别指出：转移概率也可换成 normalizing flow、EBM、矩匹配等任何生成建模器——D-IMF 是一个框架而不只是一个 GAN 方法。

### 实验结论

- **CelebA 128×128 male→female（unpaired）**：ASBM NFE=4 得 FID 16.08（ε=1）/ 17.44（ε=10）；同等参数量、同为 5 轮外层迭代的 DSBM NFE=100 得 37.8 / 89.19。大 ε 下差距急剧拉大（DSBM 需积分噪声轨迹，FID 对残留噪声敏感）。
- **Colored MNIST 32×32（2↔3）**：ASBM FID 2.7/2.8（ε=1）、4.3/4.53（ε=10）；DSBM 为 6.2/5.3、58.7/59.9。
- **EOT/SB benchmark（Gushchin et al. 2023，D=2..128）**：cBW₂²-UVP 上 ASBM 在 ε=10 全面最好（D=64: 1.9 vs DSBM 68.9；D=128: 4.7 vs 362），ε=1 相当或更好，ε=0.1 略差于 SF²M-Sink。即**它真的在解 SB**，不只是翻译得好看。
- **训练成本**：CelebA 128×128 约 7 天 × 1×A100；CMNIST < 2 天；toy/benchmark 数小时。网络 42M 生成器 + 27M 判别器（DD-GAN CelebA-HQ 架构）。

### 局限

- 对抗训练的固有不稳定 / mode collapse 风险，靠 DD-GAN 的成熟配方缓解，但换到新域（如机器人视觉）需重新调参。
- 收敛只有存在性证明，无理论收敛速率（经验指数收敛）。
- ε 越大 D-IMF 需要的外层迭代越多（×10 量级），而 ε 恰是 sim2real 中"允许多大外观改变"的旋钮——好在低 ε（保几何）一侧对 ASBM 是省的。
- 每轮外层迭代要重训 GAN（正反两方向），总训练成本随 K 线性增长；128px 已需 7 A100·天，256px 预计数倍。
- FID 是唯一图像指标，无几何/语义保持评估（对机器人应用是关键缺口）。
- 后续工作 IPMF（Kholkin et al., arXiv 2410.02601，未核验 venue，检索日期 2026-08-14）宣称统一 IPF/IMF 并可混合 diffusion 与 GAN，可作后续跟踪条目。

### 与库内工作的关系

- **vs I²SB（ICML 2023，库内）**：I²SB 靠 paired 端点绕开迭代投影、把 SB 化成条件 diffusion；ASBM 面向 unpaired，从独立/minibatch 耦合出发用 D-IMF 迭代逼近真 SB。两者正交：I²SB 解决"有配对时怎么快"，ASBM 解决"无配对时怎么少走步"。
- **vs SB Flow（2409.09347，NeurIPS 2024，库内）**：SB Flow 在连续时间内把 IMF 变成在线 flow（α-IMF），省掉外层轮换、训练更高效，但**推理仍是 SDE 数值积分**（高 NFE）；ASBM 直接把时间离散化，推理 NFE=4。二者是"训练效率"与"推理效率"两条不同的加速轴，可对照消融。
- **vs GSBM（ICLR 2024，库内）**：GSBM 加任务状态代价 V_t 换取 task-aware 路径，但仍是连续时间 matching；D-IMF 目前只支持纯 Wiener 参照（动能项），**尚无把 V_t 塞进离散转移概率学习的现成方案**——若 SB-Render-Lite 需要几何保持代价，两条线尚不能直接合并，这是一个可发表的空白点。

---

## 精读二：Light Schrödinger Bridge（LightSB）

### 元信息

- 论文：Light Schrödinger Bridge
- 方法名：LightSB
- 作者：Alexander Korotin*、Nikita Gushchin*、Evgeny Burnaev（Skoltech / AIRI）
- 会议：ICLR 2024（任务给定已核验，采用）
- 链接：https://arxiv.org/abs/2310.01174 （arXiv HTML 全文精读，检索日期 2026-08-14）
- 代码：https://github.com/ngushchin/LightSB
- 归类：simulation-free EOT/SB 闭式求解器；Gaussian mixture 参数化；中低维向量数据。

### 动机

现有 SB 求解器（IPF、IMF、dual EOT、EBM 系）全部依赖多网络、多轮迭代、模拟轨迹或 MCMC——对"只想在中等维数据上解个 EOT/SB"的用户过重。SB 领域缺一个像聚类里的 k-means、离散 OT 里的 Sinkhorn 那样**简单但被理论撑住的默认基线**。

### 方法核心

1. **参数化**：由 Léonard 刻画，EOT 计划可写成 π*(x₁|x₀) ∝ exp(⟨x₀,x₁⟩/ε)·v*(x₁)，其中 v* 是（调整后的）Schrödinger potential。LightSB 把 v_θ 参数化为**非归一化高斯混合**（K 个分量，实践用对角协方差）。关键代数事实：高斯混合 × 高斯核仍是高斯混合，于是条件计划 π_θ(x₁|x₀) 与归一化常数 c_θ(x₀) 全部**闭式可算**（Proposition 3.2）。
2. **目标函数**：最小化 KL(π*‖π_θ) 等价于 L(θ)=E_{p₀}[log c_θ] − E_{p₁}[log v_θ]（Proposition 3.1）——不需要知道 π*，两项都能从样本 Monte-Carlo 估计，**普通 minibatch SGD 即可**；无 minimax、无内外层迭代、无轨迹模拟、无 MCMC。这是把 Mokrov et al. 2024 的 EBM 视角与 Gushchin et al. 2023（EOT benchmark 构造）的 sum-exp 参数化两个已有想法拼在一起，闭式归一化常数是使 EBM 退化为直接优化的关键。
3. **从计划到过程**：在 π_θ 内插 Brownian bridge 得 T_θ，闭式 drift g_θ（Proposition 3.3），且 KL(T*‖T_θ)=KL(π*‖π_θ)——静态误差直接控制动态误差。轨迹采样可用布朗桥中点插值做到**任意时刻、无离散化误差**，也不必顺序展开。
4. **理论**：Theorem 3.4——对紧支撑 p₀,p₁，标量协方差的高斯混合 v_θ 即可把 KL(T*‖T_θ) 逼到任意小（**SB 的第一个 universal approximation 定理**）；附录 A 给出有限样本泛化误差以参数化速率收敛。

### 实验结论

- **EOT/SB benchmark**：cBW₂²-UVP 大幅优于各神经求解器（如 D=128, ε=1：0.62 vs 此前最好 15.23）。**必须带上作者自己的警示**：benchmark 本身就是用同类高斯混合 potential 构造的，LightSB 有强 inductive bias；ASBM 论文也因此把 LightSB 系排除出其 benchmark 对比。此数字不能外推为"LightSB 比神经求解器准"。
- **MSCI 单细胞（PCA 50/100/1000 维）**：energy distance 与神经求解器持平或更好，而训练只需 **65–146 秒 × 4 CPU 核**；对照组 8–71 分钟 × V100。
- **ALAE latent（D=512）unpaired 人脸翻译**：FFHQ 1024×1024 图像经 ALAE 编码后在 latent 上解 EOT，**训练 < 1 分钟（4 CPU 核）**，解码得到 male↔female、adult↔child 翻译。推理只用 π_θ 条件采样（解析混合），不需要过程 T_θ。
- 表 1（求解器特性对比）里 LightSB 是唯一同时满足：非 minimax、非迭代、simulation-free 训练、闭式 drift、闭式条件密度、无模拟推理、universal approximation、小 ε 可用。

### 局限

- **表达力**：高斯混合之于 SB ≈ GMM 之于密度估计——作者明说**不适用于像素空间的大规模生成建模**；图像必须先进 latent。
- 极小 ε 下有数值不稳定（需算 ∝exp(1/ε) 的量，同 Sinkhorn 病），合理小 ε 实测可用。
- 目标非凸，可能落入局部极小（作者类比 k-means/GMM，实践影响不大）。
- 仅支持 Wiener 先验 / 二次代价——**不能像 GSBM 那样加任务状态代价**，也没有非二次代价的推广。
- benchmark 优势含 inductive bias（见上），跨域结论要靠 MSCI/ALAE 这类真实数据实验支撑。

### 与库内工作的关系

- **vs I²SB（库内)**：I²SB 用 paired 数据换 tractability，训练是 denoising 回归、推理数十步；LightSB 用参数化假设换 tractability，unpaired、训练分钟级、推理零网络前向。两者是"结构来自数据配对"vs"结构来自函数类"的两极。
- **vs SB Flow / DSBM（库内）**：同为 unpaired marginals-only，SB Flow/DSBM 用大网络学 drift、理论上能逼近任意 SB，但训练 GPU 级、推理高 NFE；LightSB 牺牲函数类丰富度，换来三个数量级的成本下降。SB Flow 报告里"sim RGB→real RGB 的方法底座"角色不能由 LightSB 直接接替（像素空间不可行），但**latent 版可以**。
- **vs GSBM（库内）**：GSBM 的价值在把几何/动作保持写进路径代价；LightSB 结构上做不到这一点（限制 4）。若在 latent 上用 LightSB，几何保持只能靠编码器本身的性质与后验筛选，而非 bridge 目标。
- **vs BDGxRL（2602.23737，库内）**：BDGxRL 在低维状态/动力学差距上用 diffusion SB；这类 ≤ 数百维的场景恰是 LightSB 的舒适区，可作其"重求解器是否必要"的消融基线。

---

## 导航条目：Light and Optimal Schrödinger Bridge Matching（LightSB-M）

- 论文：Light and Optimal Schrödinger Bridge Matching；作者：Nikita Gushchin、Sergei Kholkin、Evgeny Burnaev、Alexander Korotin（Skoltech / AIRI）
- 会议：ICML 2024（任务给定已核验，采用）；链接：https://arxiv.org/abs/2402.03207 （HTML 方法/实验章已读，检索日期 2026-08-14）；代码：https://github.com/SKholkin/LightSB-Matching
- **一句话定位**：把 bridge matching 与 LightSB 参数化结合——提出"optimal projection"：对**任意**输入计划 π（独立、minibatch OT、真 EOT 均可）构成的 reciprocal 过程 T_π，向"SB 过程集合"做一次 KL 投影即得真 SB（Theorem 3.1），从而**一步 matching、无迭代误差累积**；drift 用 LightSB 的高斯混合 potential 闭式参数化。
- 关键性质：Theorem 3.3 证明其 matching 目标与 LightSB/EgNOT 的静态目标至多差常数——**自动继承 LightSB 的 universal approximation 与泛化保证**；实验证实对输入计划几乎不敏感（benchmark 上 ID/MB/GT 三种初始计划结果几乎相同，如 ε=0.1, D=128：1.66/1.32/1.16，同表 DSBM 为 35）。
- 成本：MSCI 58–176 秒 × 4 CPU 核（DSBM 6.6–8.9 分钟 × V100）；ALAE latent 数分钟 CPU。
- **何时选它而不是 LightSB**：(a) 需要 bridge-matching 形式以便与 DSBM/SB Flow 同框架消融；(b) 想显式验证"计划无关性"（对 minibatch OT 偏差的免疫）；(c) 关心 drift 的直接拟合而非静态计划。表达力与适用维度与 LightSB 相同，选型时可视为同一档。

---

## 核心对照表：NFE / 训练成本 / 维度 / 数据情形

Venue 核验说明：I²SB＝ICML 2023、SB Flow＝NeurIPS 2024（库内报告已核验）；DSBM＝NeurIPS 2023（papers.nips.cc 复核，检索日期 2026-08-14）；其余三篇为任务给定已核验 venue。

| 维度 | **ASBM**（NeurIPS 2024） | **LightSB**（ICLR 2024） | **LightSB-M**（ICML 2024） | I²SB（ICML 2023，参照） | DSBM（NeurIPS 2023，参照） |
|---|---|---|---|---|---|
| 求解原理 | 离散时间 IMF（D-IMF），DD-GAN 学转移概率 | 高斯混合 potential + 单目标 KL 最小化（静态） | 高斯混合 potential + 一步 optimal projection（matching） | paired 端点 → tractable 条件 bridge（denoising 回归） | 连续时间 IMF，迭代 bridge matching 学 SDE |
| 推理 NFE | **4**（N=3；理论可 N=1 即 2 步） | **0 次网络前向**（条件计划为解析高斯混合，一次采样） | 同 LightSB（π_θ 闭式）；或按 drift 积分 | 2–10 步已接近最优（restoration）；一般 10–1000 | ≈100（SDE 数值积分；ASBM 实验设定） |
| 训练成本（论文实测） | CelebA 128²：≈7 天 × 1×A100；CMNIST：<2 天；42M+27M 参数；K=5 轮 × 正反两向 GAN | **分钟级 × 4 CPU 核**（MSCI 65–146s；ALAE <1min），无 GPU | 同量级（MSCI 58–176s × 4 CPU） | 与标准 conditional diffusion 训练同量级（simulation-free 回归） | 向量数据 V100 数分钟–十几分钟；图像域 GPU·天级，且多轮 IMF 迭代 |
| 已验证的最大规模 | 像素空间 128×128（≈4.9 万维）；benchmark 向量 D≤128 | 向量 ≤1000 维（PCA）/ 512 维（ALAE latent）；**像素空间不可行**（作者自述） | 同 LightSB（≤1000 维向量） | 像素空间 256×256（ImageNet restoration） | 像素空间可训（128² 由 ASBM 复现）；大 ε 高维明显退化（benchmark ε=10, D=128：cBW₂²-UVP 362%，ASBM 同设定 4.7%） |
| 数据情形 | **unpaired**（marginals only；可 minibatch OT 起步） | **unpaired** | **unpaired**（任意计划输入，含 paired 亦可） | **必须 paired**（端点成对） | **unpaired** |
| 高维图像可行性 | ✅ 直接像素空间，低 NFE | ❌ 仅 latent 间接可行 | ❌ 仅 latent 间接可行 | ✅（有配对时最强） | ⚠️ 可行但 NFE 高、大 ε 不稳 |
| SB 解的忠实度 | benchmark ε=1/10 最优或相当；ε=0.1 略逊 SF²M-Sink | benchmark 数字最好但含结构性 inductive bias，不能外推 | 同左，且证实对输入计划不敏感 | 不解全局 SB（paired 条件桥） | ε 大时误差大；有迭代误差累积 |
| 主要风险 | GAN 不稳定；每轮重训；无收敛速率理论 | 表达力受限于高斯混合；极小 ε 数值不稳 | 同 LightSB | 需要配对数据；OT-ODE 极限对高不确定性任务退化 | 推理慢；forgetting/误差累积 |

### 对 SB-Render-Lite 的选型结论（224–256px 机器人视觉、低推理延迟）

1. **像素空间主线 → ASBM 型 D-IMF**。这是唯一同时满足 unpaired + 个位数 NFE + 十万维级像素空间已验证的路线。NFE=4 意味着推理是 4 次 42M 生成器前向——对 224–256px 在线渲染管线（策略训练时的 on-the-fly 域迁移）是现实的延迟预算；DSBM/SB Flow 的 ≈100 步 SDE 积分不是。需要自担的风险：(a) 论文最高只做到 128×128，256px 需换更大 backbone 并预算数倍于 7 A100·天；(b) 对抗训练在机器人图像域（低纹理、强结构）需重调；(c) FID 之外必须补 keypoint/depth/inverse-dynamics 保持指标（论文完全没有）。
2. **ε 的选择对我们有利**。sim2real 渲染迁移希望小 ε（保几何、低多样性）：小 ε 下 D-IMF 收敛更快（外层迭代 ∝ ε），ASBM 的 CelebA ε=1 结果也优于 ε=10。建议 ε 扫描从小往大，以几何保持指标为止损。
3. **LightSB/LightSB-M 不是像素空间选项，但应立即用作两件事**：
   - **可行性探针**：把 sim/real 图像过冻结编码器（DINO/CLIP/policy encoder，512–1024 维），用 LightSB 在 latent 上解 SB——分钟级 CPU 就能回答"两域 latent 是否可低代价桥接、最优 ε 大致多大"，再决定是否投入 GPU 训 ASBM。ALAE 实验证明"预训练自编码器 + LightSB = 翻译模型"的组合成立；对机器人，瓶颈变成解码器（一般没有高保真图像解码器），因此 latent 路线更适合**特征级对齐**（直接把迁移后的 latent 喂给策略头，类似 EgoBridge 的思路但用闭式 SB 替代其 OT 损失），而非图像重建。
   - **消融基线**：任何"重求解器必要性"的声明，都应先被"LightSB on latent"这个近零成本基线打过——这也是审稿人友好的实验设计。
4. **I²SB 保留为 paired 上界**：若渲染器能对同一 scene 产出 paired sim/real-like 帧，I²SB（或其低 NFE 采样）仍可能优于所有 unpaired 方法，应作为 oracle 对照。
5. **DSBM 只作 IMF 参照**，避免用于大 ε 或作为主力：ASBM 与 LightSB-M 两篇的 benchmark 都显示其在 ε=10 高维下退化 1–3 个数量级。

---

## 并入主库建议

1. **新增 2 篇正式精读**：将本文件的 ASBM、LightSB 两节拆分为 `reports/2405.14449_asbm_discrete_imf.md` 与 `reports/2310.01174_lightsb.md`（沿用库内报告结构），`metadata/papers.tsv` 追加两行（category 建议分别为 `discrete_imf_adversarial_sb` 与 `light_closed_form_sb`），并同步下载 PDF 至 `papers/`、全文至 `texts/`。
2. **LightSB-M 以导航条目并入** `reports/deep_research_learning_resources.md` 或 synthesis 的"求解器谱系"小节，不必单独成篇（与 LightSB 同参数化、同适用域，独立信息量主要是 optimal projection 定理与计划无关性）。
3. **INDEX.md 建议新开一节「轻量 / 低 NFE SB 求解器」**，收 ASBM、LightSB（+LightSB-M 导航），并在「对当前 SB-Render-Lite 的直接启发」段补一句：ASBM 提供像素空间低 NFE 路线，LightSB 提供 latent 探针与消融基线。
4. **synthesis.md 的方法选型段建议补充本文件的对照表**（或其精简版），把"NFE / 训练成本"正式纳入选型维度——此前库内对比只覆盖 paired/unpaired 与约束表达力两轴。
5. **后续跟踪候选**（本轮未精读，均标注检索日期 2026-08-14）：IPMF（arXiv 2410.02601，IPF+IMF 统一框架，venue 未核验）；UNSB（Kim et al.，adversarial SB 的另一条线，ASBM 参考文献标注 ICLR 2024，本轮未独立复核）；若 SB-Render-Lite 需要几何保持代价，「把 GSBM 的状态代价引入 D-IMF 离散转移学习」目前文献空白，可作为方法贡献点立项。
