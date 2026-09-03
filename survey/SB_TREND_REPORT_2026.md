# Schrödinger Bridge 趋势报告 2026

> awesome_Schrödinger_Bridge · 综合调研报告 · 2026-09-01
> 覆盖：25 篇核心精读 + 20 份专题笔记 + 141 篇扩展条目（其中 93 篇为 2025–2026 新增：21 篇来自 WebSearch 直读会议页，72 篇来自四条 arXiv API 近 12 个月扫描——SB/bridge 277 篇、采样器/adjoint 98 篇、FM×OT 61 篇、SOC 微调 12 篇——后的人工筛选；检索日 2026-09-01/04）
> 证据口径：每条论断后括注来源——`R:` 逐篇精读（`reports/`）、`E:` 专题（`topics/`）、`arXiv:` 论文页、`ICLR/ICML:` 会议页。无来源的判断以「判断」标出。

---

## §0 TL;DR

1. **IMF 已取代 IPF 成为 SB 求解的默认范式，竞争焦点转到"一次训练 + 少步采样"。** DSBM 把 SB 求解写成交替的 Markov / reciprocal 投影，避开 IPF 的误差累积（arXiv:2303.16852）；SB Flow 用 α-IMF 在线微调、无需缓存（R:2409.09347）；RSBM 证明条件速度场在整个熵正则 ε 谱上函数形式不变，同一网络覆盖 SB（ε=1）到 OT（ε→0），视觉导航 3 步积分达 92% 成功率（arXiv:2604.05673）。
2. **随机最优控制（SOC）成为统一语言。** UniDB 证明 Doob h-transform 类桥（DDBM、I²SB 一族）是 SOC 终端罚系数 →∞ 的特例（ICML 2025）；Adjoint Matching 把 SOC 变成回归问题，衍生出 AS / ASBS / FAS / DAM / DASBS 整条采样与微调谱系（R:2504.11713、R:2506.22565、R:2511.06239、R:2602.07132、R:2602.08243）；2026 年 SMP 视角给它补上了严格地基（arXiv:2604.08580）。
3. **离散状态空间是 2025–2026 增长最快的分支，且从"翻译"转向"采样与微调"。** CSBM 证明离散时间 IMF 收敛（ICML 2025）→ catsbench 给出首个有解析解的离散 SB 基准（ICLR 2026）→ MadSBM 用 ESM-2 logits 做参考过程设计肽序列（arXiv:2601.22408）→ DAM 微调 LLaDA-8B 做数学推理、DASBS 做离散能量采样（R:2602.07132、R:2602.08243）。
4. **结构先验决定应用能否落地：多边缘、分叉、反馈、非平衡、函数空间。** 3MSBM / MSBM 处理多时间点快照（R:2506.10168、arXiv:2510.16587）；BranchSBM 学分叉速度场与质量增长，单分支 SBM 在细胞命运分叉上模式塌缩（ICLR 2026）；FAS 把 adjoint 采样推到函数空间（R:2511.06239）。
5. **在具身智能里，SB 的角色正从"数据翻译器"变成"策略本身"。** BridgePolicy 把观测嵌入 SDE、从观测先验而非高斯噪声出发采样动作，52 个仿真任务 + 5 个真机任务优于现有生成式策略（ICML 2026）；BDGxRL 用 DSB 对齐源/目标域转移动力学（R:2602.23737）。但整条线仍缺真机评估的统计协议（R10 审查结论，见 §5）。

---

## §1 全景：SB 方法族谱系

![SB lineage](../assets/sb_lineage.svg)

| 层 | 代表工作 | 一句话 | 精读/专题 |
|---|---|---|---|
| 经典 | Schrödinger 1932；Léonard 2014 综述；Chen–Georgiou–Pavon SIAM Review 2021 | SB = 以参考过程为基准的最小 KL 路径测度 = 动态熵正则 OT；离散化后每步 IPF 就是 Sinkhorn | README §1 |
| 神经求解器 I（IPF） | DSB（NeurIPS 2021）、SB-FBSDE（ICLR 2022）、DeepGSB（NeurIPS 2022） | 交替回归前/后向 score；SB-FBSDE 给出似然目标；DeepGSB 把平均场代价写进 SB | R:2209.09893、E01 |
| 神经求解器 II（IMF / bridge matching） | IDBM（JMLR 2023）、DSBM（NeurIPS 2023）、SB Flow（NeurIPS 2024）、BM²（TMLR 2024） | 桥混合 → Markov 化 → 迭代收敛到 SB；第一次迭代已是合法 transport | E01、E02、R:2409.09347 |
| 成对桥 | I²SB（ICML 2023）、DDBM（ICLR 2024）、DBIM（ICLR 2025）、UniDB（ICML 2025） | 从信息丰富的 source 出发的 Doob h-transform 桥；UniDB 把它们收进 SOC | R:2302.05872、E02 |
| 任务代价与结构 | GSBM（ICLR 2024）、DMSB/3MSBM、MSBM、FSBM、UDSB、BranchSBM | 状态代价、多边缘、反馈、非平衡、分叉 | R:2310.02233、R:2506.10168 |
| 轻量与少步 | LightSB / LightSB-M、ASBM、CDBM、LBM、UniDB++ | 闭式高斯混合、对抗式 D-IMF、consistency、1-NFE latent bridge、闭式反向解 | E03、E16 |
| 离散 | DDSBM（ICLR 2025）、CSBM（ICML 2025）、catsbench（ICLR 2026）、MadSBM | 离散时间 IMF 收敛性；首个 ground-truth 基准；序列设计 | README §2.5 |
| SOC 采样与微调 | PIS / DDS / CMCD → Adjoint Matching → AS / ASBS / FAS / DAM / DASBS；MDNS | 从"有样本"到"只有能量"；从连续到离散 | E06、E14、E15、R:2504.11713 起 |
| 与 FM 的统一 | Flow Matching、OT-CFM、[SF]²M、Stochastic Interpolants、Unified bridge framework（2025）、RSBM | σ / ε 是 SB↔FM 的连续旋钮；同一速度场参数化覆盖全谱 | E04、E05 |

---

## §2 五条主线的 2024H2–2026 进展

### 2.1 求解器：一次训练、少步采样、理论收口

- **从 IPF 到 IMF 的范式切换已完成。** IPF 每轮只保一个边缘、误差在迭代间累积；IMF 在 Markov 类与 reciprocal 类之间交替投影，两个边缘始终保持（arXiv:2303.16852；E01 谱系表）。DSBM 官方代码给出 DSBM-IPF 与 DSBM-IMF 两条实现，后者是当前 unpaired 翻译的边缘保持正统基线（E01 结论）。
- **α-IMF / SB Flow 去掉了缓存与双损失。** 在线微调、可用单个双向网络，α=1 退回 IMF（R:2409.09347 方法核心）。
- **RSBM 把"SB 还是 FM"变成一个连续参数的选择。** 条件速度场在 ε∈(0,1] 上结构不变，降低 ε 线性减小速度方差、改善粗步 ODE 积分稳定性；3 步 92% 成功率、94.5% 余弦相似度，比基线少 3.8× 函数求值，无蒸馏（arXiv:2604.05673 摘要与结论）。
- **成对桥的少步化走了两条路：** 训练侧的 consistency（CDBM，NeurIPS 2024）与蒸馏到 1 NFE 的 latent bridge（LBM，ICCV 2025；E16 给出 paired/unpaired 两种部署裁决）；采样侧的免训练闭式解（DBIM ICLR 2025；UniDB++ 把 UniDB 反向 SDE 写成精确闭式 + 数据预测 + SDE-Corrector，5–10 步、最多 20× 加速，并在特定条件退化为 DBIM，arXiv:2505.21528）。
- **理论收口：** 2025 年的统一框架把 FM、minibatch-OT FM、minibatch SB-FM 与 DSBM 写成同一 bridge 问题的特例（arXiv:2503.21756）；2026 年出现两份教程级文献——SB 生成建模基础指南（arXiv:2603.18992）与 Peyresq 暑期学校讲义（arXiv:2606.30053，明确指出 IMF 收敛到熵最优耦合而 rectified flow 不收敛到 OT 耦合）。

### 2.2 结构化 SB：把领域知识写进路径

- **任务代价**：GSBM 允许在路径动能之外加可微状态代价，把 keypoint/depth、逆动力学、安全约束写进 transport 目标（R:2310.02233；synthesis §3.4）。
- **多边缘**：3MSBM 学满足多个位置约束的测度值样条，在 Lotka-Volterra、墨西哥湾洋流、scRNA-seq、北京空气质量上对比 DMSB / SBIRR / smoothSB / MMFM（R:2506.10168）；MSBM 走另一条路——逐区间局部 SB + 共享全局控制参数化，中间边缘全部满足且轨迹连续（arXiv:2510.16587）。
- **分叉**：BranchSBM 参数化多条时变速度场 + 各分支增长过程，目标是 Unbalanced CondSOC 之和；在 Clonidine 扰动数据上重建多簇终态，单分支 SBM 无法区分高维主成分上分离的簇，可扩展到 150 PCs（ICLR 2026 项目页）。
- **反馈与非平衡**：FSBM 把 SB 求解写成闭环反馈控制（ICLR 2025 Oral）；UDSB 允许质量不守恒（arXiv:2306.09099）；BranchSBM 的增长网络实质上把非平衡性变成分支间的质量分配；CytoBridge 把平均场 SB 推广到非归一化分布，用四个网络显式建模细胞转移、增殖与相互作用（NeurIPS 2025，arXiv:2505.11197）。
- **约束域**：Reflected SBM 以单位超立方体上的反射布朗运动为参考过程，把 (α-)IMF 的部分 simulation-free 训练搬到反射 SB，样本保证落在数据域内且开销可忽略（arXiv:2607.03626）。
- **函数空间**：SOC for diffusion bridges in function spaces（NeurIPS 2024）→ FAS 把 adjoint 采样推到无限维（R:2511.06239）。

### 2.3 离散状态空间：从翻译到采样与微调

- **理论**：CSBM 证明离散时间 IMF（D-IMF）在有限空间上收敛到 SB，覆盖 VQ 码本、文本 token、分子原子类别（ICML 2025，PMLR 267）。DDSBM 则用连续时间 CTMC 做图变换（ICLR 2025）。
- **基准**：catsbench 构造有解析解的离散分布对，第一次让离散 SB 求解器可以被严格评测；副产品 DLightSB / DLightSB-M（离散版闭式求解器）与 α-CSBM（ICLR 2026）。
- **应用**：MadSBM 把肽设计写成氨基酸编辑图上的受控 CTMC，参考过程来自冻结 ESM-2 logits，学时变控制场得到低作用量路径，并首次在 SB 生成模型上做离散 classifier guidance（arXiv:2601.22408）。
- **微调与采样**：DAM 提出"离散 adjoint"估计量，绕开不可微状态空间，微调 LLaDA-8B-Instruct 做数学推理（R:2602.07132）；DASBS 指出 AM 的核心机制与状态空间无关，把 AM 与 ASBS 统一推到离散空间，并识别出循环群结构是必要条件（R:2602.08243）；MDNS 用 CTMC 随机控制训练 masked 扩散采样器，在 Ising / Potts 高维目标上大幅优于学习型基线（arXiv:2508.10684）；DRAKES 是 Gumbel-Softmax 反传路线的对照（arXiv:2410.13643）。

### 2.4 采样与控制：Adjoint 谱系的三年

- **源头**：PIS / DDS / CMCD 把"从能量采样"写成 SOC，但受全轨迹反传、on-policy 耦合、先验受限三大瓶颈制约（E14）。
- **转折**：Adjoint Matching（ICLR 2025 Spotlight）用 memoryless 噪声调度 + lean adjoint 回归，把 reward 微调和采样都变成回归（E06）。
- **谱系**：AS 首次做到梯度更新数远多于能量评估数，扩展到 SPICE 分子构象的 amortized 采样（R:2504.11713）；ASBS 允许任意 source 分布（R:2506.22565，NeurIPS 2025 Oral）；NAAS 把退火参考动力学作为 base SDE，让参考轨迹自带朝目标推进的信息（NeurIPS 2025，arXiv:2506.18165）；WT-ASBS 把 well-tempered metadynamics 的在线偏置沿集体变量加进 ASBS，首次用扩散采样器刻画含键断裂/形成的反应面（ICLR 2026，arXiv:2510.11923）；FAS 上函数空间；DAM / DASBS 上离散空间。
- **竞品与评测**：iDEM / NETS / Sendera 在"无偏 + 全模态覆盖"维度仍是 adjoint 线的短板；评测需补 EUBO、前向指标与 mode-coverage 口径（E15）。
- **横向扩张（2026 上半年最密集的一条线）**：AM 从"reward 微调"进入 RL 主干——QAM 把"对 Q 函数优化 diffusion/flow 策略"重写为带学习 critic 的 memoryless SOC（arXiv:2601.14234）；TRQAM 用信赖域控制路径空间 KL 抑制病态 critic 引发的崩溃（arXiv:2605.27079）；ME-AM 给离线 RL 加最大熵放开支撑绑定（arXiv:2605.06156）；在线最大熵 RL 的 diffusion 策略也用 AM 做 simulation-free 训练（arXiv:2606.22630）；Reinforce AM 证明 KL 正则 reward 最大化只倾斜干净端点分布、噪声律不变，把 RL 后训练保留为回归结构（arXiv:2605.10759）；Efficient AM 与确定性 AM 削减全轨迹模拟与反向 adjoint 开销（arXiv:2605.11480、2605.06583）；CAM 把离散 adjoint 用于无监督组合优化求解器（ICML 2026，arXiv:2605.30920）；平均场控制的样本级 adjoint 回归与之同源（arXiv:2604.06675）。Domingo-Enrich 等给出统一视角：采样与 reward 微调都是对 base 密度的指数倾斜，AM/AS 与 Novel Score Matching 的梯度方差有限而 Target/Conditional Score Matching 无界（arXiv:2605.00229）；Tilt Matching 用随机插值给出更低方差的同类目标（arXiv:2512.21829）。
- **离散与探索**：Data-to-Energy 处理一端只有能量的 SB（ICLR 2026）；PDNS 用近端点法抗模式塌缩（ICLR 2026）；离散扩散采样器与桥的 off-policy 训练（ICML 2026）；MetaDNS 是 WT-ASBS 的离散对应（ICML 2026）。
- **竞品生态：Boltzmann 生成器**在同一年集中升级——自回归骨干（ICML 2026 Spotlight）、粗粒化 + 重加权（ICML 2026）、退火 MC 校正（TMLR 2026）、off-policy log-dispersion 正则、少步似然蒸馏（SCALLOP）、Jeffreys 散度抗塌缩，以及面向无序材料的多模态基础采样器 JANUS；Reinforced SMC 把 SMC 与最大熵 RL 采样器打通（ICML 2026）。这些是 adjoint 线在"无偏 + 模态覆盖"口径上必须对标的对象（E15）。
- **理论**：SMP 视角给出控制相关漂移/扩散与凸运行成本下的一般 Hamiltonian adjoint matching 目标，证明其期望的一阶变分与原 SOC 目标一致，lean adjoint 是状态无关扩散下的特例，AM 可解释为连续时间的逐次逼近法（arXiv:2604.08580）。

### 2.4b arXiv 近 12 个月扫描补充（2025-09 → 2026-09，277 篇候选）

- **生成式策略的熵正则改用广义 SB 表述**：FLAC 把最大熵 RL 写成相对高熵参考过程的 GSB，用速度场动能惩罚代替不可得的动作 log 密度（arXiv:2602.12829）；GSB-MDPO 把 on-policy 生成式策略优化写成状态条件生成路径上的 GSB，用路径空间镜像下降替代 PPO 式近端更新（arXiv:2603.21621）；DBC 用 diffusion bridge 建模 Q 值逆 CDF 做分布式 critic（arXiv:2602.05783）。这三篇说明 SB 在 RL 里的角色从"数据"进一步走到"策略与价值的正则语言"。
- **规划类应用成形**：BridgeDrive 用锚点引导的 diffusion bridge 策略做自动驾驶闭环轨迹规划（ICLR 2026，arXiv:2509.23589）；XFlowMP 用 SB 做任务条件运动规划（arXiv:2512.00022）；MAPF 被写成带 Markov 结构的多边缘 OT，大规模时用 SB 熵正则化迭代求解（ICML 2026 Spotlight，arXiv:2605.10917）。
- **参考过程与终端约束的设计理论**：软约束 SB 证明罚函数替代硬终端约束后解的存在与收敛（arXiv:2510.11829，与 UniDB 同一思路）；PRISM 给出桥参考过程设计理论——精确 drift + 无限步下任何参考都恢复真后验，参考只在有限步预算下才重要，最优噪声谱正比于传感器摧毁的信息谱（arXiv:2608.06893）；Twisted SBM 把参考换成带时变势的 twisted 布朗运动（arXiv:2607.16987）；NADB 发现 score-matching 式训练在靶端欠拟合并给出噪声对齐修法（CVPR 2026，arXiv:2605.28962）。
- **少步与免训练采样器继续增多**：DBMSolver 用 DBM 半线性结构做指数积分器，NFE 最多减 5×（CVPR 2026）；E-Bridge 用更短时域 + 熵正则起点 + consistency 学单步映射（ICLR 2026）；SB Mamba 一步语音增强（Interspeech 2026）。
- **多边缘与相互作用系统的第三、四条路**：MMtSBM 以因子化 IMF 从 unpaired 快照学多边缘（ICML 2026，arXiv:2510.01894）；EntangledSBM 学相互作用多粒子系统的一二阶动力学（arXiv:2511.07406）；非局部平均场 SB 用神经代理替代二次复杂度相互作用项（arXiv:2606.04265）；Curly-FM 指出最小作用量只能学梯度场、给出非梯度周期动力学的匹配目标（NeurIPS 2025）。
- **几何与约束域**：Contact Wasserstein 测地线放开能量守恒（ICLR 2026）；李群、子黎曼流形上的 SB 与反射 SB（arXiv:2603.14049 等）；Reflected SBM（§2.2）。

### 2.5 应用面：图像、具身、科学

- **图像修复/翻译**：I²SB（2–10 NFE，R:2302.05872）→ DDBM → UNSB（unpaired）→ DBIM / CDBM（少步）→ UniDB（SOC 统一 + 可调终端罚改善细节）→ UniDB++（免训练加速）→ LBM（1 NFE latent）。CVPR 2026 的两篇继续在 DDBM 框架内做结构保真：RDBM 用成对残差调制噪声、只扰动退化区域，五类修复任务平均 +1.55 dB（arXiv:2510.23116）；Bi-Bridge 利用高斯桥均值对端点的对称性做双向一致性训练（CVF）。E17 指出 DDIB 式"两段桥拼接 ≠ 跨域 OT"、精确 cycle consistency ≠ 对齐，语义漂移是对机器人数据最危险的失败模式。
- **具身**：表示对齐范式（EgoBridge、Guided OT co-training：在 joint feature-action 分布上对齐，R:2509.19626、R:2509.18631）与生成式 transport 范式（SB Flow、BDGxRL）不可平替，耦合方式需要显式定义（synthesis §4.4）。2026 年的新变量是"策略即桥"：BridgePolicy 把观测嵌入 SDE、从观测先验出发采样动作（ICML 2026），RSBM 用 ε 谱做少步导航策略（arXiv:2604.05673）。
- **科学**：React-OT 用确定性 OT 生成化学反应过渡态（Nat. Mach. Intell. 2025，R:2404.13430）；SBUnfold 在 60 万仿真样本 + 1000 条伪数据下保持稳定（R:2308.12351）；单细胞轨迹推断是结构化 SB 的主战场（DMSB → 3MSBM → MSBM → BranchSBM）；Adjoint 线主攻分子构象与 Boltzmann 采样。
- **语音**：Bridge-TTS 用 SB 替代扩散做 TTS（arXiv:2312.03491）、SB 语音增强（Interspeech 2024）。

---

## §3 Insight

**I1 · ε（或 σ）已成为方法选择的主旋钮，而不是超参数。** SB Flow 的 α、[SF]²M 的 σ、RSBM 的 ε、E04 里 entropic ε 谱（ε=2σ² 恰对应 SB）说的是同一件事：SB 与 FM 是一条连续谱的两端，同一网络参数化覆盖全谱。工程含义：先在大 ε 下用 SB 的边缘保持性学粗耦合，再向小 ε 走换取直轨迹与少步（E04、arXiv:2604.05673）。这条谱上还缺的是**自动选 ε 的准则**——判断。

**I2 · SOC 把"桥"和"控制"两套语言合并了，Doob h-transform 是它的一个极限点。** UniDB 的结论（终端罚 →∞ 即 h-transform）解释了为什么 DDBM/I²SB 类方法会过度平滑细节，也给出了修法（可调终端罚）；Adjoint Matching 反过来把 SOC 的求解变成 bridge matching 式回归。两个方向合流后，"选参考过程 + 选终端罚 + 选回归目标"成了统一的设计三元组（ICML 2025；E06；arXiv:2604.08580）。

**I3 · 离散 SB 的爆发由两件事驱动：dLLM 的兴起与 ground truth 的出现。** 没有解析解就无法说"求解质量"，catsbench 补上了这一块（ICLR 2026）；DAM/DASBS/MDNS 则把离散 SB 从"图像 VQ 码本翻译"带进了 LLM 微调与统计物理采样（R:2602.07132、R:2602.08243、arXiv:2508.10684）。下一步的竞争在**参考过程的选择**——MadSBM 用蛋白语言模型 logits 当参考，是把领域先验塞进 SB 的范例（arXiv:2601.22408）。

**I4 · 结构先验决定生物应用的成败，而不是求解器精度。** 多边缘、分叉、非平衡三者缺一，单细胞任务就会模式塌缩或质量守恒失真（BranchSBM 对单分支 SBM 的对照，ICLR 2026；UDSB，arXiv:2306.09099）。3MSBM 与 MSBM 在同一年给出两种不同的多边缘构造（相空间样条 vs 逐区间局部 SB），说明这个问题还没收敛（R:2506.10168、arXiv:2510.16587）。

**I5 · 具身领域的 SB 正在从"数据管道"进入"策略头"，但评测没跟上。** BridgePolicy 和 RSBM 把 bridge 当策略采样器，收益来自更好的先验（观测）而不是更好的翻译（ICML 2026、arXiv:2604.05673）。原库 R10 审查指出全库没有任何真机评估的统计协议（试次数、种子、置信区间）；E11 基于 SimplerEnv 提出"四层指标 + 两档协议"并给出功效计算（约 170 rollouts/臂）。这一缺口在 2026 年的新论文里依然存在——判断。

**I6 · 评测基础设施是整个领域的短板：连续高维 SB 仍无 ground truth。** 离散有 catsbench，能量采样有 DW4/LJ13/LJ55/alanine dipeptide 与 SPICE 构象（E15、R:2504.11713），unpaired 翻译只有 FID + NFE。E19 的结论适用于全部：coupling 质量必须用独立于视觉质量的指标验证，类别失衡时 balanced coupling 数学上必然错配。

**I7 · 轻量闭式求解器的角色是探针与教师。** LightSB / LightSB-M 分钟级训练适合 latent 上做 ε 扫描（E03）；catsbench 的 DLightSB 把同一思路搬到离散（ICLR 2026）；LBM 的 1-NFE 蒸馏与 CDBM 的"先训桥再压缩"说明重模型最终要靠轻模型部署（E16）。

**I9 · Adjoint Matching 正在成为"生成式策略 RL"的默认优化器候选。** 2026 年 1–6 月连续出现 QAM / TRQAM / ME-AM / MaxEnt-AM / Reinforce AM 五条 RL 路线与 CAM（组合优化）、MFC（平均场控制）两条外延，共同点是用"回归 adjoint 目标"替代"反传多步去噪"来利用 critic 或 reward 的一阶信息（arXiv:2601.14234 起）。这把 SB/SOC 社区的工具直接送进了 RL 主干；对具身方向的含义是：diffusion/flow 策略的在线微调不再必须走 DDPO 式 policy gradient——判断，需在真机协议下验证（E11）。

**I8 · 开源生态呈三簇：** Meta/FAIR（Adjoint 系、flow_matching、GSBM）、Guan-Horng Liu 及合作者（SB-FBSDE、DeepGSB、GSBM、I²SB、ASBS、FAS、DASBS）、Korotin 组（LightSB 系、ASBM、CSBM、catsbench、Neural OT）。三簇分别押注"SOC 与采样"、"控制视角的 SB"、"闭式与基准"（README §5 代码库；仓库 star 与更新日期见 `metadata/resources.tsv`）。

---

## §4 对具身 sim2real（SB-Render-Lite 一类项目）的启示

1. **先定范式再选方法。** 表示对齐（EgoBridge / Guided OT）与生成式 transport（SB Flow / GSBM / BDGxRL）不可平替；叠加时按 synthesis §4.4 的归因口径分别消融（synthesis §4.4）。
2. **unpaired 主线用 DSBM-IMF 起步，用 ε 谱做少步化。** DSBM-IMF 是边缘保持正统基线（E01）；RSBM 的结论表明可以在同一网络上从 ε=1 走到小 ε 换取 3 步部署（arXiv:2604.05673）。
3. **paired 对照双基线：I²SB + DDBM，再用 UniDB 的可调终端罚查细节是否过平滑**（E02、ICML 2025）。
4. **把任务约束写进代价而不是后处理。** GSBM 的状态代价可以承载 keypoint/depth/逆动力学一致性（R:2310.02233）。
5. **"策略即桥"是 2026 年值得一试的替代路线。** 若真机数据极少，BridgePolicy 式的观测先验采样可能比翻译数据更直接（ICML 2026）——判断，需在 SimplerEnv 协议下验证（E11）。
6. **评测先于方法。** 采用 E11 的四层指标与功效计算；coupling 质量用独立指标（E19）；所有视觉指标服从下游 policy success（synthesis §5）。

---

## §5 未来 12 个月观察清单

| 观察点 | 判据 | 触发动作 |
|---|---|---|
| ε 自动选择 | 出现按任务/数据自适应 ε 的准则并在 unpaired 翻译上验证 | 更新 I1；补入 README §2.6 |
| 连续高维 SB 基准 | catsbench 式解析解基准扩展到连续高维 | 更新 I6；补入 README §5 |
| dLLM 的 SB/SOC 微调 | DAM/DASBS 之外出现第二条独立路线并在 ≥7B 模型验证 | 更新 §2.3 |
| 真机统计协议 | 具身 SB 论文报告试次数/种子/置信区间 | 更新 I5 |
| 放榜后复核 | 2105.11739、2602.23737 两条 preprint；FAS/DASBS 的 PMLR 页码；MSBM、MadSBM、RSBM、AM-SMP 等标 `arXiv` 的 2025–26 条目 | `metadata/*.tsv` 更新 venue，重建 README |
| 雷达复扫 | 每季度 `python3 scripts/arxiv_scan.py --months 4`，新高分未收录条目 ≥ 10 | 人工筛选入 `extended.tsv` |

---

## §6 方法与证据

- **来源层次**：核心 25 篇有逐篇精读（经 10 路审查 + 修复，见原库审查记录摘要）；20 份专题笔记；扩展 141 条中 48 条为专题反复引用的基础论文（arXiv ID 与标题经 Semantic Scholar 批量核验，47/49 通过，1 条记忆错误已剔除），21 条为 WebSearch 直读会议页新检索，72 条来自四条 arXiv API 近 12 个月扫描（SB/bridge 277 → 39；采样器/adjoint 98 → 27；FM×OT 61 → 7；SOC 微调 12 → 3；人工按相关性筛选，venue 只取 arXiv Comments 明示者；完整雷达表见 `survey/raw/S2_arxiv_scan_*.md`）。
- **未核验即不写**：作者不确定的条目作者栏留空；venue 无会议页证据的写 `arXiv`；XFlowMP、FreeBridge 等仅在搜索摘要中出现、未能打开原页的条目未收录（Reflected SBM 与 CytoBridge 在第二轮检索中已打开原页并收录）。
- **局限**：检索日 arXiv API 与 Semantic Scholar 均出现限流，2025H2–2026 的覆盖以 WebSearch 命中为主，不保证完备；数值均取自摘要或论文页可见内容，未复现。
- **复现**：`python3 scripts/build_readme.py` 重建 README；`python3 scripts/s2_verify.py --ids ...` 核验新 ID；`bash scripts/translate_batch.sh` 生成译本；`python3 scripts/qa_table.py` 重建译本 QA 表。
