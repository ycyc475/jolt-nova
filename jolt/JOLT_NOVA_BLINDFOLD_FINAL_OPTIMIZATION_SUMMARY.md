# Jolt-Nova BlindFold 优化主线最终总结

> 日期：2026-08-11
> 最终分支：`codex/stage-9.10`
> 最终代码：`6abaa72`

## 1. 完成状态

BlindFold 的**工程优化主线已完成**。这里的“完成”表示：现有协议、production ZK、透明 Pedersen/IPA 安全模型和动态 R1CS shape 设计不变的前提下，已完成主要热点定位、候选实现、正反顺序 A/B、跨 workload 验证以及无效候选回退。它不表示 BlindFold 已达到密码学或专用硬件上的理论最优。

## 2. 最终累计结果

环境为 Slurm `comput03` 独占节点、64 个单线程物理核；Fibonacci-32、block size 256，双方 warmup 1 次后各测量 3 次。全部样本通过 production ZK、native Jolt、streaming proof、完整递归验证及 resident blocks 限制。

| 指标 | Stage 19 初始 baseline | 最终版本 | 累计改善 |
|---|---:|---:|---:|
| BlindFold | 40.822 s | 21.151 s | **48.19%** |
| Spartan prove | 24.097 s | 8.806 s | **63.45%** |
| Nova setup | 9.395 s | 5.321 s | **43.36%** |
| 端到端总时间 | 48.676 s | 28.746 s | **40.95%** |
| 递归绝对峰值 RSS | 4016.991 MiB | 3446.635 MiB | **14.20%** |
| 递归 RSS 增量 | 3468.029 MiB | 2869.603 MiB | **17.26%** |

完整累计 artifact：`jolt/benchmark-runs/stage19-blindfold-cumulative-final/`。

## 3. 已保留的优化

| 提交 | 优化 | 主要作用 |
|---|---|---|
| `9534da4` | deterministic Pedersen generator 进程内缓存与 `Arc` 共享 | 热路径 Nova setup、内存 |
| `6063c2d` | IPA range commitment 与 zero-copy folding | 数据搬运、分配、Spartan prove |
| `9a4177a` | 公开 challenge 的 joint two-bit window | generator folding、Spartan prove |
| `1ff4b4a` | BN256 GLV decomposition + 四路 wNAF | generator folding、Spartan prove |
| `6abaa72` | dense MSM 直达路径 + setup-bound padded shape 共享 | IPA commitment、矩阵克隆、峰值内存 |

共享 padded shape 只发生在同一次 setup 生成的 pk/vk 之间，不跨 `shape_id` 复用。ProverKey 落盘时不序列化 cache；反序列化后自动回退到原 padding 路径。

## 4. 已验证但未保留的候选

- generator batch normalization：分块内串行计算导致明显退化；
- 两阶段并行 batch normalization：generator fold 与端到端均无收益，且增加临时内存；
- shared scalar-window schedule、joint-window table 微调：正反顺序收益消失，属于噪声；
- IPA scalar/generator buffer 原地复用：时间无稳定改善，峰值内存未下降；
- 完整 shape-specific setup/key 跨证明缓存：不同 public inputs 会改变 R1CS 系数与 `shape_id`，不满足正确性边界。

## 5. 剩余瓶颈与后续边界

最终版本中，BlindFold 仍约 21.2 秒：Nova setup 约 5.3 秒，Spartan prove 约 8.8 秒；主 Bn256 evaluation argument 约 5.2 秒，其中 generator folding 约 3.5 秒。继续进行小型 allocation、window-table 或 batch-normalization 微调，预期收益已低于稳定测量噪声。

下一步若继续显著下降，需要进入新的研究/架构阶段：

1. 将 BlindFold 电路改造为 fixed-shape，使不同 public inputs 不再改变 R1CS 系数，从而安全复用 Nova/Spartan setup；
2. 引入曲线专用 SIMD/GPU/Blitzar 后端，或继续研究更强的 BN256 endomorphism/MSM 实现；
3. 评估 HyperKZG 等不同 commitment backend，但需要单独说明 Powers-of-Tau、证明格式与信任模型变化；
4. 若部署是一证明一进程，实现带版本、曲线、label、长度和完整性校验的持久化 generator cache，改善冷启动。

这些方向会改变电路结构、后端或部署模型，不再属于本轮低风险 BlindFold 工程优化主线。
