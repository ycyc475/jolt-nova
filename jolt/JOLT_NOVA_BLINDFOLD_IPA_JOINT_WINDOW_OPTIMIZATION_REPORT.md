# Jolt-Nova BlindFold IPA Joint-Window 优化报告

> 日期：2026-08-10
> 基线：`6063c2d`（IPA range commitment / zero-copy folding）
> 环境：Slurm `comput03` 独占节点，64 个单线程物理核，Rust 1.95

## 1. 瓶颈与优化

新增的 `JOLT_NOVA_PROFILE_IPA=1` round-level profiling 显示，在主 Bn256 IPA（2,097,152 generators）中，21 轮 generator folding 共耗时 18.293 秒，占 20.315 秒 evaluation argument 的主要部分。原地复用 scalar/generator buffer 的 A/B 只有 0.03% BlindFold 改善且绝对峰值 RSS 略升，因此该方案已回退。

最终保留的优化针对每个 generator pair 的 `w1 · L + w2 · R` 实现 variable-time joint two-bit window：两个公开 Fiat–Shamir 挑战共享一条 doubling chain，避免两次独立 scalar multiplication 的重复 doubling。generator 和挑战均为公开值，因此该 variable-time 路径不处理秘密 witness scalar；Blitzar feature 继续使用原后端。

同时增加多曲线等价性单元测试，将 joint-window 结果与原两项 MSM 结果比较；生产正确性由完整 Bn256/Grumpkin 递归证明进一步验证。

## 2. A/B 结果

所有实验均在同一独占节点顺序执行，双方 warmup 1 次后测量；全部样本通过 production ZK、native Jolt、streaming proof、完整递归验证及 `max_resident_trace_blocks <= 2` 校验。

| workload | 指标 | baseline | optimized | 改善 |
|---|---|---:|---:|---:|
| Fibonacci-32（3 runs） | BlindFold | 35.982 s | 23.119 s | **35.75%** |
|  | Spartan prove | 23.528 s | 10.791 s | **54.14%** |
|  | 端到端 | 43.902 s | 30.565 s | **30.38%** |
|  | 递归绝对峰值 RSS | 3752.910 MiB | 3716.699 MiB | **0.96%** |
| SHA3-chain-100（2 runs） | BlindFold | 36.343 s | 23.230 s | **36.08%** |
|  | Spartan prove | 23.802 s | 10.855 s | **54.40%** |
|  | 端到端 | 54.871 s | 41.626 s | **24.14%** |
|  | 递归绝对峰值 RSS | 4020.260 MiB | 3990.574 MiB | **0.74%** |

单次同作业 cold warmup 中，Fibonacci 总时间由 49.336 秒降至 36.669 秒；SHA3 由 60.342 秒降至 47.706 秒。该优化同时改善冷、热路径，但冷启动仍需生成 deterministic Pedersen generators。

## 3. Round-level 复核

主 Bn256 IPA profile：

| 指标 | baseline | joint-window | 改善 |
|---|---:|---:|---:|
| 21 轮 generator folding | 18.293 s | 5.537 s | **69.73%** |
| evaluation argument | 20.315 s | 7.570 s | **62.74%** |

收益准确落在目标 generator folding，commitment、Nova setup 和协议其他阶段未发生异常变化。

## 4. 客观结论与后续

本轮是 BlindFold 主线的显著进展，但主线尚未完成：优化后 BlindFold 仍约 23.2 秒，其中 Spartan prove 约 10.8 秒，Nova setup 约 5.3 秒，Spartan setup 仍有固定成本。

下一步建议：

1. 比较 joint-window 2/3/4-bit 及 signed joint sparse form，继续压缩约 5.5 秒 generator folding；
2. 重新评估 batch normalization，因为 scalar multiplication 大幅加速后，逐点 affine conversion 的相对占比已经上升；
3. 优化约 1.7 秒的 IPA `L/R` commitment，并评估大轮次 NUMA/线程布局；
4. 曲线计算接近平台上限后，进入 fixed-shape BlindFold 电路改造，以安全复用 Nova/Spartan setup。

完整实验 artifact 位于服务器：

- `jolt/benchmark-runs/stage19-ipa-joint-w2-final/`
- `jolt/benchmark-runs/stage19-ipa-joint-w2-sha3-final/`
- `jolt/benchmark-runs/stage19-ipa-joint-round-profile/`
