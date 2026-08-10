# Jolt-Nova BlindFold IPA Folding 优化报告

> 日期：2026-08-10
> 基线：`9534da4`（Pedersen generator cache）
> 环境：Slurm `comput03` 独占节点，64 个单线程物理核，Rust 1.95

## 1. 优化内容

本轮继续优化 BlindFold 最大瓶颈 `spartan-prove` 中的 Bn256 IPA evaluation argument，保持证明关系、transcript 和验证逻辑不变：

1. generator folding 直接索引原 key 的左右半区，不再为每轮复制两个子 key；
2. 两点 folding 使用栈上固定数组，消除逐点临时 `Vec`；
3. IPA 的 `L/R` commitment 直接基于 generator range 和一个附加 generator 计算，不再拼接 commitment key 和 scalar vector；
4. 请求完整 generator 前缀时复用 `Arc` 存储，避免一次 2,097,152 点的复制。

## 2. A/B 结果

两组实验均在同一独占节点顺序执行，分别 warmup 1 次；全部样本通过 production ZK、native Jolt、streaming proof、完整递归证明及 `max_resident_trace_blocks <= 2` 校验。

| workload | 指标 | baseline | optimized | 改善 |
|---|---|---:|---:|---:|
| Fibonacci-32（3 runs） | BlindFold | 36.820 s | 35.950 s | **2.36%** |
|  | Spartan prove | 24.323 s | 23.553 s | **3.17%** |
|  | 端到端 | 44.716 s | 43.769 s | **2.12%** |
|  | 递归绝对峰值 RSS | 3850.081 MiB | 3775.797 MiB | **1.93%** |
| SHA3-chain-100（2 runs） | BlindFold | 36.627 s | 35.947 s | **1.86%** |
|  | Spartan prove | 24.138 s | 23.351 s | **3.26%** |
|  | 端到端 | 55.154 s | 54.677 s | **0.87%** |
|  | 递归绝对峰值 RSS | 4096.500 MiB | 3958.887 MiB | **3.36%** |

SHA3 的 recursive RSS delta 受测量起始 RSS 波动影响较大，因此内存结论采用绝对峰值 RSS；两个 workload 的 Spartan prove 均稳定下降约 3.2%，与优化目标一致。

## 3. 客观结论

该优化有效但属于增量改进：它降低了 IPA 数据搬运和分配成本，没有改变占主导的椭圆曲线标量乘法复杂度。优化后 BlindFold 仍约 35.9 秒，`spartan-prove` 仍约 23.4–23.6 秒，因此 BlindFold 主线尚未完成。

下一步应增加 IPA round-level profiling，并依次评估：

1. 原地折叠并复用 `a_vec`、`b_vec` 和 generator scratch buffer；
2. 根据 IPA round 长度切换 Rayon/串行执行，降低小轮次调度成本；
3. 为 `w1·L + w2·R` 实现曲线专用双标量乘法或批处理路径；
4. 在 IPA 优化接近平台上限后，再进行 fixed-shape BlindFold 电路改造，以安全复用 Nova/Spartan setup。

完整实验 artifact 位于服务器：

- `jolt/benchmark-runs/stage19-ipa-range-opt-final/`
- `jolt/benchmark-runs/stage19-ipa-range-sha3-final/`
