# Jolt-Nova BlindFold IPA GLV/wNAF 优化报告

> 日期：2026-08-11
> 基线：`9a4177a`（IPA joint two-bit window）
> 环境：Slurm `comput03` 独占节点，64 个单线程物理核，Rust 1.95

## 1. 优化内容

本轮继续优化 BlindFold 的主 Bn256 IPA generator folding。BN256 已具有高效曲线 endomorphism，因此将每个公开 Fiat–Shamir 挑战分解成两个约 128-bit 的有符号标量，再使用四路 4-bit wNAF 同时计算两点线性组合。标量分解和 wNAF schedule 在每轮只计算一次，由全部 generator pair 共享。

与原 254-bit joint-window 路径相比，该实现把 doubling 链缩短约一半，并降低非零加法密度。该路径只处理公开的 transcript challenge 和公开 generator，允许 variable-time；秘密 witness scalar、证明关系、transcript 和证明格式均未改变。Blitzar feature 继续使用原后端。

## 2. 正确性与门禁

- BN256 GLV/wNAF 与原 joint-window 实现进行零、单位、负单位及 64 组随机挑战的多点等价性测试；
- `cargo fmt --all -- --check`、`cargo check --locked --tests`、`cargo clippy --locked --tests -- -D warnings` 通过；
- Nova 全量 `cargo nextest run --locked` 通过；
- Fibonacci 与 SHA3 的全部样本均通过 production ZK、native Jolt、streaming proof、完整递归验证及 `max_resident_trace_blocks <= 2` 校验。

## 3. A/B 结果

双方均在同一独占节点 warmup 1 次后测量。Fibonacci 另做反向顺序复验，结果方向一致。

| workload | 指标 | baseline | GLV/wNAF | 改善 |
|---|---|---:|---:|---:|
| Fibonacci-32（3 runs） | BlindFold | 23.489 s | 21.516 s | **8.40%** |
|  | Spartan prove | 11.021 s | 8.990 s | **18.43%** |
|  | 端到端 | 31.052 s | 28.910 s | **6.90%** |
|  | 递归绝对峰值 RSS | 3774.023 MiB | 3783.393 MiB | -0.25% |
| SHA3-chain-100（2 runs） | BlindFold | 23.470 s | 21.811 s | **7.07%** |
|  | Spartan prove | 10.999 s | 9.194 s | **16.41%** |
|  | 端到端 | 41.899 s | 40.071 s | **4.36%** |
|  | 递归绝对峰值 RSS | 4058.883 MiB | 3951.215 MiB | **2.65%** |

反向 Fibonacci A/B 中，GLV/wNAF 仍比原实现快 8.61% BlindFold、22.87% Spartan prove，排除了执行顺序造成的假收益。主 Bn256 IPA profile 中，21 轮 generator folding 从 5.539 秒降至 3.575 秒，改善 **35.45%**；evaluation argument 约从 7.37 秒降至 5.47 秒。

## 4. 客观边界与后续

该优化显著降低了曲线计算，但优化后 BlindFold 仍约 21.5–21.8 秒，其中 Nova setup 约 5.4 秒，Spartan prove 约 9 秒，固定 setup 和约 1.7 秒的 IPA `L/R` commitment 仍是后续重点。共享窗口预计算、分块 batch normalization 和两阶段 batch normalization 已经实测无稳定收益并撤销。

完整实验 artifact 位于服务器：

- `jolt/benchmark-runs/stage19-ipa-bn-glv-final/`
- `jolt/benchmark-runs/stage19-ipa-bn-glv-reverse/`
- `jolt/benchmark-runs/stage19-ipa-bn-glv-sha3/`
- `jolt/benchmark-runs/stage19-ipa-bn-glv-smoke/`
