# Jolt-Nova BlindFold dense MSM 与 padded shape 复用报告

> 日期：2026-08-11
> 基线：`1ff4b4a`（BN256 GLV/wNAF IPA folding）
> 环境：Slurm `comput03` 独占节点，64 个单线程物理核，Rust 1.95

## 1. 优化内容

本轮继续处理 GLV 优化后的 IPA commitment 与 Spartan 内存开销：

1. IPA `L/R` commitment 的 scalar 为稠密域元素。BN256 新增 dense MSM 路径，直接调用 full-width `msm_best`，跳过通用 MSM 的小标量分类、排序以及 base/scalar 整批复制；其他调用和 Blitzar 后端保持原行为。
2. Spartan setup 已经生成与 verifier-key digest 绑定的 padded R1CS shape。prover key 与 verifier key 在同一进程中通过 `Arc` 共享该 shape，prove 不再重复 padding 和克隆三张稀疏矩阵。
3. padded shape 只是 `serde(skip)` 的进程内 cache，不写入 ProverKey。反序列化 key 会安全回退到原 `S.pad()` 路径，避免 key 文件膨胀并保持落盘工作流正确。

## 2. 正确性门禁

- dense MSM 与原通用 MSM 等价性测试通过；
- DirectSNARK prover key 序列化/反序列化后完成多后端 prove/verify，验证 cache 缺失时的 padding 回退；
- `cargo fmt --all -- --check`、`cargo check --locked --tests`、`cargo clippy --locked --tests -- -D warnings` 通过；
- Nova 全量 `cargo nextest run --locked`：98/98 通过，3 个按项目配置跳过；
- Fibonacci、SHA3 的 production ZK、native、streaming、完整递归验证及 resident blocks 限制全部通过。

## 3. 性能结果

dense MSM 的主 Bn256 IPA profile 中，21 轮 `L/R` commitment 从约 1.742 秒降至 1.575 秒（约 **9.6%**），evaluation argument 从约 5.39 秒降至 5.17 秒。其端到端时间收益较小，但正反实验的绝对峰值 RSS 均下降约 0.7–1.35%。

共享 padded shape 的 Fibonacci-32 独立 r5 A/B：

| 指标 | dense-only | dense + shared shape | 改善 |
|---|---:|---:|---:|
| BlindFold | 21.239 s | 21.067 s | **0.81%** |
| Spartan prove | 8.936 s | 8.591 s | **3.86%** |
| 端到端 | 28.860 s | 28.603 s | **0.89%** |
| 递归绝对峰值 RSS | 3709.672 MiB | 3456.429 MiB | **6.83%** |

反向 r5 中，共享 shape 仍比 dense-only 快约 2.8% Spartan prove，绝对峰值 RSS 低约 7.0%。profile 确认主/次曲线的 `shape-padding` 均从毫秒到百毫秒级降为 0 微秒。

SHA3-chain-100 累计复验（相对 `1ff4b4a` GLV 基线）：

| 指标 | GLV baseline | 最终候选 | 改善 |
|---|---:|---:|---:|
| BlindFold | 21.455 s | 21.191 s | **1.23%** |
| Spartan prove | 8.872 s | 8.609 s | **2.97%** |
| 端到端 | 39.641 s | 39.675 s | -0.08% |
| 递归绝对峰值 RSS | 4030.043 MiB | 3692.221 MiB | **8.38%** |

SHA3 的端到端结果被约 18 秒的 workload/native 阶段稀释，但 BlindFold、Spartan 与内存收益均落在目标路径。

## 4. 边界与后续

- shape 共享只发生在同一次 setup 生成的 pk/vk 之间，不复用不同 `shape_id` 的 key，也不改变证明关系或 transcript。
- 从磁盘反序列化的 ProverKey 不含 shape cache，因此仍会执行一次原 padding；若未来需要持久化 key 后也获得收益，应提供显式的 pk/vk cache 重新关联接口。
- 优化后主 Bn256 evaluation argument 仍约 5.2 秒，generator folding 约 3.5 秒；Nova setup 约 5.3 秒仍是最大固定成本之一。进一步明显提升需要曲线后端/协议级改造或 fixed-shape BlindFold 电路，而不是继续增加小型数据搬运优化。

完整 artifact 位于：

- `jolt/benchmark-runs/stage19-ipa-dense-msm-final/`
- `jolt/benchmark-runs/stage19-ipa-dense-msm-reverse/`
- `jolt/benchmark-runs/stage19-shared-padded-shape-final/`
- `jolt/benchmark-runs/stage19-shared-padded-shape-reverse/`
- `jolt/benchmark-runs/stage19-dense-shape-sha3/`
