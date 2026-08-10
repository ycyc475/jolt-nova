# Jolt-Nova BlindFold 优化阶段报告

> 日期：2026-08-10
> 环境：`conda activate jolt-nova-build`，Slurm `comput01` 独占节点，64 CPU cores，Rust 1.95
> 配置：production ZK，Stage 19 runner，warmup 1 次后测量

## 1. 瓶颈复核

分阶段观测与 `perf` 采样显示，BlindFold 的主要冷路径成本为：

- `spartan-prove`：约 24.2 秒（64 核节点），其中主 Bn256 IPA evaluation argument 约 20.9 秒；
- `nova-setup`：约 9.4 秒，其中 BlindFold 主 Pedersen commitment key 需要生成 2,097,152 个 deterministic generators，单次 hash-to-curve 约 4.0 秒；
- `spartan-setup`：约 3.6 秒。

一次 generator batch-normalization 试验仅带来约 0.4% 变化，未保留。直接缓存完整 shape-specific setup 也被正确性检查拒绝：BlindFold 的 public inputs 会烘焙进 R1CS 系数，不同证明的 `shape_id` 不同，因此不能复用完整 PublicParams/Spartan keys。

## 2. 最终优化

最终实现只复用**与 witness/R1CS 系数无关的透明 Pedersen generators**：

1. 按曲线类型、label 和 generator size 建立进程内缓存；较大 key 可为较小请求提供相同确定性前缀；
2. 同一曲线/label 只保留最大的缓存 key，限制缓存增长；
3. `CommitmentKey` 使用 `Arc<Vec<_>>` 共享 generator 存储，避免 cache 与 PublicParams 各持有一份 2,097,152 点的副本；
4. `JOLT_NOVA_PROFILE_SETUP=1` 可输出 key size、缓存命中与 setup 时间；`JOLT_NOVA_PROFILE_SPARTAN=1` 可输出 Spartan 内部分段时间。

该优化不复用 proof-dependent R1CS shape、prover key 或 verifier key，不改变 transcript、证明格式或验证逻辑。

## 3. 最终 A/B 结果

Fibonacci-32、block size 256，baseline 和 optimized 均先 warmup 1 次，再各测量 3 次；6 个测量样本全部通过 production ZK、native Jolt、streaming、完整递归证明及 `max_resident_trace_blocks <= 2` 校验。

| 指标 | baseline 均值 ± σ | optimized 均值 ± σ | 改善 |
|---|---:|---:|---:|
| BlindFold | 41.016 ± 0.145 s | 36.816 ± 0.204 s | **10.24%** |
| 端到端总时间 | 49.055 ± 0.273 s | 44.665 ± 0.359 s | **8.95%** |
| Nova setup | 9.439 ± 0.096 s | 5.327 ± 0.006 s | **43.56%** |
| Spartan prove | 24.165 ± 0.051 s | 24.256 ± 0.227 s | -0.38% |
| 递归绝对峰值 RSS | 4013.453 ± 45.204 MiB | 3844.173 ± 41.100 MiB | **4.22%** |
| 递归 RSS 增量 | 3483.219 ± 11.357 MiB | 3240.132 ± 31.670 MiB | **6.98%** |

SHA3-chain-100（335,084 cycles，block size 16,384）单次泛化复验得到：BlindFold 下降 9.01%，端到端下降 7.06%，Nova setup 下降 43.43%，绝对递归峰值 RSS 下降 6.73%。这与约 4 秒的固定 generator setup 收益一致。

## 4. 客观边界

- 本优化只改善同一进程中首次证明之后的**热路径**；冷启动仍需生成 2,097,152 generators，因此 cold proof 基本没有时间提升。
- 该结果适用于持续运行的 prover 服务、批处理或同一进程连续证明。若部署模式是“一次证明启动一个新进程”，需要进一步实现可信的磁盘预计算 key 加载，才能获得相同收益。
- `spartan-prove` 仍约 24.2 秒且没有改善，依然是下一阶段的首要瓶颈；后续应集中优化 Bn256 IPA evaluation argument，或把 HyperKZG 作为需要 Powers-of-Tau 的可选后端单独评估，不能直接替换当前透明 setup 安全模型。

## 5. 复现

```bash
conda activate jolt-nova-build
cd /public/share/td20062985/dyc/projects/jolt-nova/jolt

# BASELINE_BINARY 和 OPTIMIZED_BINARY 应分别由优化前、后的提交构建并复制得到。
BASELINE_BINARY=target/release/examples/jolt_nova_stage19_benchmark-baseline \
OPTIMIZED_BINARY=target/release/examples/jolt_nova_stage19_benchmark-final-cache \
OUTPUT_DIR=benchmark-runs/stage19-blindfold-opt-rerun \
LOG_DIR=logs/stage19-blindfold-opt-rerun \
bash scripts/run_blindfold_optimization_ab.sh

python3 scripts/compare_blindfold_optimization.py \
  benchmark-runs/stage19-blindfold-opt-rerun/fibonacci-32-baseline-warm-r3.json \
  benchmark-runs/stage19-blindfold-opt-rerun/fibonacci-32-final-cache-warm-r3.json \
  --markdown-output benchmark-runs/stage19-blindfold-opt-rerun/BLINDFOLD_OPTIMIZATION_REPORT.md \
  --json-output benchmark-runs/stage19-blindfold-opt-rerun/blindfold-optimization-comparison.json
```

完整 artifact 位于 `benchmark-runs/stage19-blindfold-opt-final/`。
