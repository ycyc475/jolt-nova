# Jolt-Nova 阶段 19：真实性能基线与瓶颈分析

## 1. 阶段目标与完成结论

阶段 19 不再增加新的密码学关系，而是在阶段 18 已完成的 production 路径上建立可复现、可审计的性能实验：对同一个真实 guest 执行语句分别生成并验证原生 Jolt ZK proof，以及分块 trace → Nova folding → final Spartan → recursive BlindFold/Dory PCS 的完整证明，并记录时间、内存、证明大小、逐关系耗时和两块驻留上界。

本阶段已经完成以下闭环：

1. 固化版本化 benchmark artifact schema、矩阵摘要与严格校验器；
2. 实现多 workload、多 scale、多 block size、多次测量和 warmup 的 release runner；
3. 对同一 ELF、输入和 statement 比较原生 Jolt 与 Jolt-Nova 递归扩展；
4. 使用 Stage 18 的真实 instrumentation 记录 CPU/R1CS、I/O、register、RAM、Lasso、receipt、Nova 与 Spartan 八类关系；
5. 增加时间、吞吐量、内存与证明大小回归阈值，以及伪造/缺失数据负向测试；
6. 在 Windows/Rust 1.95 release 模式下完成两组真实 workload scale、四个矩阵单元，所有密码学验证门均通过。

因此，阶段 19 的“实验框架、真实基线、瓶颈定位和发布门禁”已经形成完整实现。这里的结论是性能测量结果，不改变阶段 18 的安全模型。

## 2. 实际测量路径

```text
真实 guest ELF + 输入
        │
        ├─ 原生路径：RV64IMACProver → Jolt ZK proof → production verifier
        │                                      │
        │                                      └─ 导出已认证 ZK artifacts/receipt
        │
        └─ 相同 ELF/输入：lazy trace blocks → Stage18 production pipeline
                                           → per-relation checks
                                           → Nova block folding
                                           → final folded Spartan
                                           → recursive BlindFold + group + Dory PCS
                                           → 完整 end-to-end verification
```

每个矩阵单元只有在以下条件全部成立后才允许写入 artifact：

- receipt 来自 production ZK verifier，而不是 synthetic/opaque receipt；
- 原生 Jolt 验证通过；
- streaming final proof 验证通过；
- BlindFold、group obligation 与 Dory PCS 的完整 ZK 验证通过；
- lookup backend 明确为原生 Jolt Lasso subclaim；
- resident trace block 数不超过 2；
- 八类关系都具有真实调用次数和非零 instrumentation 数据；
- proof-size 分项、矩阵单元和 SHA3-256 matrix digest 内部一致。

## 3. 本地 release 基线

环境：Windows x86_64、Rust 1.95、release profile、每个 block size 运行 1 次。数值是阶段验收基线，不应被解释为跨机器的稳定性能承诺。

### 3.1 Fibonacci-32：959 个有效周期

| block size | blocks | resident blocks | 原生 Jolt prove | streaming/Nova | BlindFold prove | 完整总时间 | throughput | 原生峰值增量 | 递归峰值增量 | Jolt proof | recursive payload |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 15 | 2 | 0.829 s | 5.107 s | 126.439 s | 134.585 s | 7.126 cycles/s | 52.594 MiB | 3378.512 MiB | 68,219 B | 22,789 B |
| 256 | 4 | 2 | 0.906 s | 2.411 s | 125.566 s | 130.871 s | 7.328 cycles/s | 72.359 MiB | 3408.613 MiB | 68,219 B | 22,789 B |

- matrix digest：`4ac7271e813d4521cb3b2aa8aaefc5e7d4e268cc66e2a34c33e043cc996acbc6`
- 主瓶颈：`recursive-blindfold-prove`，占测量总时间约 94.93%。

### 3.2 Fibonacci-1024：12,468 个有效周期

| block size | blocks | resident blocks | 原生 Jolt prove | streaming/Nova | BlindFold prove | 完整总时间 | throughput | 原生峰值增量 | 递归峰值增量 | Jolt proof | recursive payload |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 13 | 2 | 1.455 s | 4.997 s | 125.260 s | 133.733 s | 93.231 cycles/s | 71.383 MiB | 3383.051 MiB | 75,195 B | 22,789 B |
| 4,096 | 4 | 2 | 1.400 s | 2.307 s | 122.739 s | 128.422 s | 97.086 cycles/s | 52.109 MiB | 3394.125 MiB | 75,195 B | 22,789 B |

- matrix digest：`5d4e5c17292ee51112db1c42d0a311cf413cc7111b6e50f11e053402f0bcc70a`
- 主瓶颈：`recursive-blindfold-prove`，占测量总时间约 94.60%。
- block size 从 1,024 增至 4,096 后，block 数从 13 降至 4，streaming/Nova 时间约下降 53.8%，但完整总时间只下降约 4.0%，因为固定成本较高的 BlindFold 路径占主导。

## 4. 逐关系实测

下面汇总 Fibonacci-1024 的两个 block-size 单元。它们来自 Stage 18 production instrumentation，不是静态权重估计。

| relation | calls | total | max call |
|---|---:|---:|---:|
| `cpu-r1cs` | 17 | 3.885 ms | 0.713 ms |
| `trace-io-claims` | 17 | 3.408 ms | 0.552 ms |
| `register-read-write` | 17 | 6.632 ms | 1.508 ms |
| `ram-read-write` | 17 | 1.489 ms | 0.258 ms |
| `jolt-lasso-lookup-claim` | 17 | 28.722 ms | 6.159 ms |
| `verified-jolt-receipt-binding` | 17 | 2.068 ms | 0.184 ms |
| `nova-block-fold` | 17 | 3438.707 ms | 1353.869 ms |
| `spartan-final-compression` | 2 | 3613.580 ms | 1879.827 ms |

这张表说明：在分块主路径内部，Nova fold 和最终 Spartan compression 是主要成本；而放到完整递归 ZK 路径中，BlindFold proving 又远大于这两者。Lasso、RAM、register 与 CPU 关系本身不是当前总时间瓶颈。

## 5. 时间、空间与证明大小结论

1. 当前实现没有带来“时间加速”。12,468-cycle 样本的原生 Jolt proving 约 1.4 秒，而完整递归扩展约 127 秒。阶段 19 的价值是确认递归语义和稳定内存边界，并精确指出优化目标。
2. 分块 trace 的空间边界成立。四个单元都只同时驻留至多 2 个 trace block；较大 workload 下估算驻留 trace 从约 196 KiB 增至 772 KiB，随 block size 增长而增长，但不会随总 block 数线性累积。
3. 当前约 3.3 GiB 的递归峰值增量并不是 trace 全量堆积造成的；它几乎不随 block 数变化，说明主要来源是 BlindFold/递归电路及其证明数据。后续内存优化应优先针对该路径。
4. `recursive payload = 22,789 B` 只统计 folded Spartan 与 BlindFold Spartan 两段字节串。typed group obligation 和 Dory PCS 对象已被验证，但没有被悄悄计入该小计，因此不能仅凭 22,789 B 宣称“整个最终 artifact 小于原生 Jolt proof”。
5. `trace` timing 字段仅记录 lazy block iterator 的构造；真正的流式 trace 生成被消费并计入 `nova_spartan_stream`。`preprocessing_micros` 包含 guest build、为 I/O layout 执行的 host trace 和 Jolt preprocessing。

## 6. Artifact、回归门与攻击测试

可提交的本地参考数据位于：

- `jolt/benchmark-baselines/stage19/fibonacci-32.windows-rust-1.95.json`
- `jolt/benchmark-baselines/stage19/fibonacci-32.windows-rust-1.95.blocks.jsonl`
- `jolt/benchmark-baselines/stage19/fibonacci-1024.windows-rust-1.95.json`
- `jolt/benchmark-baselines/stage19/fibonacci-1024.windows-rust-1.95.blocks.jsonl`

runner 支持 `--baseline-input` 和 `--max-regression-percent`。只有平台、workload digest 与 block-size 矩阵一致的 artifact 才可比较；门禁覆盖完整总时间、递归扩展时间、吞吐量、递归峰值内存和 recursive payload 大小。

matrix digest 只吸收原始整数测量、验证状态、statement/receipt 标识和逐关系记录；吞吐量、统计聚合与瓶颈等浮点派生值由原始样本重新计算并校验。这样 JSON round trip 或不同 build profile 不会改变 artifact 身份，同时也不能通过篡改派生报告绕过验证。

负向测试覆盖：未验证 receipt、超过两块驻留、静态估算冒充实测关系、关系缺失、proof-size 错账、矩阵缺项/重复、不可比较 baseline、时间/吞吐回退和 matrix digest 篡改。

## 7. 复现方式

在 `jolt` 目录配置 Jolt guest toolchain 后执行：

```bash
JOLT_STAGE19_REAL=1 \
JOLT_STAGE19_WORKLOAD=fibonacci \
JOLT_STAGE19_SCALE=1024 \
JOLT_STAGE19_BLOCK_SIZES=1024,4096 \
JOLT_STAGE19_OUTPUT=benchmark-runs/stage19/fibonacci-1024.json \
./scripts/run_stage19_validation.sh
```

也可以直接使用 runner 选择 `fibonacci`、`sha3-chain`、`memory-ops` 或 `collatz`，并设置 warmup、重复次数、内存采样间隔和 regression baseline。GitHub Actions 默认运行 schema、runner、Stage 18 回归及真实集成的编译门禁；不会在每次提交中执行数分钟的完整性能测量，以避免 CI 噪声和资源浪费。

## 8. 阶段 20 建议

阶段 20 应优先优化 `recursive-blindfold-prove` 的约束规模、witness 生成和证明内存；其次并行化/批处理 Nova fold 与最终 Spartan compression。完成优化后，应在同平台上使用本阶段 JSON baseline 进行至少 3 次 measurement run，并补充 Linux 与更大 SHA3/RAM workload 的对照，避免把单次 Windows 测量误判为稳定回归。
