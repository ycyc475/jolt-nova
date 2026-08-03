# Jolt–Nova 阶段 18：流式生产化、性能与安全收尾

## 1. 阶段目标

阶段 18 将阶段 14–17 已有的 Jolt、Lasso、CPU/R1CS、RAM/register、BlindFold、PCS、Nova 与 Spartan 关系组合为一个生产级数据流：

```text
lazy trace iterator
  → 当前 block + 单 block 前瞻
  → CPU / register / RAM / Jolt-Lasso claims
  → verified Jolt receipt 精确绑定
  → Nova absorb（立即丢弃 block witness）
  → final Spartan compressed proof
  → BlindFold recursive proof + group obligation + deferred Jolt PCS
  → Stage18ZkEndToEndProof::verify
```

摘要、object ID、receipt digest 和 obligation ID 只承担不可变绑定作用，不能单独导致接受。最终接受必须实际执行两份 Spartan 验证、BlindFold 群等式和 Jolt/Dory PCS opening。

## 2. 有界内存 streaming

`BlockProofPipeline::prove_streaming_blocks_with_final_proof` 直接消费 `IntoIterator<Item = TraceBlock>`。实现只保存当前块和下一块，用下一块首 cycle 认证 CPU lookahead；每个 bundle 和 fold input 在 `NovaFoldingBackend::absorb` 成功后立即释放。

- 生产上限固定为两个驻留 trace blocks。
- 不再构造 `Vec<BlockProofBundle>` 或 `Vec<BlockFoldInput>`。
- 跨块 RAM 连续性仅保存每个已访问地址的最新值；因此 trace witness 空间有界，RAM 状态空间仍与程序实际触及地址数相关。
- 最后一块必须具有 terminal machine state；终止后继续执行、截断和空 trace 均拒绝。

## 3. 发布参数

`Stage18ReleaseParameters` 固定以下安全边界并生成统一 digest：

- 非零、2 的幂 block target size；
- 最多两个驻留 trace blocks；
- 128-bit security policy；
- `jolt-nova-block-fold-v3` Nova relation；
- 原生 `jolt-lasso-subclaim-v1`，不接受实验 LogUp backend；
- 真实 `spartan-final-proof`，不接受 placeholder；
- ZK mode 必须开启；
- 非零且与本次 verifier R1CS 完全匹配的 BlindFold circuit shape ID。

不同程序规模可能产生不同 verifier shape，因此 shape ID 是每个发布参数集的一部分，不是一个跨所有程序硬编码的常量。

## 4. 同一次 Jolt verifier invocation 的完整绑定

`JoltVerifier::verify_with_recursive_zk_complete_artifacts` 只在生产 Jolt ZK verifier 完整成功后导出：

1. `RecursiveBlindFoldRelationArtifact`；
2. 真实 deferred PCS opening；
3. 同一次调用生成的完整 `VerifiedJoltLookupProofReceipt`。

三个对象的生产构造器均为 crate-private。`Stage18ZkEndToEndProof` 要求 streaming proof 的 receipt digest、共同 Jolt statement ID、BlindFold object/group IDs、PCS ID、shape ID、transcript checkpoints 和最终 folded proof 全部进入同一个 linkage digest，并在构造后立即执行一次完整自验证。

## 5. 性能与内存输出

累计 profile 覆盖：

- CPU/R1CS；
- trace IO claim extraction；
- register read-write；
- RAM read-write 与跨块连续性；
- Jolt Lasso lookup claim；
- verified receipt binding；
- Nova block fold；
- final Spartan compression。

`Stage18BlockProfile` 通过回调逐块输出 JSONL，输出后立即释放；`Stage18StreamingMetrics` 提供总耗时、最大耗时、block/cycle 数、最多驻留块与 cycle、RAM 地址峰值、采样物理内存、估算 trace bytes 和最终 proof bytes。发布参数和最终 streaming proof 均可输出 pretty JSON。

Rust 1.95 Windows 本地两块 NoOp 基线的一次样本（仅用于确认采集链路，不作为跨机器性能结论）：最多驻留 2 blocks/4 cycles，估算 peak trace storage 4,736 bytes，最终 Spartan payload 10,664 bytes，Nova 两步合计约 1.86 s，最终 Spartan compression 约 2.16 s。完整原始 JSON 会由 `--nocapture` 验证命令输出。

## 6. 攻击矩阵

阶段 18 的正向与负向测试覆盖：

- block 重排、index/cycle gap、截断、终止后继续执行；
- machine/register boundary state splice；
- RAM 跨块 first/final value splice；
- 错误 block target size 和 release manifest；
- 不同 verified Jolt receipt；
- final folded Spartan proof 篡改；
- BlindFold transcript checkpoint 篡改；
- recursive verifier shape/setup substitution；
- deferred PCS ID/opening substitution；
- recursive Spartan proof bytes 和最终 linkage digest 篡改。

已有阶段 14–17 测试继续覆盖 sumcheck coefficient、Fiat–Shamir challenge、R1CS row、Pedersen/Hyrax opening 和 Dory pairing 的底层篡改。

## 7. 验证入口

Linux/macOS 的确定性入口：

```bash
bash jolt/scripts/run_stage18_validation.sh
```

默认执行格式、minimal/nova/zk 编译、Stage-18 streaming/攻击测试、Stage-17 recursive verifier 回归和真实 Jolt ZK 集成编译。配置好 Jolt guest toolchain 后执行真实 Fibonacci 完整闭环：

```bash
JOLT_STAGE18_REAL_ELF=1 bash jolt/scripts/run_stage18_validation.sh
```

本地 Windows Rust 1.95 验证已经完成：两块真实 Nova/Spartan streaming 测试通过；真实 Fibonacci Jolt ZK → Stage-18 end-to-end 测试及全部最终对象篡改测试通过。GitHub Actions 是最终 Linux 跨平台发布门禁。

## 8. 安全边界与后续研究

阶段 18 的最终 verifier 仍显式执行 typed BlindFold commitment-group obligation 与 typed Dory PCS obligation。这不是摘要信任，而是真实密码学验证。如果后续目标被加强为“验证者只验证一个 Spartan proof，不执行任何额外群运算”，则需要新的阶段把这些群等式以非原生域 gadget 或适合递归的 PCS 完全电路化；该目标不属于当前阶段 18 的生产验收定义。
