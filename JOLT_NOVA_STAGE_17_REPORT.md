# Jolt–Nova 阶段 17：ZK/BlindFold verifier 递归内部化

## 1. 阶段目标

阶段 17 将阶段 15 中“最终重新调用完整主机 BlindFold verifier”的兼容路径升级为可递归验证关系。真实性不再由 BlindFold receipt digest 或主机布尔值表示：Nova 电路验证标量关系，无法经济地放进标量电路的承诺群等式由显式 typed obligation 在最终验收时执行。

## 2. 生产数据流

```text
真实 ZK Jolt proof
  → 原生 JoltVerifier 完整验证
  → verifier 内部捕获 BlindFold proof/input/R1CS/generators/transcript
  → RecursiveBlindFoldRelationArtifact（调用者不能注入 witness）
       ├─ RecursiveBlindFoldVerifierCircuit
       │    → Nova folding → Spartan compression → pinned verification
       └─ RecursiveBlindFoldGroupObligation
            → Pedersen/Hyrax 群等式 + 独立 transcript 重放
  → deferred Jolt PCS opening
  → RecursiveJoltZkCompleteFinalAcceptance
```

## 3. 已完成内容

### 17.1 verifier-derived ZK artifact

- `JoltVerifier::verify_with_recursive_zk_relation_artifact` 只在完整原生 Jolt 验证成功后返回 artifact。
- artifact 保留真实 `BlindFoldProof`、`BlindFoldVerifierInput`、`VerifierR1CS`、Pedersen generators、evaluation generators 和 BlindFold 前置 Poseidon checkpoint。
- 生产构造器为 crate-private；调用者不能提交替代 R1CS、挑战、witness 或 `accepted` 标志。

### 17.2 完整 Poseidon transcript 重放

- verifier adapter 记录 BlindFold 实际执行的每次 raw append 和 challenge 操作。
- Nova gadget 精确实现 Jolt Poseidon 的 `(state, n_rounds, data)` 变换；多块 byte append 的后续块使用 round `0`，整个 append 只增加一次 round counter。
- folding challenge、`tau`、outer sumcheck challenges、`ra/rb/rc` 和 inner sumcheck challenges 全部由电路重放得出，不能作为自由 witness 注入。
- 初始/最终 transcript state 和 round 均进入公开 statement。

### 17.3 BlindFold folding 与 relaxed R1CS

- 电路约束 `u_folded = 1 + r · u_random`。
- 约束外层终点：

  `eq(tau, rx) · (Az(rx)·Bz(rx) − u·Cz(rx) − E(rx))`。

- `E(rx)` 由真实 Hyrax error opening row 与 `rx` 的列坐标在电路内求值。

### 17.4 两层 Spartan sumcheck

- 外层和内层 compressed univariate polynomial 均恢复缺失的一次项。
- 每轮约束 `g(0)+g(1)=claim`，并使用 transcript 派生挑战执行 Horner 求值。
- 内层初始 claim 由 R1CS 的公开列投影和 `ra/rb/rc` 重新计算。
- 内层终点在电路中从稀疏 A/B/C 矩阵计算 `L_W(ry)`，并约束 `claim_final=L_W(ry)·W(ry)`。

### 17.5 evaluation binding 与 typed 群 obligation

- final PCS evaluation output/blinding opening rows的稀疏位置、零填充和选中值在电路内约束。
- Pedersen evaluation commitment、folded witness row、E opening 和 W opening 的真实群等式由 `RecursiveBlindFoldGroupObligation::verify` 执行。
- 群 obligation 不接受摘要或布尔收据；它持有完整曲线对象、挑战和生成元。
- 群 obligation 独立重放记录的 Poseidon 操作，并检查最终 checkpoint；不再以重新调用完整主机 scalar verifier 代替递归关系。

### 17.6 固定参数 Nova/Spartan 与最终验收

- 新增固定 circuit shape 的 prover parameters 和 verification key。
- verifier 只接收公开 `RecursiveBlindFoldStatement`、固定 key 和压缩 proof，不接收 witness circuit。
- 最终验收依次检查 object ID、共同 Jolt statement ID、group obligation ID、最终 transcript checkpoint、Nova/Spartan proof 和真实群等式。
- `RecursiveJoltZkCompleteFinalAcceptance` 进一步组合真实 deferred PCS opening，PCS ID 必须与外层 trace/folding linkage 一致。

### 17.7 clear/ZK statement 等价

- `VerifiedJoltLookupProofReceipt::recursive_execution_statement_id` 只绑定 preprocessing、verifier setup、public I/O、trace length 和 trusted advice identity。
- proof mode、ZK randomness 和 proof commitments 不进入该 ID，因此 clear 与 ZK proof 可声明同一个执行 statement；任何执行输入变化都会改变 ID。

## 4. 安全边界

以下值只是承诺/连续性标识，不能单独导致接受：

- `object_id`
- `deferred_group_id`
- `jolt_statement_id`
- deferred PCS ID
- 旧版 receipt digest

最终接受必须同时拥有并验证：

1. pinned Nova/Spartan recursive scalar proof；
2. typed BlindFold commitment-group obligation；
3. typed Jolt PCS obligation；
4. 外层 trace-to-fold linkage 中相同的执行 statement 和 obligation IDs。

## 5. 测试覆盖

- synthetic BlindFold proof 的完整 scalar circuit satisfaction；
- 真实 Nova recursive proof、Spartan compression 和 pinned verification；
- 真实 Pedersen/Hyrax group obligation；
- 真实 Fibonacci ZK Jolt proof → production verifier → Stage-17 artifact → circuit synthesis → group verification；
- clear/ZK execution statement ID 等价；
- 篡改 `Az`、outer sumcheck coefficient、transcript append、W opening、object ID、statement ID 和 final transcript checkpoint 均被拒绝。

## 6. 验证结果

- `cargo check --locked -p jolt-core --lib --no-default-features --features "nova zk"`：通过。
- minimal block 路径：78/78 通过。
- Nova block 路径：207/208 通过；唯一失败是旧 Dory 负向测试未区分透明/ZK 模式。修正后两种模式的定向测试均通过，最终结果由 Linux CI 全量复核。
- Stage-17 no-default recursive verifier：2/2 通过。
- clear/ZK execution statement 等价：1/1 通过。
- 默认特性下 Stage-17：标量关系、篡改拒绝和真实 Nova/Spartan pinned proof 共 3/3 通过。
- 真实 Fibonacci ZK Jolt proof 经 production verifier 生成 Stage-17 artifact，电路综合及 typed group obligation 验证通过。
- Windows 全量 `nova zk`：650 项通过；其余 guest 失败均由原项目 `/tmp` guest target 在 Windows 映射为不可写路径导致，不是密码学关系失败。Linux Actions 作为最终跨平台基准。
- `cargo fmt --check -p jolt-core`：通过；本地 `tracer` 检查受工作树既有 CRLF 换行影响，未修改或提交这些文件，Linux Actions 将验证仓库 LF 内容。

## 7. 阶段 18 边界

阶段 17 完成的是 ZK verifier 的密码学递归内部化。以下内容明确留给阶段 18：

- streaming trace → block claims → folding，避免完整 trace 常驻内存；
- 按 block size 测量 proving time、peak memory、constraints、proof size 和 verifier time；
- 跨块重排/截断、state splice、opening substitution、shape/setup substitution 的完整攻击矩阵；
- 发布级参数治理、跨平台复现脚本和最终安全/性能报告。
