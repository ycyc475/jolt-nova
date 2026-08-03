# Jolt–Nova 阶段 16：原生 verifier relation 自动生成

## 1. 阶段目标

阶段 16 消除 clear-Jolt 递归路径中的最后一个手工可信输入：生产调用方不再提交 `VerifierR1CS` 或 assigned witness。完整递归关系必须由一次成功的原生 Jolt 验证自动导出，并通过一个不可拆分的 typed artifact 进入 Nova step circuit。

本阶段覆盖三项任务：

1. 从真实 clear Jolt proof、preprocessing 和 verifier transcript 自动构造完整 endpoint `VerifierR1CS` 与 witness。
2. 建立 Lasso、register、RAM、CPU/R1CS、PCS opening 到 R1CS 变量的可审计映射。
3. 关闭生产 API 中手工注入 verifier witness 的入口，同时保持既有 Nova/Spartan/pinned verification/final PCS acceptance 路径兼容。

## 2. 已完成内容

### 16.1 clear verifier relation capture

- `SumcheckInstanceParams` 的 input/output claim constraint 元数据可在 `nova` clear 路径使用，不再只存在于 `zk` 编译路径。
- 原生 `JoltVerifier` 在八个 regular sumcheck stage 以及两个 univariate-skip stage 中收集真实配置、挑战、claim constraint 和 opening 值。
- clear proof 中省略的线性系数由当前 claim 和其余系数确定性恢复，生成完整 round polynomial witness。
- Stage 8 的 PCS opening 线性组合被加入 verifier relation 的 extra constraint，而不是只保留一个摘要。

### 16.2 自动生成 R1CS 与 assigned witness

- verifier 使用原生 BlindFold layout builder 自动生成十个 stage 的 `VerifierR1CS`。
- `BlindFoldWitness::assign` 根据同一布局生成完整 witness vector。
- artifact 生成前立即执行 `VerifierR1CS::check_satisfaction`；不满足关系时原生验证接口返回错误。
- 该过程只发生在原生 Jolt proof 已经成功验证之后，调用方无法单独提供替代 R1CS 或 witness。

### 16.3 opening → variable 可审计映射

- `VerifierR1CS` 保存 builder 实际分配的 `OpeningId → variable index` 映射。
- alias 被显式解析为 canonical opening，并保留原始 ID、canonical ID、变量位置和 endpoint family。
- 每个 opening 被分类到 `Lasso`、`Register`、`Ram`、`Cpu`，Stage 8 opening 额外标记为 `Pcs`。
- artifact 构造要求五类 endpoint 全部出现；缺少任一类会拒绝生成递归关系。

### 16.4 生产 API 收紧

- 新的生产入口为 `JoltVerifier::verify_with_recursive_relation_artifact`。
- 返回的 `RecursiveJoltVerifierRelationArtifact` 字段私有，构造器仅在 crate 内由成功的 verifier run 调用。
- `RecursiveJoltVerifierCircuit::from_verified_relation` 是唯一公开的生产构造入口，参数中没有 R1CS 或 witness。
- 旧的手工 `new`、`from_verified_artifacts`、`from_blindfold_witness` 只在 `cfg(test)` 下存在，不能进入生产构建。

### 16.5 Nova/Spartan 路径集成

生产数据流现在为：

```text
clear Jolt proof
  → native Jolt verification
  → complete verifier-derived relation artifact
  → RecursiveJoltVerifierCircuit
  → Nova folding
  → Spartan compression / pinned verification
  → deferred PCS final acceptance
```

- regular sumcheck artifacts按其在十阶段 R1CS coefficient grid 中的真实 row index 绑定；两个 univariate-skip row 不再造成错误的顺序对齐。
- 电路构造时再次检查 assigned witness 满足 R1CS，防止 artifact 在内部传递期间被篡改。
- 现有 pinned setup、public statement、Spartan proof 和 final PCS acceptance 接口无需接受私有 witness，保持阶段 15 的外部验证语义。

## 3. 负向测试与安全边界

真实 Fibonacci/Dory e2e 覆盖：

- 从序列化后的真实 Jolt proof 重新运行 verifier 并导出八个 sumcheck artifact。
- 检查五类 endpoint 均有 opening-variable binding。
- 将 verifier witness 任一非常量位置篡改后，生产 circuit constructor 必须拒绝。
- 使用 verifier-derived artifact 合成完整 Nova step circuit，所有约束必须满足。

阶段 16 取消的是“调用方提供 opaque/manual clear verifier witness”的信任，不代表研究系统已经完成所有生产化工作。当前仍保留以下明确边界：

- ZK/BlindFold verifier 的完整递归内部化属于阶段 17；阶段 16 的自动化对象是 clear-Jolt 路径。
- pairing/group-heavy PCS 检查仍通过显式 deferred obligation 在最终 acceptance 执行。
- streaming trace-to-fold、跨块攻击矩阵和系统化性能基线属于阶段 18。
- univariate-skip witness 已进入自动生成的 verifier R1CS；后续阶段仍应继续审计其 transcript/public-statement 绑定是否适合跨 proof 复用 pinned setup。

## 4. 验证矩阵

本地环境固定为 Rust 1.95.x，并使用独立 Stage-16 target 目录。阶段完成前执行：

- `cargo fmt --package jolt-core -- --check`
- `cargo check -p jolt-core --lib --features minimal`
- `cargo check -p jolt-core --lib --features nova`
- `cargo check -p jolt-core --lib --features "nova zk"`
- `recursive_verifier` 定向测试集合
- `stage15_end_to_end_linkage` 回归测试
- `fib_e2e_dory` 真实 proof / relation / circuit 测试
- `cargo test -p jolt-core --lib --features nova --no-fail-fast`

Windows 全量测试若仍仅有既知的 `stdlib_e2e_dory` ZeroOS 工具链平台问题，将记录为平台限制，并由 GitHub Actions Linux runner 验证该项。

实际本地结果：

- `minimal`、`nova`、`nova + zk` 三种特性编译通过。
- `recursive_verifier`：18/18 通过，包括真实 Nova/Spartan、deferred Dory 和负向关系测试。
- `stage15_end_to_end_linkage`：1/1 通过。
- `fib_e2e_dory`：1/1 通过，并执行 Stage-16 自动 relation、opening map、篡改拒绝与完整电路合成。
- `deferred_blindfold_tests`（`nova + zk`）：1/1 通过。
- 全量 `nova`：669/670 通过；唯一失败为 Windows 上既知的 `stdlib_e2e_dory` ZeroOS/std guest 工具链命令缺失。其余真实 guest/Dory e2e 均通过。

## 5. 后续路线

### 阶段 17：ZK/BlindFold verifier 递归内部化

- 把 BlindFold folding、Spartan 和 Pedersen opening 检查拆成 Nova-compatible gadgets。
- 明确区分电路内标量关系和必须 deferred 的群/配对关系。
- 证明 clear 与 ZK 路径对同一 Jolt statement 的接受语义等价。

### 阶段 18：流式生产化、性能与安全收尾

- 完成 streaming trace → block claims → folding，避免持有完整 trace。
- 测量不同 block size 下的 proving time、peak memory、proof size 和 verification time。
- 完成跨块状态、重排、截断、transcript splice、opening substitution、shape/setup substitution 等攻击测试。
