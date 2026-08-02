# Jolt–Nova 阶段 15：真实验证器集成与收尾报告

## 1. 阶段目标

阶段 15 的目标不是再增加一种摘要 receipt，而是把阶段 14 已有的递归验证关系接到真实 Jolt 验证流程，并形成可复用、可验证、可组合的最终证明接口。其安全边界是：摘要只用于标识和连续性绑定，最终接受必须执行 Nova/Spartan、Jolt verifier relation 与真实 PCS/BlindFold 验证。

## 2. 已完成内容

### 15.1 真实 clear Jolt verifier 适配

- `BatchedSumcheck::verify` 返回验证器实际观察到的 transcript checkpoint、初始 claim、最终 claim 和 degree bound。
- `JoltVerifier::verify_with_recursive_sumcheck_artifacts` 先执行原生生产验证器，再导出八个 clear-sumcheck 的无损 artifact。
- artifact 适配器重新运行原生 `ClearSumcheckProof::verify`，不接受主机传入的布尔 `verified` 标志。
- 每个 sumcheck stage 保存独立的 transcript checkpoint，允许 Jolt 在相邻 sumcheck 之间插入 lookup、RAM、register、CPU 等 transcript 事件。

### 15.2 递归电路的 transcript 与关系绑定

- Nova step circuit 从各 stage 的真实 checkpoint 独立重放 Poseidon Fiat–Shamir 过程。
- 所有 checkpoint 通过 Poseidon capsule root 绑定为公开 statement；修改任一 checkpoint 都会改变公开输入。
- sumcheck 恢复出的完整多项式系数逐项绑定到原生 verifier R1CS/Hyrax witness coefficient grid。
- verifier R1CS 的矩阵、维度、Hyrax 布局和公开关系都进入 circuit shape hash。

### 15.3 固定参数的 Nova/Spartan 接口

- 增加固定 circuit shape 的 setup，生成可复用 prover parameters 和 verification key。
- 证明阶段不再为每个 proof 临时生成参数。
- 验证阶段只需要公开 statement、压缩证明和固定 verification key，不需要重新传入私有 witness circuit。
- 验证器检查 object ID、deferred PCS obligation ID、初始 transcript checkpoint、checkpoint capsule root 和 shape ID。

### 15.4 多块最终证明组合

- `JoltNovaEndToEndProof` 同时封装分块执行 folding proof 与递归 Jolt verifier proof。
- linkage digest 绑定最终 folded instance、folded proof、Jolt object、PCS obligation、transcript checkpoints 和 verifier shape。
- 最终验证必须依次通过 linkage、folded Nova/Spartan、递归 verifier Nova/Spartan 和真实 deferred PCS 检查。
- 增加证明大小、公开输出、块数、周期数和 linkage 的基线数据结构。

### 15.5 BlindFold/ZK 等价接受路径

- ZK 路径保存真实 `BlindFoldProof`、verifier input、Pedersen generators、完整 R1CS、evaluation generators 和 transcript。
- obligation ID 对上述密码学对象作一致性绑定，但不代替验证。
- 最终接受直接调用生产 `BlindFoldVerifier` 与真实 deferred PCS verifier；篡改 Spartan/BlindFold proof 会被拒绝。

### 15.6 安全与回归测试

- 正向：统一 verifier relation 可生成并验证真实压缩 Nova/Spartan proof。
- 负向：修改公开输出、object ID、shape、sumcheck coefficient grid、deferred PCS 或 BlindFold proof 均被拒绝。
- 多阶段：两个拥有不同 Poseidon checkpoint 的 sumcheck stage 可独立重放，并共同绑定到 checkpoint capsule root。
- 组合：执行 folding proof 与 verifier proof 的不可交换 linkage 得到验证。

## 3. 当前真实安全边界

阶段 15 已取消“opaque receipt digest 等于真实性”的接受方式。clear 路径的 sumcheck proof、checkpoint 和 endpoint 由生产 Jolt verifier 自动提取；Nova 电路验证 transcript、sumcheck arithmetic、coefficient-grid binding 和传入的原生 verifier R1CS。

仍需明确两个边界：

1. clear 路径目前仍由调用者以类型化数据提供完整 `VerifierR1CS` 和 assigned witness；尚未从普通 clear Jolt proof 自动重建 BlindFold 风格的 endpoint R1CS/witness。
2. ZK/BlindFold 路径采用“最终一次真实 deferred verification”，尚未把整个 BlindFold verifier 编译进 Nova step circuit。

因此，阶段 15 完成的是可验证的真实集成里程碑，不代表研究项目已经达到“所有原生 Jolt 验证逻辑完全在 Nova 内部递归执行”的最终形态。摘要仍可作为承诺和索引，但不是真实性依据。

## 4. 后续阶段建议

### 阶段 16：完全自动化原生关系生成

- 从 clear Jolt proof/preprocessing 自动生成完整 endpoint `VerifierR1CS` 与 assigned witness。
- 对 Lasso、register、RAM、CPU/R1CS 的每个 claim 建立 proof 字段到 R1CS 变量的可审计映射。
- 去除生产 API 中手工提供 verifier witness 的入口。

### 阶段 17：ZK verifier 的递归内部化

- 将 BlindFold verifier 的 folding、Spartan 与 Pedersen opening 检查拆成 Nova 兼容 gadget。
- 仅将无法在标量电路中经济验证的群/配对等式保留为明确的 deferred obligation。
- 证明 clear 与 ZK 两条路径对同一 Jolt statement 具有等价接受语义。

### 阶段 18：生产化与实验

- 完成 streaming trace → block → claims → folding，避免持有完整 trace。
- 进行不同 block size 下的时间、峰值内存、证明大小和 verifier 时间基准。
- 完成跨块状态、重排、截断、transcript splice、opening substitution 和参数替换攻击测试。

## 5. 验证记录

本阶段本地验证结果（Rust 1.95.0，Windows）：

- `cargo check -p jolt-core --lib --features nova --offline`：通过。
- `cargo check -p jolt-core --lib --features "nova zk" --offline`：通过。
- `recursive_verifier` 定向集合：18/18 通过，包括 pinned Spartan、真实 Dory、clear adapter、多 checkpoint 与负向测试。
- `stage15_end_to_end_linkage`：1/1 通过。
- `deferred_blindfold_tests`（`nova + zk`）：1/1 通过，真实 BlindFold 正向与篡改负向均成功。
- `cargo test -p jolt-core --lib --features nova --offline --no-fail-fast`：安装本地 Jolt CLI 后，669/670 个测试可在 Windows 验证通过。
- 唯一未通过项为 `stdlib_e2e_dory`：Jolt 的 ZeroOS/std guest 构建脚本在 Windows 报 `Toolchain setup failed: program not found`；普通 guest 的 Fibonacci、RAM、advice、SHA2/SHA3、BTreeMap、MulDiv 与真实 Dory e2e 均通过。
- 提交后还需由 GitHub Actions 的 Linux runner 验证全部工作流，并覆盖上述 std/ZeroOS 平台项。

构建过程中曾出现 Rust 1.96.1/1.95.0 混用的缓存错误；最终验证显式固定 `cargo`、`rustc`、`rustdoc` 为 1.95.0，并使用独立 target 目录，确认该错误与源码无关。
