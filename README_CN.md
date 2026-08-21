<h1 align="center">Locus</h1>

<p align="center">
  <strong>跨推理引擎的策略感知放置控制面。</strong>
</p>

<p align="center">
  每个请求只规范化一次，Fleet 策略只执行一次。<br/>
  在同一个决策中选择计算目标与可复用状态。
</p>

<p align="center">
  <a href="README.md">English</a>
</p>

<p align="center">
  <a href="https://github.com/imReese/Locus/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/imReese/Locus/actions/workflows/ci.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="Apache 2.0" src="https://img.shields.io/badge/license-Apache--2.0-blue.svg"></a>
  <img alt="Rust 1.85+" src="https://img.shields.io/badge/Rust-1.85%2B-orange.svg">
  <img alt="API 状态：pre-1.0" src="https://img.shields.io/badge/API-pre--1.0-yellow.svg">
</p>

Locus 位于应用 API 与推理引擎之间。它统一规范化请求，执行租户与流量策略，
发现每个执行目标真正具备的能力，评估可复用状态路径，并生成一个可解释的
放置计划。

推理引擎继续拥有 Continuous Batching、加速器内存、Kernel 与模型执行。
NexusKV 等可选状态系统可以提供复用证据和物化选项，但不决定最终执行目标。

## 两分钟跑通

仓库内置的 Fixture 可以跑通完整 HTTP 路径，背后使用确定性 Fake Engine。
不需要 GPU、模型下载、托管 API Key 或外部服务：

~~~bash
git clone https://github.com/imReese/Locus.git
cd Locus
cargo run -p locus-server --example sdk_fixture
~~~

在另一个终端发送 OpenAI Responses 请求：

~~~bash
curl http://127.0.0.1:18080/v1/responses \
  -H 'Authorization: Bearer locus-test-key' \
  -H 'Content-Type: application/json' \
  -d '{"model":"locus-test","input":"respond with JSON"}'
~~~

也可以通过固定版本的 OpenAI 官方 Python SDK 检查 Responses、Chat
Completions、Raw Completions、JSON/SSE、Structured Output、错误格式、
鉴权与取消：

~~~bash
python -m pip install -r scripts/openai-sdk-e2e-requirements.txt
python scripts/openai_sdk_e2e.py --fixture-counts
~~~

这组 Quickstart 能证明本地协议与编排路径，不能证明真实模型已经运行。

## 跟随一次请求

<p align="center">
  <img src="docs/assets/locus-architecture-cn.svg" alt="Locus 架构：应用协议进入控制面，结合可复用状态证据选择执行目标">
</p>

Locus Planner 能使用普通 HTTP 路由层看不到的事实：

- Alias 背后的精确 Tokenizer、Template、Parser 与模型版本；
- 执行目标是否支持请求所需的输出与执行语义；
- 租户额度、优先级、Deadline、取消与 Drain 状态；
- Queue、Prefill、Decode、拓扑和策略成本；
- 兼容可复用状态的可执行范围、位置与物化成本。

规划与副作用保持分离。Planner 返回 <code>PlacementPlan</code>；
<code>PlanExecutor</code> 重新验证计划、预留目标、协调可选状态导入、提交
执行、应用有界降级并完成清理。

## Locus 在系统中的位置

这些组件解决的是不同层次的问题：

| 组件 | 决定什么 | 刻意留给其他层的职责 |
| --- | --- | --- |
| 通用 L7 Proxy | 哪个 Upstream 接收 HTTP 请求 | 模型语义、运行时能力、可复用状态、引擎内执行 |
| 推理引擎 Scheduler | 单个运行时内部如何组批与执行 | Fleet 级租户策略与跨运行时放置 |
| 状态/缓存系统 | 哪些状态可复用、如何物化 | 最终请求放置与运行时本地消费 |
| **Locus** | 哪个合格目标与状态路径服务规范化请求 | Batching、设备分配、Kernel 与物理数据面 |

当路由依赖推理语义，而不只是 Endpoint 健康状态或请求数时，Locus 才有
意义。它不替代以上任何一层。

## 当前可用能力

> [!IMPORTANT]
> Locus 处于 pre-1.0 阶段。仓库包含可部署的控制面切片、确定性 CI 和真实
> 本地 SDK Transport。真实 GPU 执行、原生引擎状态导入与物理状态传输需要
> 独立部署验证，不能由绿色 CI Badge 推导得出。

| 接口面 | 当前可用 | 验证边界 |
| --- | --- | --- |
| Northbound API | OpenAI-compatible Responses、Chat Completions、Raw Text/Token Completions、模型列表、JSON/SSE 与 OpenAI 风格错误 | 进程内测试和官方 SDK 本地 HTTP 测试；属于 API 子集，不是完整参数对齐 |
| 模型语义 | 版本化 Profile、Hugging Face <code>tokenizer.json</code>、有界 MiniJinja Template、Reasoning/Tool Parser、Structured Output 与内容派生身份 | 确定性 Profile 与 Fixture；其他模型 Dialect 需要显式添加 |
| 流量控制 | 凭证绑定租户、加权准入、请求/Token 限制、Deadline、取消、过载拒绝与有界 Drain | 确定性和本地 HTTP 覆盖；真实 Workload 公平性仍属于部署证据 |
| 放置 | 能力过滤、状态查询与成本估算、可解释计划、有界 Replan、Shadow Evaluation 与受控 Calibration Promotion | Planner 和 Mock Telemetry 证据；CI 不能证明 Live Accuracy |
| 引擎边缘 | SGLang 与 vLLM Completion、SSE 和 Telemetry Adapter | Mock HTTP/SSE/Prometheus 一致性；Live Runtime 验证为可选门禁 |
| 状态边缘 | 通用 <code>StateStore</code> 与版本化 NexusKV Bridge | 跨进程协议 CI；物理传输和原生引擎导入仍未验证 |
| 运维 | Health/Readiness、Prometheus Metrics、Request ID、结构化 Trace、持久化有界 Calibration 与优雅退出 | 确定性 Server 与失败路径覆盖 |

## 系统边界

| 责任方 | 职责 |
| --- | --- |
| **Locus** | 协议与模型语义、准入与策略、目标/状态路径规划、计划执行与跨目标可观测性 |
| **推理引擎** | Continuous Batching、引擎内调度、加速器内存、Kernel、并行与模型执行 |
| **状态系统** | 可复用状态查询、兼容证据、位置、传输选项、物理操作与生命周期 |

核心契约不依赖特定 Northbound API、推理引擎、状态系统或 Transport。
运行时专属类型只停留在系统边缘。

设计遵循五个不变量：

1. **稳定内部模型：** 运行时专属 API 只停留在 Adapter 边界；
2. **语义一致：** 兼容目标接收同一个规范化请求，并产生相同应用语义；
3. **能力协商：** Adapter 显式拒绝或降级不支持的需求，不进行猜测；
4. **计算与状态共同规划：** Prefix Match 本身不是放置决策；
5. **Fail-closed Promotion：** 缺少兼容或 Calibration 证据时，不能静默进入
   Active Placement。

## 运行配置化服务

[<code>examples/locus-server.json</code>](examples/locus-server.json) 展示模型
Profile、租户策略、Runtime Discovery、有界 Telemetry、Shadow Placement 与
优雅 Drain。部署前必须将其中的 Artifact 版本、路径、凭证和 Endpoint
Placeholder 替换为真实环境事实：

~~~bash
LOCUS_PREMIUM_API_KEY=replace-me \
LOCUS_BATCH_API_KEY=replace-me \
cargo run -p locus-server -- examples/locus-server.json
~~~

Server 暴露 <code>/healthz</code>、<code>/readyz</code>、
<code>/metrics</code>、<code>/v1/models</code>、
<code>/v1/responses</code>、<code>/v1/chat/completions</code> 与
<code>/v1/completions</code>。新部署应先在 <code>shadow</code> 模式完成
Placement Calibration；进入 <code>active</code> 需要持久化合格证据和精确的
Operator Confirmation。

## 每项检查能证明什么

Locus 按实际观察内容报告测试结果，不会把所有绿色检查都升级为
“Production Ready”：

| 证据等级 | GitHub CI | 能证明什么 |
| --- | --- | --- |
| 静态与确定性 | 是 | 契约、职责、顺序、限制、失败策略与 Mock HTTP/SSE 行为 |
| 官方 SDK + 本地 HTTP | 是 | 真实 Socket 上的 Client 解析与 Transport 兼容 |
| 跨进程状态协议 | 是 | 版本化 lookup/estimate/materialize 兼容，以及 prepare/commit 编排 |
| 真实推理运行时 | 可选 | 一个配置化 Runtime 与模型的实际执行和 Telemetry 变化 |
| 真实多运行时流量 | 可选 | 不同 Runtime 的实测工作量，以及策略、延迟、取消、过载和 Metrics 门禁 |
| 真实状态与硬件 | 部署专属 | 原生导入、物理传输、拓扑与生产性能 |

精确 Harness 与 Acceptance Gate 见
[Serving 验证与证据等级](docs/validation/serving.md)。

## 延伸阅读

| 目标 | 文档 |
| --- | --- |
| 理解职责与系统边界 | [架构](docs/design/architecture.md) |
| 实现引擎集成 | [Canonical Engine Protocol](docs/design/canonical-engine-protocol.md) · [Engine Adapter Contract](docs/design/engine-adapter-contract.md) |
| 添加模型语义或 API Dialect | [Model I/O](docs/design/model-io.md) · [OpenAI-compatible API](docs/design/openai-api.md) |
| 跟踪计算与可复用状态规划 | [State-aware Scheduling](docs/design/state-aware-scheduling.md) · [NexusKV Bridge](docs/design/nexuskv-bridge.md) |
| 配置和运行 Server | [Serving 与配置](docs/operations/serving.md) |

Workspace 按请求路径组织：基础 Crate（core、model-io、parser）、Engine/Store
Port、Planner/Runtime 控制面、边缘 Adapter 与 locus-server 应用。Package
名称和职责见 [Crate Map](crates/README.md)。

## 开发

运行与 GitHub CI 相同的仓库门禁：

~~~bash
bash scripts/ci.sh
~~~

它检查格式、Workspace 严格 Clippy、All-feature Test、Warnings-as-errors
Rustdoc、Python Syntax 与 Traffic Harness 单元测试。Rust API 与公开 Wire
Format 仍处于 pre-1.0。

## 范围

Locus 不是模型执行运行时、引擎内 Scheduler 或通用 Reverse Proxy。它不保证
每个执行目标都能模拟每项 API 功能，也不会把协议一致性解释为真实 GPU、
状态传输、Soak 或容错证据。

## 许可证

Locus 采用 [Apache License 2.0](LICENSE) 许可证。
