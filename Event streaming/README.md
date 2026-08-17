# Event streaming

官方文档：https://docs.langchain.com/oss/python/langchain/event-streaming

Event streaming 是 LangChain 1.3 引入的投影式流式 API。一次 `stream_events(version="v3")` 得到 run 对象，然后分别消费 messages、tool_calls、values。

## 这个组件是什么

底层仍是 LangGraph 的流。差别在于：它把原始协议事件整理成类型化投影，前端不用再解析 `stream_mode` 元组。

LangChain Agent 建在 LangGraph 上，因此继承同一套流，并加上面向 Agent 的 messages / tool calls / state / custom 投影。

## 为什么要设计它

`stream_mode="messages"` 对调试够用，但产品代码会充满 `if chunk["type"] == ...`。投影让「打字机」「工具进度」「最终状态」各走各的迭代器。

---

# 官方知识点讲解

## 1. 入门：stream_events(version="v3")

**官方文档：** 大多数应用/前端应使用 Event Streaming。返回的 run 对象上，每个投影可独立消费。

```python
stream = agent.stream_events({...}, version="v3")
for message in stream.messages:
    for delta in message.content:
        print(delta, end="", flush=True)
final_state = stream.output
```

**对应示例：** `01_stream_events_v3.py`

**限制 / 注意：** 运行时可能看到 BetaWarning：v3 协议目前仍是实验性的。不要把 v2 事件字典循环和 v3 投影混用。

## 2. 可以流什么（投影一览）

**官方文档：**

| 投影 | 用途 |
| --- | --- |
| `for event in stream` | 原始协议事件，完整信封 |
| `stream.messages` | 每次 LLM 调用一条消息流 |
| `message.content` | 文本增量与全文 |
| `message.reasoning` | 推理增量（模型要暴露 reasoning） |
| `message.tool_calls` | 工具参数分片与最终 tool calls |
| `message.output` | 该次模型调用的最终消息 |
| `stream.values` | Agent 状态快照 |
| `stream.output` | 最终 Agent state |
| `stream.subgraphs` | 嵌套图（子 Agent 和普通 subgraph） |
| `stream.subagents` | 仅命名的 `create_agent` 子 Agent |
| `stream.extensions` | 自定义 transformer 投影 |
| `stream.tool_calls` | 工具执行生命周期 |

同步投影既可迭代看增量，也可 drain 取终值：`str(message.content)` 取全文，`message.tool_calls.get()` 取最终工具调用。

**对应示例：** 下表各节。

## 3. Agent messages

**官方文档：** `stream.messages` 给出每次 LLM 调用。`message.node` 标明节点。`message.output` 是最终 AIMessage（含厂商 content blocks）。Python 里 token 用量读 `message.output.usage_metadata`（TS 可用 `message.usage`）。

**对应示例：** `01_stream_events_v3.py`

## 4. Reasoning content

**官方文档：** 形状与 text 相同，仅当模型发出 reasoning 块时可用。循环 `message.reasoning` 再循环 `message.content`。配置见 Models 的 reasoning 指南。

**对应示例：** 无独立文件。需 `enable_thinking=True`。FAQ 指向该能力。

## 5. Tool calls 两种投影

**官方文档：**

- `message.tool_calls`：模型**正在写出**参数
- `stream.tool_calls`：工具**开始执行之后**的生命周期（input、output_deltas、output、error）

**对应示例：** `02_tool_calls_projection.py`

**限制 / 注意（官方 + 实测）：** 同步代码不要先后两个 `for` 分别消费 `messages` 和 `tool_calls`，会漏后一个。用 `stream.interleave("messages", "tool_calls")`。先消费 `output_deltas` 再读 `call.output`。

## 6. Sub-agents / subgraphs

**官方文档：** 内部 `create_agent(..., name="weather_agent")` 的事件在嵌套 namespace。`stream.subagents` 只包含**命名**的 create_agent；每个 handle 有自己的 `.messages` / `.values` / `.tool_calls` / `.output`，以及 `.name`、`.cause`（派发它的那次工具调用）。普通 `StateGraph` 子图走 `stream.subgraphs`，compile 时 `name=` 会出现在 `graph_name`。

**对应示例：** 无独立文件（需要嵌套 Agent）。UI 按需求选 subagents（只要子 Agent）或 subgraphs（所有嵌套图）。

## 7. values 与 output

**官方文档：** `stream.values` 是过程快照；`stream.output` 是最终 state。

**对应示例：** `03_values_and_interleave.py`

## 8. 多投影：async gather vs 同步 interleave

**官方文档：**

- 异步：`astream_events` + `asyncio.gather` 并行消费多个投影
- 同步：`stream.interleave("messages", "tool_calls", "values")`
- 要看未投影的 channel 或完整信封：`for event in stream`

**对应示例：** `03_values_and_interleave.py`

## 9. Custom transformers / middleware

**官方文档：** 应用需要检索进度、artifact 等内置没有的投影时，传 `transformers=[...]`。`langchain>=1.3.2` 起，`AgentMiddleware.transformers` 可注册工厂 `factory(scope) -> StreamTransformer`。编译顺序：

1. 内置 `ToolCallTransformer`
2. 中间件注册的工厂（按 middleware 顺序）
3. `create_agent(..., transformers=)` 调用方传入的

内置 `PIIMiddleware(..., apply_to_output=True)` 用这个钩子在流出前打码，避免只在 `after_model` 改 state 时 live reader 仍看到原文。

**对应示例：** 无独立文件。实现自定义投影见 LangGraph「Build your own projection」。

---

## 知识点目录（本目录代码）

```text
01_stream_events_v3.py          入门 + 天气 Agent
02_tool_calls_projection.py     工具参数与执行
03_values_and_interleave.py     快照、最终输出、交错消费
04_event_vs_stream.py           与 Streaming 对比
```

## 学习顺序

```text
stream_events v3
↓
messages 投影
↓
tool_calls 投影
↓
values / interleave
↓
subagents / transformers（读本节）
↓
决定产品用哪套流式 API
```

## 常见问题

1. 用了 v2 事件字典循环，却按 v3 投影去读：版本要匹配。
2. 同步里先后两个 `for` 会漏事件：改用 `interleave` 或异步 gather。
3. `stream_events(version="v3")` 目前仍是实验协议，可能看到 BetaWarning。
4. 需要推理 token：模型得开启 thinking，再读 `message.reasoning`。

## Event streaming vs Streaming

| | Streaming `stream_mode` | Event streaming v3 |
| --- | --- | --- |
| 是什么 | 按模式拆 chunk | 类型化投影 |
| 适合 | 调试节点、兼容旧代码 | 聊天 UI、多投影并行 |
| 何时选 A | 要精确控制 Pregel 模式 | |
| 何时选 B | 新前端、工具进度条 | |

不要在同一条请求里混用两套并期待顺序一致。

## 与其他组件的关系

```text
Agent + Tools 产生事件
    ↓
Event streaming 投影给 UI
    ↓
Short-term memory 的 thread 仍然通过 config 传入
```
