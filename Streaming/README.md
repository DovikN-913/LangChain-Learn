# Streaming

官方文档：https://docs.langchain.com/oss/python/langchain/streaming

Streaming 让应用在模型还没说完时就把中间结果推给用户。本目录对应底层 Pregel `stream` / `astream`。

> **官方当前建议：** 新应用优先用 [Event streaming](https://docs.langchain.com/oss/python/langchain/event-streaming) 的 `stream_events(..., version="v3")`。本目录保留 `stream_mode` API，因为它仍是理解图执行的基础。

## 这个组件是什么

`create_agent` 返回编译后的图。调用 `graph.stream(..., stream_mode=...)` 可以按模式订阅执行过程。

## 为什么要设计它

LLM 延迟高。等整段结束再显示，产品会像卡住。流式把「等待」变成「正在发生的事」。官方列出的能力包括：Agent 进度、LLM token、thinking/reasoning、自定义进度、多模式同时订阅。

---

# 官方知识点讲解

## 1. 支持的 stream_mode

**官方文档：** 把一个或多个模式作为 list 传给 `stream` / `astream`：

| 模式 | 内容 |
| --- | --- |
| `updates` | 每个 Agent 步骤后的状态增量。同一步里多个节点会分开推 |
| `messages` | 任意调用了 LLM 的节点产出的 `(token, metadata)` |
| `custom` | 节点/工具里用 stream writer 推的任意数据 |

**对应示例：** `04_stream_modes_compare.py`

## 2. Agent progress（updates）

**官方文档：** `stream_mode="updates"` 每步推一次。一次工具调用通常看到：model（带 tool_calls 的 AIMessage）→ tools（ToolMessage）→ model（最终文本）。`thread_id` 与 `stream_mode` 独立；要跨轮续聊仍需 checkpointer。

**对应示例：** `01_agent_stream_updates.py`

**限制 / 注意：** 本地要持久化必须自己传 checkpointer。LangSmith 部署会自动配。

## 3. LLM tokens（messages）

**官方文档：** `stream_mode="messages"` 看 token 和 tool_call_chunk。chunk 带 `langgraph_node` 等 metadata。把 Agent 当父图节点时，父图默认收不到子图 token，需 `subgraphs=True`。

**对应示例：** `02_agent_stream_messages.py`

**限制 / 注意：** 用了 `updates` 却期望打字机效果，是选错模式。建议 `version="v2"`，否则 chunk 可能是 `(mode, data)` 元组。

## 4. Custom updates

**官方文档：** 工具里 `writer = get_stream_writer(); writer("查到一半了")`，再以 `stream_mode="custom"` 订阅。

**对应示例：** `03_custom_updates.py`（真实天气查询进度）

**限制 / 注意（官方）：** 工具一旦用了 `get_stream_writer()`，就不能在 LangGraph 执行上下文之外单独 invoke。

## 5. 同时订阅多种模式

**官方文档：** `stream_mode=["updates", "custom"]`。v2 下每个 chunk 是 `{"type", "ns", "data"}`，用 `chunk["type"]` 区分。

**对应示例：** `03_custom_updates.py`

## 6. Streaming thinking / reasoning

**官方文档：** 模型若产出 reasoning 块，用 `stream_mode="messages"` 过滤 `content_blocks` 里 `type=="reasoning"`。Event streaming 则读 `message.reasoning`。必须在模型侧打开 reasoning。

**对应示例：** 无独立文件。模型层见 `Models/06_model_parameters.py`（`enable_thinking=True`）。Agent 投影见 Event streaming README。

## 7. Streaming tool calls

**官方文档：** `messages` 模式看参数 JSON 分片；完整解析后的 tool_calls / ToolMessage 在 state 里，可同时订 `["messages", "updates"]`。若完成消息不进 state，就自己累加 chunk，或用 custom writer。

**对应示例：** `02_agent_stream_messages.py`；更完整的工具生命周期见 `Event streaming/02_tool_calls_projection.py`

## 8. Human-in-the-loop

**官方文档：** 配 `HumanInTheLoopMiddleware(interrupt_on={...})` + checkpointer。在 `updates` 里收集 `__interrupt__`，再用 `Command` 按顺序回传 approve/edit 决策。

**对应示例：** 无独立文件。这是独立的 HITL 专题，官方放在 Streaming common patterns 下。本学习单元不展开人工审批。

## 9. Streaming from sub-agents

**官方文档：** 父图默认不冒泡子 Agent 的 token。需要 `subgraphs=True`，或改用 Event streaming 的 `stream.subagents` / `stream.subgraphs`。

**对应示例：** 无独立文件。Event streaming README 有官方投影说明。

## 10. Disable streaming

**官方文档：** 多 Agent 时只想让部分模型流式、或某些模型不支持流、或不想把输出推到 LangSmith 客户端时，初始化设 `streaming=False`。不是所有集成都认这个参数，通用后备是 `disable_streaming=True`（基类上有）。

**对应示例：** 无独立文件。本项目默认需要流式教学，不关闭。

## 11. v2 streaming format

**官方文档：** 需 LangGraph >= 1.1。`version="v2"` 时无论订几种 mode，chunk 都是 `StreamPart`：`type` / `ns` / `data`，不必解包元组。

对比：

```text
v2: for chunk in stream: chunk["type"], chunk["data"]
v1: for mode, chunk in stream:  # 默认仍可能是元组
```

v2 的 `invoke()` 还会返回带 `.value` 和 `.interrupts` 的 `GraphOutput`。

**对应示例：** 本目录示例均传 `version="v2"`。

**限制 / 注意：** 不传 version 时若按 v2 去读 `chunk["type"]` 会对不上。

---

## 知识点目录（本目录代码）

```text
01_agent_stream_updates.py     按节点看进度
02_agent_stream_messages.py    流式 token
03_custom_updates.py           天气查询进度
04_stream_modes_compare.py     三种模式对比
```

## 学习顺序

```text
updates（看步骤）
↓
messages（看 token）
↓
custom（看业务进度）
↓
v2 信封 / 关闭流式 / HITL（读本节）
↓
转到 Event streaming
```

## 常见问题

1. 没有 token：用了 `updates` 而不是 `messages`。
2. 工具里 writer 报错：工具在图外被单独 invoke 了。
3. 子图没有 token：需要 `subgraphs=True`。
4. chunk 结构对不上：补上 `version="v2"`。

## 与 Event streaming 的关系

```text
Streaming：自己按 stream_mode 拆 chunk
Event streaming：stream.messages / stream.tool_calls 投影
```

新前端两者选 Event streaming；排错图执行选本目录。
