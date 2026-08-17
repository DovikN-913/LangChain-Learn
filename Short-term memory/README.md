# Short-term memory

官方文档：https://docs.langchain.com/oss/python/langchain/short-term-memory

短期记忆让 Agent 在同一条 **thread** 里记住刚才发生的事。它不是跨用户、跨会话的长期画像。

## 这个组件是什么

LangChain Agent 把对话历史放在图状态的 `messages` 里。加上 **checkpointer** 后，状态按 `thread_id` 存下来，下次 invoke 还能读到。

官方把 thread 比作「一封邮件会话」：同一会话里的多轮交互编在一起。

## 为什么要设计它

Chat Model 本身无状态。没有记忆层，每次请求都是失忆的。短期记忆解决单次会话连续性；同时用裁剪/摘要控制上下文膨胀——即使模型号称能塞下全部历史，过长上下文也会让模型分心、更慢、更贵。

跨会话用户偏好应走 [long-term memory](https://docs.langchain.com/oss/python/langchain/long-term-memory)，不是本组件。

---

# 官方知识点讲解

## 1. 用法：checkpointer + thread_id

**官方文档：** 创建 Agent 时传入 `checkpointer`。状态在 invoke 或某步（如工具调用）完成后更新，下一步开始时再读出。同一 `thread_id` 共享历史，不同 id 隔离。

**对应示例：** `01_checkpointer_thread.py`

**限制 / 注意：** 没传同一 `thread_id`，或没挂 checkpointer，第二轮就不会记得。

## 2. 生产环境的 checkpointer

**官方文档：** 开发可用 `InMemorySaver`。生产换数据库，例如 Postgres：

```text
uv add langgraph-checkpoint-postgres "psycopg[binary]"
```

```python
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()
    agent = create_agent(..., checkpointer=checkpointer)
```

还有 SQLite、Azure Cosmos DB 等，见 Persistence 文档。

**对应示例：** 无独立文件（需要本机 Postgres）。概念对比见 `05_memory_vs_stateless.py`。

**限制 / 注意：** `InMemorySaver` 进程重启即丢失。LangSmith 部署会自动提供 checkpointer。

## 3. 自定义 AgentState

**官方文档：** 默认短期记忆主要是 `messages`（`AgentState`）。可继承并加字段，通过 `state_schema=` 传给 `create_agent`。invoke 时把自定义字段和 messages 一起传入。

**对应示例：** `02_custom_state.py`（`home_city` + 天气工具）

## 4. 上下文膨胀的三种对策

**官方文档：** 长对话会撑爆窗口。常见做法：

| 策略 | 做什么 | 代价 |
| --- | --- | --- |
| Trim | 只留最近 N 条 / 按 token 截断 | 丢掉被删事实 |
| Delete | 用 `RemoveMessage` 从 state 永久删除 | 同上，且必须保持合法消息序 |
| Summarize | 用模型压缩旧对话 | 额外耗 token，但保留要点 |
| 自定义 | 过滤、按角色保留等 | 自己维护正确性 |

### 4.1 Trim

在 `@before_model` 里改 messages。官方示例常用 `RemoveMessage(id=REMOVE_ALL_MESSAGES)` 再写入保留列表。

**对应示例：** `03_trim_messages.py`

### 4.2 Delete

对带 `add_messages` reducer 的 key（默认 AgentState 的 messages 就是）使用 `RemoveMessage(id=某条id)` 删指定消息，或 `REMOVE_ALL_MESSAGES` 清空。

**对应示例：** 无独立文件。`03_trim_messages.py` 内部已使用 `RemoveMessage`。官方另有 `@after_model` 删除最早两条的示例。

**限制 / 注意（官方）：** 删完后历史必须合法。例如有的厂商要求以 user 开头；有 tool_calls 的 AIMessage 后面必须跟对应 ToolMessage。

### 4.3 Summarize

用内置 `SummarizationMiddleware(model=..., trigger=("tokens", 4000), keep=("messages", 20))`。`trigger` 决定何时摘要，`keep` 决定摘要后留多少新消息。

**对应示例：** `04_summarization.py`（教学把阈值调低，才能当场看到效果）

**限制 / 注意：** 短对话没必要开。生产按模型窗口设 token 阈值。

## 5. 从工具读取短期记忆

**官方文档：** 工具参数加 `runtime: ToolRuntime`（对模型隐藏）。`runtime.state["user_id"]` 等自定义字段可直接读。

**对应示例：** `02_custom_state.py`、`Tools/06_tool_runtime_state.py`

## 6. 从工具写入短期记忆

**官方文档：** 工具返回 `Command(update={...})`。若模型需要看到成功回执，update 里带 `ToolMessage(..., tool_call_id=runtime.tool_call_id)`。

**对应示例：** 无独立文件。官方模式见 Tools README「返回 Command」。本目录 `02_custom_state.py` 演示的是 invoke 时写入 `home_city`，而不是工具内 Command。

## 7. 从 Prompt 访问记忆

**官方文档：** `@dynamic_prompt` 可根据 `request.runtime.context` 或 state 生成系统提示，例如「请称呼用户为 {user_name}」。

**对应示例：** 无独立文件。属于 middleware，官方示例在 Short-term memory → Access memory → Prompt。

## 8. Before model / After model

**官方文档：**

- `@before_model`：每次调模型前处理 state（trim 就发生在这里）。流程：start → before_model → model → tools → 再回到 before_model。
- `@after_model`：模型输出后再改 state（例如删旧消息）。

**对应示例：** before_model 见 `03_trim_messages.py`。after_model 删除示例未单列文件。

## 9. 有记忆 vs 无记忆

**官方文档：** 不挂 checkpointer 时每次 invoke 独立。挂上并复用 thread_id 才有短期记忆。自己维护 messages list 是同一问题的应用层解法。

**对应示例：** `05_memory_vs_stateless.py`

## 10. 实战：多轮天气

用户先说城市，后问「那明天呢」时应沿用同一城市——这是客服/助手里最常见的短期记忆形态。

**对应示例：** `06_weather_multiturn.py`

---

## 知识点目录（本目录代码）

```text
01_checkpointer_thread.py   thread 隔离
02_custom_state.py          自定义状态 + 天气
03_trim_messages.py         裁剪历史
04_summarization.py         摘要中间件
05_memory_vs_stateless.py   有无记忆对比
06_weather_multiturn.py     实战：多轮天气追问
```

## 学习顺序

```text
checkpointer + thread_id
↓
自定义 state
↓
trim / delete / summarize
↓
工具读写 / 中间件
↓
多轮天气实战
```

## 常见问题

1. 第二轮不记得：没传同一 `thread_id`，或没挂 checkpointer。
2. InMemorySaver 重启丢失：换数据库 checkpointer。
3. 裁剪后工具调用错乱：删掉了 AIMessage，却留下孤立 ToolMessage。
4. 和长期记忆混淆：跨会话走 store，不是 thread state。

## Memory vs State

| | 自己维护 messages | Agent state + checkpointer |
| --- | --- | --- |
| 是什么 | 应用层 list | 图状态持久化 |
| 适合 | 单进程脚本 | 多请求、可恢复会话 |
| 何时选 A | 逻辑极短、完全自控 | |
| 何时选 B | 产品会话、中间件、工具读写状态 | |

## 与其他组件的关系

```text
Messages 累积
    ↓
Short-term memory 按 thread 保存
    ↓
Tools 通过 ToolRuntime 读取
    ↓
过长时 trim / summarize
    ↓
Streaming 仍按同一 thread 输出
```
