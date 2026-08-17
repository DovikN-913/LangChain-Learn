# Agents

官方文档：https://docs.langchain.com/oss/python/langchain/agents

Agent 是「模型在循环里调用工具，直到任务完成」。套在循环外面的一切——提示、工具、中间件——叫做 **harness**。

> **Agent = Model + Harness**  
> Harness 的工作：在正确的时间，把正确的上下文交给模型。

[`create_agent`](https://reference.langchain.com/python/langchain/agents/factory/create_agent) 是官方高度可配置的 harness。更重的长任务栈（规划、文件系统、子 Agent、记忆）预装在 [Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview) 里；需要自己组装时用 `create_agent`。

本项目模型走阿里百炼兼容接口；天气工具走 `utils.get_weather`。

## 这个组件是什么

一次 `create_agent` 调用会得到编译后的图。图内部循环：

```text
用户输入
    ↓
模型决定：回答 or 调工具
    ↓
执行工具，写回 ToolMessage
    ↓
再交给模型……直到不再调工具
```

## 为什么要设计它

`bind_tools` 只解决「模型会点菜」。真实产品还要：自动循环、会话记忆、流式进度、结构化出口、重试、PII、人工审批。这些都是 harness 的职责，不该每次手写。

---

# 官方知识点讲解

每一节对应官方 Agents 页的一个章节。

## 1. 核心循环与 create_agent

**官方文档：** Agent 就是模型循环调工具直到任务完成。最简写法：`create_agent(model=..., tools=...)`。基础能力用 `model` / `tools` / `system_prompt` 配置；进阶用 middleware。

**对应示例：** `01_create_agent_loop.py`

**限制 / 注意：** 旧版 `initialize_agent` / `AgentExecutor` 已不是 1.x 推荐入口。当前推荐 `create_agent`。

## 2. Model

**官方文档：** 可传 `"provider:model"` 字符串，或已初始化的模型实例。参数、厂商、动态选模型见 Models 页。

**对应示例：** `02_harness_model_tools_prompt.py`（传入 `get_chat_model()` 实例）

**限制 / 注意：** 本项目不能写 `model="openai:gpt-5.5"`，必须走 DashScope 的 `ChatOpenAI(base_url=...)`。

## 3. Tools

**官方文档：** 可传普通 Python 可调用对象、LangChain Tool、或 tool dict。定义、上下文、动态选择见 Tools 页。

**对应示例：** `01_create_agent_loop.py`（普通函数）、`02_harness_model_tools_prompt.py`（`@tool` + 普通函数混用）

## 4. System prompt

**官方文档：** 塑形 Agent 如何做事。接受 `str` 或 `SystemMessage`。运行时动态提示用 [middleware](https://docs.langchain.com/oss/python/langchain/middleware)。

**对应示例：** `02_harness_model_tools_prompt.py`

## 5. Structured output

**官方文档：** `response_format=` 让 Agent 返回校验后的 schema。策略与细节见 Structured output 页。结果在 `result["structured_response"]`。

**对应示例：** `06_structured_output.py`、`11_weather_assistant.py`

**限制 / 注意：** 百炼兼容接口请用 `ToolStrategy(...)`，不要依赖自动 ProviderStrategy。

## 6. Agent state

**官方文档：** 每个 Agent 用 `AgentState`（TypedDict）管理执行上下文。内置字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `messages` | `list[BaseMessage]` | 当前 thread 完整对话。只追加，不替换 |

`before_model` / `after_model` 等钩子收到这份 state，返回的 dict 会合并回去。自定义字段：子类化 `AgentState`，经 `state_schema=` 传入。

**对应示例：** `03_agent_state.py`

完整自定义记忆见 Short-term memory。

## 7. Invocation

**官方文档：** `invoke` 是在更新 Agent State。所有 Agent 的 state 都包含 messages；调用时传入新消息，并带 `thread_id` 以便持久化/恢复。

本地必须显式 `checkpointer=InMemorySaver()`（或数据库 checkpointer）。LangSmith 部署会自动提供。

`thread_id` 管**会话**（历史、checkpoint）；`context` 管**本轮**只读数据（user_id、功能开关）。生产里两者经常一起传。`context_schema` 定义形状，工具经 `runtime.context` 读取。

**对应示例：** `04_invoke_thread_context.py`

## 8. Streaming

**官方文档：** `invoke` 只在跑完后返回。多步工具时要用流式把中间消息和工具活动推出去。Agents 页示例用 `stream_events(..., version="v3")` 迭代 `stream.values` 快照。更完整的模式见 Streaming / Event streaming 页。

**对应示例：** `05_agent_streaming.py`

## 9. 配置 Harness（中间件）

**官方文档：** Middleware 是定制原语：一件事一个中间件，钩进循环的正确位置，可自由组合。常见模式已预置；其余可写 [custom middleware](https://docs.langchain.com/oss/python/langchain/middleware/custom)。

官方把能力分成六类：

### 9.1 Execution environment

给 Agent 工作区：工具、跨轮文件系统、沙箱/代码执行。官方示例用 Deep Agents 的 `FilesystemMiddleware` + `StateBackend`（需 `uv add deepagents`）。

**对应示例：** 无独立文件。本学习单元不引入 deepagents。客户端工具执行见 `01` / `11`。

### 9.2 Context management

上下文窗口会填满历史和工具结果。摘要在溢出前压缩；memory 在启动时加载跨会话指令；skills 按需加载领域知识。官方组合示例同样来自 deepagents。langchain 内置的 `SummarizationMiddleware` 已在 `Short-term memory/04_summarization.py`。

**对应示例：** 摘要见 Short-term memory；Deep Agents 的 Memory/Skills 不在本目录安装。

### 9.3 Planning and delegation

复杂任务超出单窗口时：主 Agent 拆任务，交给隔离上下文的子 Agent，可并行，主上下文保持干净。

- `TodoListMiddleware`：langchain 内置待办
- `SubAgentMiddleware`：deepagents 子 Agent

**对应示例：** `09_todo_list.py`（TodoList）。子 Agent 无独立文件（需 deepagents）。给 Agent 起名：`create_agent(..., name="research_assistant")`，见 `01_create_agent_loop.py` 的 `name=`。

### 9.4 Fault tolerance

生产会出现开发时少见的 429、超时、瞬时错误。`ModelRetryMiddleware` / `ToolRetryMiddleware` 在基础设施层处理，业务工具不必每个都 try/catch。

**对应示例：** `07_fault_tolerance.py`

### 9.5 Guardrails

有些策略必须确定性执行，不能只靠 prompt。`PIIMiddleware` 在循环中拦截邮箱等 PII。

**对应示例：** `08_guardrails_pii.py`

### 9.6 Steering（Human-in-the-loop）

高影响动作前暂停，等人 approve / edit / reject。`HumanInTheLoopMiddleware(interrupt_on={"write_file": True})`，需要 checkpointer，用 `Command` 恢复。细节见 [HITL](https://docs.langchain.com/oss/python/langchain/human-in-the-loop)。

**对应示例：** 无独立可无人值守脚本（会停住等输入）。流式收集 interrupt 的模式见 Streaming README。

### 9.7 中间件文档入口

- [Middleware overview](https://docs.langchain.com/oss/python/langchain/middleware/overview)
- [Prebuilt middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in)
- [Custom middleware](https://docs.langchain.com/oss/python/langchain/middleware/custom)

---

## 知识点目录（本目录代码）

```text
01_create_agent_loop.py              核心循环 + name
02_harness_model_tools_prompt.py     model / tools / system_prompt
03_agent_state.py                    自定义 AgentState
04_invoke_thread_context.py          thread_id + context
05_agent_streaming.py                stream_events 快照
06_structured_output.py              response_format + 天气
07_fault_tolerance.py                模型/工具重试
08_guardrails_pii.py                 PII 拦截
09_todo_list.py                      规划中间件 + 多工具
10_agent_vs_bind_tools.py            对比：何时选 Agent
11_weather_assistant.py              实战：记忆 + 重试 + 结构化天气卡片
```

## 学习顺序

```text
核心循环
↓
harness 三件套
↓
state / invoke / context
↓
流式
↓
结构化输出
↓
中间件（重试、PII、待办）
↓
对比 bind_tools
↓
实战出行助手
```

## 常见问题

1. 第二轮不记得：没挂 checkpointer，或 `thread_id` 变了。
2. 工具没执行：那是 `bind_tools`，不是 Agent 循环。
3. 找不到结构化结果：看 `structured_response`，不是最后一条 AIMessage。
4. ProviderStrategy 400：百炼用 `ToolStrategy`。
5. HITL 脚本「卡住」：这是预期，需要 resume Command。
6. 和 Deep Agents 混淆：`create_agent` 自己组装；`create_deep_agent` 预装文件系统/子 Agent/记忆。

## Agent vs bind_tools vs Deep Agents

| | bind_tools | create_agent | create_deep_agent |
| --- | --- | --- | --- |
| 是什么 | 模型输出调用请求 | 可配置工具循环 harness | 预装长任务 harness |
| 谁执行工具 | 你 | LangChain | LangChain + 更多中间件 |
| 何时选 | 一轮、精确控流 | 产品助手、要中间件 | 编码/研究等长任务 |

## 与其他组件的关系

```text
Models + Messages + Tools
        ↓
    create_agent（本目录）
        ↓
Short-term memory / Structured output / Streaming
        ↓
Middleware（重试、PII、HITL、摘要……）
```

## 旧写法 vs 当前推荐

```text
旧版本：initialize_agent / AgentExecutor / langchain.agents.load_tools
当前：  from langchain.agents import create_agent
```
