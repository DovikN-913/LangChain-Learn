# Tools

官方文档：https://docs.langchain.com/oss/python/langchain/tools

Tools 让模型能查数据、执行动作。底层是「带 JSON schema 的可调用函数」。

## 这个组件是什么

一个 Tool 包含：

1. 名称、描述、参数 schema（给模型看）
2. 真正执行的 Python 函数（给运行时看）

模型只负责决定「调不调、传什么参数」。执行仍在你的进程里（除非是厂商 server-side 工具或 headless 客户端工具）。

## 为什么要设计它

把「推理」和「副作用」分开：模型选工具，代码碰网络、数据库、文件系统。schema 可校验，失败可隔离，也能交给 Agent 自动循环。

天气查询一律走 `utils.get_weather`（UAPI）。

---

# 官方知识点讲解

## 1. 用 @tool 定义工具

**官方文档：** 最简单的方式是 `@tool`。类型注解决定输入 schema，docstring 决定模型何时调用。docstring 要短而具体。工具名用 `snake_case`，避免空格和特殊字符（部分厂商会拒收）。

**对应示例：** `01_basic_tool.py`

## 2. 自定义 name / description

**官方文档：** `@tool("web_search")` 覆盖函数名。`@tool("calculator", description="...")` 覆盖描述。

**对应示例：** `01_basic_tool.py`

## 3. 高级 schema：Pydantic / JSON Schema

**官方文档：** 多字段、枚举、默认值时用 Pydantic `args_schema` 或 JSON Schema。`Field(description=...)` 会进入发给模型的 schema。

**对应示例：** `02_schema_pydantic.py`

## 4. 保留参数名

**官方文档：** `config` 和 `runtime` 被保留。拿来当业务字段会运行时报错。要访问运行时信息，使用 `ToolRuntime` 参数，它会被注入且对模型隐藏。

**对应示例：** `01_basic_tool.py`（说明）、`06_tool_runtime_state.py`（正确用法）

旧注入方式（`InjectedState` / `InjectedStore` / `get_runtime()` / `InjectedToolCallId`）应迁移到 `ToolRuntime`。

## 5. 访问 State（短期记忆）

**官方文档：** `runtime.state` 是当前对话可变数据（messages + 自定义字段）。读：直接取 dict。写：返回 `Command(update={...})`，并带上对应 `ToolMessage`（`tool_call_id=runtime.tool_call_id`）。并行工具写同一字段时要配 reducer。

**对应示例：** 读取见 `06_tool_runtime_state.py`、`Short-term memory/02_custom_state.py`。写 State 的 Command 模式在 README 本节讲解，完整官方示例见 Short-term memory「从工具写入」。

## 6. Context（只读、每次调用传入）

**官方文档：** Context 是 invoke 时传入的不可变配置（user_id、会话信息）。`thread_id` 管「哪一次会话的历史」；`context` 管「这一跑工具能读到的身份」。生产里通常两者一起传。通过 `runtime.context` 读取，Agent 要设 `context_schema`。

**对应示例：** 无独立文件。官方示例用 dataclass `UserContext` + `agent.invoke(..., context=UserContext(user_id=...))`。本目录用 state 演示了类似「常驻城市」需求。

## 7. Store（长期记忆）

**官方文档：** Store 跨会话持久化（用户偏好、知识库）。通过 `runtime.store` 访问，和短期 state 不是一回事。

**对应示例：** 无。官方 Short-term memory 页明确：跨会话走 long-term memory。本任务范围是短期记忆。

## 8. Stream writer / Execution info / Server info

**官方文档：**

- `runtime.stream_writer` 或 `get_stream_writer()`：工具执行中推实时进度
- Execution info：thread_id、run_id、重试次数
- Server info：跑在 LangGraph Server 时的 assistant/graph/认证用户
- `runtime.config`：callbacks、tags、metadata
- `runtime.tool_call_id`：关联本次调用

**对应示例：** stream writer 见 `Streaming/03_custom_updates.py`。**限制：** 工具里用了 `get_stream_writer()` 后，不能在图外单独 invoke 该工具。

## 9. 工具执行与返回值

**官方文档：** Agent 里工具由框架执行；LangGraph 工作流里是 `ToolNode`。返回值有几种：

| 返回 | 行为 | 何时用 |
| --- | --- | --- |
| `str` | 变成 ToolMessage 文本 | 给人/模型读的说明 |
| `dict`/对象 | 序列化后给模型看字段 | 结构化数据 |
| 标准 content blocks | 多模态 ToolMessage | 模型支持该模态时 |
| `Command` | 更新 graph state | 改偏好、写计数器 |
| `return_direct=True` | **不再**把结果送回模型，直接结束 | 状态查询等不需再推理 |

**对应示例：** 字符串/JSON 见 `04_weather_tool.py`。`return_direct` 无独立文件：结果还要总结、串联其他工具时不要开。

## 10. bind_tools 手工循环 vs Agent

**官方文档：** `bind_tools` 只绑定 schema。没有 Agent 时必须自己「调用 → 执行 → 回填」。`create_agent` 自动完成该循环。普通函数只要有类型注解和 docstring 也能当工具。

**对应示例：** `03_bind_and_execute_loop.py`、`05_agent_with_tools.py`

## 11. 错误处理

**官方文档：** Agent 侧用中间件 `@wrap_tool_call` 捕获异常，转成模型能读的 `ToolMessage`，而不是让进程崩溃。

**对应示例：** 业务层返回 error JSON 见 `07_tool_errors.py`。中间件包装是官方推荐的 Agent 级做法，本目录未单列文件。

## 12. Dynamic tool selection

**官方文档：** 不必把所有工具始终暴露给模型。可用 `@wrap_model_call` 按鉴权、对话轮次、Store 偏好过滤 `request.tools`。若工具在运行时才发现（MCP、动态注册），还要用 `wrap_tool_call` 告诉 Agent 怎么执行这些新工具。

**对应示例：** 无独立文件。官方强调：工具太多会挤占上下文、增加误调。

## 13. Headless tools

**官方文档：** 只在服务端注册 schema，真正执行在客户端（浏览器 geolocation、剪贴板、本地文件等）。模型发出调用后图会 **interrupt**，客户端做完再 resume。Python 侧 `tool(name=..., description=..., args_schema=...)` 且没有函数体时得到 `HeadlessTool`，没有 `.implement()`。

**对应示例：** 无。需要前后端 handshake，超出本学习单元的单脚本范围。

## 14. Prebuilt tools / Server-side tool use

**官方文档：** LangChain 有大量现成 toolkit（搜索、代码、数据库）。另外，部分 Chat Model 自带服务端工具（web search、code interpreter），由厂商执行，不必你托管逻辑。

**对应示例：** 无。本项目实战工具是 UAPI 天气，属于客户端自定义工具。

---

## 知识点目录（本目录代码）

```text
01_basic_tool.py              @tool 基础
02_schema_pydantic.py         复杂 schema + 真实天气裁剪
03_bind_and_execute_loop.py   手工工具循环
04_weather_tool.py            实战：改造 utils 天气工具
05_agent_with_tools.py        create_agent 自动编排
06_tool_runtime_state.py      ToolRuntime 读短期状态
07_tool_errors.py             参数错误与 API 失败
```

## 学习顺序

```text
定义工具
↓
schema
↓
手工执行循环
↓
真实天气工具
↓
Agent
↓
Runtime / 错误处理
↓
README 中的 Command / Context / Headless / 动态工具
```

## 依赖

```text
uv add langchain langchain-openai uapi-sdk-python python-dotenv
```

`.env` 需要 `UAPI_API_KEY` 以及 DashScope 三项配置。

## 常见问题

1. 工具没执行：只做了 `bind_tools`。
2. 模型乱调工具：描述太泛，或没写 system prompt。
3. 参数名用了 `config`/`runtime`：运行时冲突。
4. 天气工具 print 了结果：模型读不到，必须 `return`。
5. `return_direct=True` 后模型不再总结：不要用在还需要推理的工具上。

## 与其他组件的关系

```text
Models.bind_tools 发出调用
    ↓
Tools 执行
    ↓
Messages.ToolMessage 回填
    ↓
Agent 循环 / Short-term memory
    ↓
Streaming 展示工具进度
```

## Tool Calling vs Agent

| | Tool Calling | Agent |
| --- | --- | --- |
| 是什么 | 模型输出 tool_calls | 自动「想-做-再想」 |
| 谁执行工具 | 你 | LangChain |
| 何时用 | 只要一次调用、要精确控流 | 多工具、未知步数 |
