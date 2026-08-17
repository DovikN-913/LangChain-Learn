# LangChain 1.x 核心组件学习项目

基于 [LangChain 最新官方文档](https://docs.langchain.com/oss/python/langchain/models)，把核心组件做成可运行的 Python 学习单元。

模型统一走阿里百炼 **OpenAI 兼容接口**（`.env` 中 `DASHSCOPE_*`）。天气类工具统一改造自根目录 `utils.py` 的 UAPI 查询，不使用假数据。

## 组件一览

| 目录 | 官方文档 | 你将学会 |
| --- | --- | --- |
| [Models](Models/README.md) | [models](https://docs.langchain.com/oss/python/langchain/models) | 初始化、invoke/stream/batch、参数、tool calling |
| [Messages](Messages/README.md) | [messages](https://docs.langchain.com/oss/python/langchain/messages) | 四种消息、元数据、ToolMessage、多轮历史 |
| [Tools](Tools/README.md) | [tools](https://docs.langchain.com/oss/python/langchain/tools) | @tool、schema、手工循环、真实天气、Agent |
| [Agents](Agents/README.md) | [agents](https://docs.langchain.com/oss/python/langchain/agents) | create_agent、harness、state、中间件、多工具天气助手 |
| [Short-term memory](Short-term%20memory/README.md) | [short-term-memory](https://docs.langchain.com/oss/python/langchain/short-term-memory) | checkpointer、thread、裁剪与摘要 |
| [Streaming](Streaming/README.md) | [streaming](https://docs.langchain.com/oss/python/langchain/streaming) | updates / messages / custom |
| [Event streaming](Event%20streaming/README.md) | [event-streaming](https://docs.langchain.com/oss/python/langchain/event-streaming) | `stream_events` v3 投影 |
| [Structured output](Structured%20output/README.md) | [structured-output](https://docs.langchain.com/oss/python/langchain/structured-output) | response_format、两种 Strategy |
| [LangChain-RAG](LangChain-RAG/README.md) | [retrieval](https://docs.langchain.com/oss/python/langchain/retrieval) | Loader、分块、Embedding、Milvus、2-Step / Agent 知识问答 |
| [LangGraph-入门](LangGraph-入门) | [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) | State、顺序/分支/循环、人机协作、写作工作流 |

每个组件 README 都按官方文档章节展开：**官方结论 + 对应示例文件 + 限制/注意**。没有单独 `.py` 的进阶点（如 Headless tools、HITL、Deep Agents 文件系统）也写在 README 里，并标明为何没有可运行示例。

## 学习路线

先配环境，再按阶段往上叠。前一阶段解决「模型怎么说话、工具怎么被调用」，后面才是记忆、流式、图编排和 RAG。同一道天气题会在 Tools / Agents / Memory / Streaming 里反复出现，方便对照同一能力在不同层长什么样。

每个目录先读该目录 `README.md`，再按文件名前缀 `01`、`02`… 往下跑。

### 第 0 阶段：能跑通一次调用

目标：密钥、依赖、模型接口都通。

1. `uv sync`
2. 复制 `.env.example` → `.env`，填 `DASHSCOPE_*` 和 `UAPI_API_KEY`
3. 需要 Milvus / Postgres 时，再复制 `config.example.py` → `config.py`

验证：

```text
uv run python Models/02_model_invoke.py
```

### 第 1 阶段：模型与消息（地基）

目标：会调 Chat Model，分得清四种消息，知道多轮历史是怎么拼的。

| 顺序 | 目录 | 先跑这些 | 你该带走的结论 |
| --- | --- | --- | --- |
| 1 | [Models](Models/README.md) | `01` 初始化 → `02` invoke → `03` stream → `04` batch → `09` 三种调用对比 | 同一模型接口同时服务单次调用和 Agent |
| 2 | [Models](Models/README.md) | `06` 参数 → `07` tool calling → `08` 结构化输出 → `10` 错误处理 | `bind_tools` 只是「模型会点菜」，还不会自动执行 |
| 3 | [Messages](Messages/README.md) | `01` 四种消息 → `04` ToolMessage → `06` 多轮对话 | Agent 的短期记忆，底层就是 messages 列表 |

本阶段不必等 Agent。把 `Models/07` 和 `Messages/04` 对照着看，后面 Tools 里的手工循环会立刻对上。

### 第 2 阶段：工具与 Agent（从点菜到上菜）

目标：工具能被执行；Agent 能自己循环，直到不再调工具。

| 顺序 | 目录 | 先跑这些 | 你该带走的结论 |
| --- | --- | --- | --- |
| 4 | [Tools](Tools/README.md) | `01` `@tool` → `02` schema → `03` 手工循环 → `04` 真实天气 | 自己写循环：模型点菜，你来上菜 |
| 5 | [Tools](Tools/README.md) | `05` Agent 挂工具 → `06` runtime state → `07` 工具错误 | 同一天气工具，后面所有 Agent 示例都复用 |
| 6 | [Agents](Agents/README.md) | `01` 核心循环 → `02` harness → `03` state → `04` thread | Agent = Model + Harness；`create_agent` 是编译后的图 |
| 7 | [Agents](Agents/README.md) | `05`～`09` 流式 / 结构化 / 容错 / 护栏 / todo → `10` vs bind_tools → `11` 天气助手 | 中间件往循环外挂能力，不必重写循环 |

阶段检查：`uv run python Agents/11_weather_assistant.py` 能查真实天气并给出完整回答。

### 第 3 阶段：记忆、流式、结构化出口

目标：多轮能追问、前端能打字机显示、答案能按 schema 落地。

| 顺序 | 目录 | 先跑这些 | 你该带走的结论 |
| --- | --- | --- | --- |
| 8 | [Short-term memory](Short-term%20memory/README.md) | `01` checkpointer + thread → `03` trim → `04` 摘要 → `05` 有无记忆对比 → `06` 多轮天气 | 没挂 checkpointer，或 `thread_id` 变了，第二轮就会失忆 |
| 9 | [Streaming](Streaming/README.md) | `01` updates → `02` messages → `03` custom → `04` 对比 | v2 信封是 `{type, ns, data}`；要打字机选 `messages`，要看节点选 `updates` |
| 10 | [Event streaming](Event%20streaming/README.md) | `01` v3 投影 → `02` tool_calls → `04` 与 stream_mode 对比 | 新 UI 优先 `stream_events(version="v3")` |
| 11 | [Structured output](Structured%20output/README.md) | `01` response_format → `02` Pydantic vs TypedDict → `03` 两种 Strategy → `04` 天气报告 | Provider 策略走模型原生 schema；Tool 策略把 schema 当工具 |

阶段检查：`uv run python "Short-term memory/06_weather_multiturn.py"` 第二轮能理解「那」指贵阳。

### 第 4 阶段：图编排（看清 Agent 底下那张图）

目标：自己画 State、边、分支、循环；理解 `create_agent` 为什么是一张编译后的图。

| 顺序 | 目录 | 先跑这些 | 你该带走的结论 |
| --- | --- | --- | --- |
| 12 | [LangGraph-入门](LangGraph-入门) | `01` 顺序 → `02` 条件分支 → `03` 循环 → `04` 人机协作 → `05` 文章写作 | State 是共用记事本；边决定下一步；HITL 靠 checkpointer 把图停住再恢复 |

这一阶段放在 Agent 之后：先会用现成 harness，再拆开看图。不必和 RAG 抢顺序，但写复杂工作流前建议先跑完 `01`～`04`。

### 第 5 阶段：RAG 实战（把外部知识接进来）

目标：会入库、会检索、会做成 2-Step 和 Agentic 两种问答。需要本机/局域网 Milvus（`config.py`）。

| 顺序 | 目录 | 先跑这些 | 你该带走的结论 |
| --- | --- | --- | --- |
| 13 | [LangChain-RAG](LangChain-RAG/README.md) | `01` Loader → `02` 分块 → `03` Embedding → `04` Milvus | 切块带 `source`；查询和入库必须同一 Embedding 模型 |
| 14 | [LangChain-RAG](LangChain-RAG/README.md) | `05` 最小流水线 → `06` 企业 Chain（2-Step）→ `07` 企业 Agent（按需检索） | Chain 每次必检索；Agent 自己决定要不要搜 |

阶段检查：问 `knowledge_docs/` 里的制度或 FAQ，答案能溯源到具体文件，而不是模型编造。

### 怎么自学效率更高

1. **纵向走完一条线，再横向对比。** 例如先把天气从 `Tools/04` 跑到 `Agents/11` 再到 `Short-term memory/06`，比把 Models 目录 10 个文件一次看完更不容易晕。
2. **注释分三种口气：** 「官方文档」是结论，「源码分析」是实现，「工程实践」是本仓库约定。卡住时先看这三行。
3. **同一问题换一层再问一遍。** 「贵阳今天天气怎么样？」在 Tools、Agents、Streaming、Structured output 里各跑一次，差异就是各组件的职责边界。
4. **RAG 不要提前插入。** 它依赖 Embedding、检索器和 Agent，跳过第 1～2 阶段会很难判断是模型问题、切块问题还是向量库问题。

## 环境

Python 3.12。在项目根目录：

```text
uv sync
```

`.env` 至少包含：

```text
DASHSCOPE_API_KEY=...
DASHSCOPE_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_CHAT_MODEL_NAME=qwen3.6-35b-a3b
DASHSCOPE_EMBEDDING_MODEL_NAME=text-embedding-v1
UAPI_API_KEY=...
```

运行示例：

```text
uv run python Models/02_model_invoke.py
uv run python Tools/04_weather_tool.py
uv run python Agents/01_create_agent_loop.py
uv run python "Short-term memory/06_weather_multiturn.py"
uv run python LangChain-RAG/03-文本Embedding.py
```

公共能力在 `utils.py`：

- `get_chat_model()`：DashScope Chat Model，默认关闭 Qwen thinking
- `get_embedding_model()`：DashScope Embedding（`chunk_size=25`，给 RAG 用）
- `get_weather(city)`：UAPI 真实天气，返回 JSON 字符串给 Tool 使用

## 设计约定

1. 一个 Python 文件讲一个知识点，注释解释「为什么」而不是复述代码。
2. 密钥只从 `.env` 读取。
3. 官方结论、源码分析和工程实践会在注释里区分。
4. 天气相关示例都调用 `utils.get_weather`，便于对照 Tool / Agent / Memory / Streaming。
