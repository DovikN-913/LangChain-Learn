# Models

官方文档：https://docs.langchain.com/oss/python/langchain/models

LangChain 把各厂商大模型收成同一套 Chat Model 接口。上层的 Agent、Tool Calling、Structured Output 都建立在它之上。

## 这个组件是什么

Chat Model 负责「把 messages 送进模型，再把 AIMessage 拿回来」。LangChain 1.x 明确推荐 Chat Model，而不是旧式 Completion LLM。

## 为什么要设计它

没有统一接口时，换厂商就要改业务代码。有了标准 `invoke` / `stream` / `batch` / `bind_tools` / `with_structured_output`，应用逻辑可以保持稳定。

模型可以单独调用，也可以交给 Agent。官方文档强调：**同一套模型接口同时服务这两种用法**。

本项目模型走阿里百炼 OpenAI 兼容接口（`.env` 中 `DASHSCOPE_*`）。Qwen3.6 默认带 thinking，教学示例通过 `extra_body={"enable_thinking": False}` 关闭。

---

# 官方知识点讲解

每一节都对应官方文档一个章节。标记含义：

- **官方文档**：官方明确给出的结论
- **对应示例**：本目录可运行文件；没有文件则说明原因
- **限制 / 注意**：官方限制 + 本项目（百炼兼容接口）实测差异

## 1. 初始化模型

**官方文档：** 最简单的入口是 `init_chat_model`。也可以直接用厂商类，例如 `ChatOpenAI`。传入 `{provider}:{model}` 可同时指定厂商和模型名。

**对应示例：** `01_init_chat_model.py`

**限制 / 注意：**

- 旧写法 `from langchain.llms import OpenAI`（Completion LLM）已不推荐。
- 百炼不是内置 provider，必须 `model_provider="openai"` + `base_url`，或直接 `ChatOpenAI(base_url=...)`。
- **工程实践：** 密钥只从 `.env` 读取，不要写进代码。

## 2. 支持的厂商与模型

**官方文档：** 各厂商通过独立集成包实现同一套标准接口，换厂商不必改业务逻辑。新模型名会直接传给厂商 API，不必等 LangChain 发版。

**对应示例：** 无独立文件。本项目固定 DashScope。完整厂商列表见 [integrations](https://docs.langchain.com/oss/python/integrations/providers/overview)。

**限制 / 注意：** `model_provider="openai"` 表示「走 OpenAI Chat Completions 协议」，不等于「一定是 OpenAI 官方服务」。路由器/代理的专有字段可能不会被完整保留。

## 3. 关键方法

**官方文档：** Chat Model 的核心方法是 `invoke`（完整消息）、`stream`（边生成边产出）、`batch`（客户端并行多个独立请求）。Embedding / VectorStore 是相邻技术，不在本页展开。

**对应示例：** `02_model_invoke.py`、`03_model_stream.py`、`04_model_batch.py`、`09_invoke_vs_stream_vs_batch.py`

## 4. 参数

**官方文档：** 通用参数包括：

| 参数 | 作用 |
| --- | --- |
| model | 模型名，也可用 `provider:model` |
| api_key | 鉴权，通常来自环境变量 |
| temperature | 越高越发散，越低越确定 |
| max_tokens | 限制生成长度 |
| timeout | 单次请求等待上限（秒） |
| max_retries | 失败重试次数 |

用 `init_chat_model(..., temperature=0.7)` 以 kwargs 传入。

**对应示例：** `06_model_parameters.py`

**限制 / 注意：** 各集成还有厂商专有参数。例如 `ChatOpenAI` 的 `use_responses_api`。百炼 Qwen 的 thinking 用 `extra_body={"enable_thinking": ...}`，这是厂商扩展，不是 LangChain 标准参数。

## 5. 连接韧性

**官方文档：** 默认最多重试约 6 次，指数退避。会重试网络错误、429、5xx；**不会**重试 401、404。长任务/不稳网络可把 `max_retries` 调到 10–15，并配合 checkpointer。

**对应示例：** `06_model_parameters.py`、`10_error_handling.py`

## 6. invoke

**官方文档：** `invoke` 一次请求，返回完整 `AIMessage`。输入可以是字符串、消息对象列表、OpenAI 风格 dict。字符串会被当成单条 HumanMessage。

**对应示例：** `02_model_invoke.py`

**限制 / 注意：** 如果返回值是 `str`，说明误用了旧式 LLM。Chat Model 前缀是 `Chat`，例如 `ChatOpenAI`。

## 7. stream

**官方文档：** `stream()` 返回 `AIMessageChunk` 迭代器。chunk 可以相加拼回完整消息：`full = chunk if full is None else full + chunk`。也可从 `content_blocks` 里读 reasoning / tool_call_chunk。`astream_events()` 能按事件类型过滤，并在后台聚合完整消息。

在 Agent 图里即使节点写的是 `model.invoke()`，只要外层在 streaming，LangChain 也会内部切到流式并触发 `on_llm_new_token`。

**对应示例：** `03_model_stream.py`；Agent 流式见 `Streaming/`、`Event streaming/`

**限制 / 注意：** 下游如果必须等全文才能处理，不要用 stream。流式链路里每一步都要会处理 chunk。

## 8. batch

**官方文档：** `batch()` 是**客户端并行**，不是 OpenAI/Anthropic 的厂商 Batch API。结果按输入顺序返回。`batch_as_completed()` 谁先完成谁先返回，可能乱序，带原始下标。用 `config={"max_concurrency": 5}` 限制并发。

**对应示例：** `04_model_batch.py`

**限制 / 注意：** 不要用 batch 模拟多轮对话。后一个问题依赖前一个答案时，必须把 messages 串起来。

## 9. 异步

**官方文档：** 同步方法都有异步对应：`ainvoke` / `astream` / `abatch`。Web 服务应走异步，避免阻塞事件循环。

**对应示例：** `05_model_async.py`

## 10. Tool calling

**官方文档：** 工具 = schema（名字、描述、参数）+ 可执行函数。`bind_tools` 只把 schema 交给模型；模型返回的是**调用请求**，不会自动执行。自己执行后要把 `ToolMessage` 回填。`create_agent` 才会自动循环。

相关能力：

- `tool_choice="any"` / 指定工具名：强制调用
- 多数模型默认允许并行 tool calls；可用 `parallel_tool_calls=False` 关掉
- 流式时参数以 `ToolCallChunk` 逐步拼出

**对应示例：** `07_model_tool_calling.py`（真实 UAPI 天气）。完整 Agent 循环见 `Tools/`。

**限制 / 注意：** 函数调用（function calling）和 tool calling 在官方文档里当作同一件事。

## 11. Structured output（模型层）

**官方文档：** `with_structured_output(schema)` 约束输出。schema 可以是 Pydantic、TypedDict、JSON Schema。`method` 常见取值：

- `json_schema`：厂商原生结构化能力
- `function_calling`：用强制工具调用推导结构
- `json_mode`：只保证 JSON，schema 要写进 prompt

`include_raw=True` 同时返回解析结果和原始 AIMessage。

**对应示例：** `08_model_structured_output.py`

**限制 / 注意：**

- Agent 场景请用 `create_agent(response_format=...)`，见 `Structured output/`。
- **工程实践（百炼实测）：** 默认 `json_schema`/`json_object` 会 400，要求 prompt 含 `"json"`。本项目显式 `method="function_calling"`。

## 12. Model profiles（进阶）

**官方文档：** `langchain>=1.1`。`model.profile` 描述能力：上下文窗口、是否支持图片、reasoning、tool calling、structured output 等。数据主要来自 models.dev。Agent 会用它自动选择结构化策略、决定是否摘要。可用 `profile=` 覆盖。这是 **beta**，格式可能变。

**对应示例：** 无独立文件。本项目用兼容接口时 profile 往往不完整，所以 Structured output 目录显式指定 `ToolStrategy`。

## 13. Multimodal

**官方文档：** 能处理图/音/视频的模型，可通过 messages 的 content blocks 传入。同时接受：LangChain 标准块、OpenAI chat completions 格式、厂商原生格式。部分模型也能在 `AIMessage.content_blocks` 里返回图片。

**对应示例：** 结构演示在 `Messages/05_content_blocks.py`。本目录无独立视觉模型调用。

**限制 / 注意：** 当前 `qwen3.6-35b-a3b` 是文本模型，不能当真视觉模型用。

## 14. Reasoning

**官方文档：** 若模型支持，可从 `content_blocks` 里 `type=="reasoning"` 取出思考过程。`reasoning_effort` 是标准参数（需较新的 partner 包），可在构造或单次调用时设置。也可用厂商别名（如 Anthropic `effort`）。

**对应示例：** `06_model_parameters.py` 用 `enable_thinking=True` 观察 usage 里的 reasoning token。

**限制 / 注意：** 百炼兼容接口不走 `reasoning_effort` 标准参数，而用 `extra_body.enable_thinking`。开启后一次简单问答可能消耗上千 reasoning token。

## 15. Local models

**官方文档：** 可在本地跑模型（隐私、自定义、省云费用）。Ollama 是常见入口。

**对应示例：** 无。本项目使用云端百炼。

## 16. Prompt caching

**官方文档：** 三级：厂商隐式缓存、厂商显式控制（如 `prompt_cache_key`、Anthropic `cache_control`）、Agent 中间件缓存稳定的 system/tool 内容。通常有最小 token 门槛。命中情况会出现在 usage metadata。

**对应示例：** 无。取决于厂商，百炼兼容接口未作为本项目教学重点。

## 17. Server-side tool use

**官方文档：** 部分厂商在服务端完成 web search / code interpreter。一次对话里就能看到 `server_tool_call` / `server_tool_result` content blocks，**不需要**你回填 `ToolMessage`。

**对应示例：** 无。这是 OpenAI 等厂商内置工具，不是 `utils.get_weather` 这种客户端工具。

## 18. Rate limiting

**官方文档：** 可用 `InMemoryRateLimiter(requests_per_second=..., max_bucket_size=...)` 在初始化时传入 `rate_limiter=`。它只限制请求次数，不限制请求体积。

**对应示例：** 无独立文件。限流报错见 `10_error_handling.py` 的异常处理思路。

## 19. Base URL 与代理

**官方文档：** 对实现了 OpenAI Chat Completions 的服务，可配自定义 `base_url`。

**对应示例：** `01_init_chat_model.py`、`utils.get_chat_model()`

**限制 / 注意：** 这正是本项目对接百炼的方式。

## 20. Log probabilities

**官方文档：** 部分模型支持 `.bind(logprobs=True)`，结果在 `response_metadata["logprobs"]`。

**对应示例：** 无。百炼兼容接口不一定暴露该字段。

## 21. Token usage

**官方文档：** 用量在 `AIMessage.usage_metadata`。OpenAI/Azure 流式场景可能要额外 opt-in。跨多次调用可用 `UsageMetadataCallbackHandler` 或 `get_usage_metadata_callback()` 聚合。

**对应示例：** `06_model_parameters.py`、`Messages/03_ai_message_metadata.py`

## 22. Invocation config / Configurable / Dynamic model

**官方文档：**

- `RunnableConfig` 可传 callbacks、tags、metadata、`max_concurrency`。
- Configurable models：运行时用配置切换模型。
- Dynamic model selection：按状态/上下文换模型。

**对应示例：** `max_concurrency` 见 `04_model_batch.py`。后两项无独立文件，属于 Agent/中间件进阶，官方放在 Models 进阶里。

---

## 知识点目录（本目录代码）

```text
01_init_chat_model.py            三种初始化方式
02_model_invoke.py               invoke 与三种输入形态
03_model_stream.py               流式与 chunk 累加
04_model_batch.py                batch / batch_as_completed
05_model_async.py                异步接口
06_model_parameters.py           参数、重试、thinking
07_model_tool_calling.py         bind_tools + 真实天气查询
08_model_structured_output.py    with_structured_output
09_invoke_vs_stream_vs_batch.py  对比：何时选哪种调用
10_error_handling.py             错误模型名、超时
```

## 学习顺序

```text
初始化
↓
invoke
↓
stream / batch / async
↓
参数与异常
↓
tool calling / structured output
↓
进阶能力（profile / reasoning / 限流）读 README
↓
与 Messages、Tools、Agent 组合
```

## 依赖

```text
uv add langchain langchain-openai python-dotenv
```

```text
uv run python Models/02_model_invoke.py
```

## 常见问题

1. **拿到的是 str 而不是 AIMessage**：误用了旧 LLM。
2. **工具没有真正执行**：`bind_tools` 只产生调用请求。
3. **一次问答特别慢、token 很多**：Qwen thinking 开着。
4. **401**：检查 `DASHSCOPE_API_KEY`。
5. **with_structured_output 报 json_object 400**：显式 `method="function_calling"`。

## 与其他组件的关系

```text
Messages 提供输入/输出载体
    ↓
Models 负责推理
    ↓
Tools 扩展模型能做的事
    ↓
Structured Output 约束返回形态
    ↓
Streaming / Event streaming 把过程推给前端
    ↓
Short-term memory 跨轮次记住上下文
```

## 旧写法 vs 当前推荐

```text
旧版本：from langchain.llms import OpenAI
当前：  from langchain.chat_models import init_chat_model
        或 from langchain_openai import ChatOpenAI
```
