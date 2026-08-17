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

每个组件 README 都按官方文档章节展开：**官方结论 + 对应示例文件 + 限制/注意**。没有单独 `.py` 的进阶点（如 Headless tools、HITL、Deep Agents 文件系统）也写在 README 里，并标明为何没有可运行示例。

建议顺序：

```text
Models → Messages → Tools → Agents → Short-term memory → Streaming → Event streaming → Structured output → LangChain-RAG
```

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
