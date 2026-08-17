# Structured output

官方文档：https://docs.langchain.com/oss/python/langchain/structured-output

结构化输出让 Agent/模型返回 JSON、Pydantic 实例或 dict，而不是再从散文里抠字段。

> 模型层 `with_structured_output` 见 [Models - Structured output](https://docs.langchain.com/oss/python/langchain/models#structured-output) 和 `Models/08_model_structured_output.py`。**本目录聚焦 `create_agent(response_format=...)`。**

## 这个组件是什么

给 Agent 一个 schema。模型生成符合 schema 的数据后，框架校验并放进 `state["structured_response"]`。

## 为什么要设计它

自然语言不稳定。订单、天气卡片、工单这些下游系统需要字段级数据。结构化输出把「生成」变成「填写表单」。

本项目模型是百炼兼容接口，示例默认 **ToolStrategy**。

---

# 官方知识点讲解

## 1. response_format

**官方文档：** `create_agent(..., response_format=...)` 接受：

| 值 | 行为 |
| --- | --- |
| `ToolStrategy[T]` | 用工具调用得到结构 |
| `ProviderStrategy[T]` | 用厂商原生 structured output |
| `type[T]`（直接传类） | 按模型能力自动选策略 |
| `None` | 不要求结构化 |

直接传 schema 类型时：若 profile 显示支持原生结构化（OpenAI / Anthropic / xAI 等）→ Provider；否则 → Tool。`langchain>=1.1` 会动态读 model profile；没有数据时需手动指定或传入 `profile={"structured_output": True, ...}`。

**JSON Schema 字典必须包进显式 Strategy，不能裸传给 `response_format`。**

若同时配了 tools，模型必须支持「工具 + 结构化」一起用。

最终结果在 `result["structured_response"]`，不是只在最后一条 AIMessage 里。

**对应示例：** `01_agent_response_format.py`

## 2. ProviderStrategy

**官方文档：** 部分厂商 API 原生约束输出（OpenAI、xAI、Gemini、Anthropic）。`ProviderStrategy(schema, strict=None)`。`strict` 需 `langchain>=1.2`，部分厂商（OpenAI、xAI）支持严格遵从 schema。

schema 支持：Pydantic（返回实例）、dataclass（返回 dict）、TypedDict（返回 dict）、JSON Schema（须有顶层 title/description，返回 dict）。

传类名和传 `ProviderStrategy(ProductReview)` 在厂商支持时功能等价；不支持则回退到 tool calling。

**对应示例：** `03_provider_vs_tool_strategy.py`

**限制 / 注意（百炼实测）：** ProviderStrategy 会走 `json_object`，报 400（prompt 须含 json）。本项目不要把它当默认策略。

## 3. ToolStrategy

**官方文档：** 对所有支持 tool calling 的模型可用。把 schema 当成强制工具。参数：

- `schema`：Pydantic / dataclass / TypedDict / JSON Schema / **Union（多 schema，由模型选一个）**
- `tool_message_content`：结构化成功后写入对话的 ToolMessage 文本；默认会把结构数据写进去
- `handle_errors`：校验失败如何处理，默认 `True`

**对应示例：** `01_agent_response_format.py`、`03_provider_vs_tool_strategy.py`

## 4. 三种 schema 形态

**官方文档：**

| 形态 | 返回 | 校验 |
| --- | --- | --- |
| Pydantic | 模型实例 | 运行时校验、默认值、嵌套 |
| Dataclass / TypedDict | dict | 需自己校验 |
| JSON Schema | dict | 须包 Strategy；要有 title、description |

**对应示例：** `02_pydantic_vs_typeddict.py`

**如何选择：** 要校验选 Pydantic；只要 dict 选 TypedDict；schema 来自外部系统选 JSON Schema。

## 5. Custom tool message content

**官方文档：** `ToolStrategy(schema=..., tool_message_content="已写入会议纪要")` 会把这句放进历史，而不是默认的 `Returning structured response: {...}`。适合不想把原始 JSON 再塞回上下文、又想给模型一个成功回执时。

**对应示例：** 无独立文件。API 就是 `ToolStrategy` 的一个参数，见官方 MeetingAction 示例。

## 6. 错误处理

**官方文档：** 用 ToolStrategy 时模型可能出错，框架可用 ToolMessage 反馈并重试。

### 6.1 一次调用了多个结构化工具

Union 场景下模型若同时填了两个 schema，默认会收到错误：「只期望一个 structured response」，然后重试。

### 6.2 Schema 校验失败

例如 rating 要求 1–5，模型给了 10：Pydantic 报错写进 ToolMessage，模型再改到合法值。

### 6.3 handle_errors 策略

| 值 | 行为 |
| --- | --- |
| `True`（默认） | 捕获所有错误，用默认模板重试 |
| `str` | 捕获所有错误，用这句自定义 ToolMessage 重试 |
| `type[Exception]` | 只对该异常重试，其它上抛 |
| `tuple[type[Exception], ...]` | 只对列出的异常重试 |
| `Callable[[Exception], str]` | 自定义函数生成错误文案 |
| `False` | 不重试，异常冒泡 |

**对应示例：** `05_error_handling.py`

**如何选择：** 面向用户的助手打开重试；数据管道关掉，避免悄悄吞掉坏数据。

## 7. Union 多 schema

**官方文档：** `ToolStrategy(Union[ProductReview, CustomerComplaint])` 让模型按上下文选择最合适的一种结构。

**对应示例：** 无独立文件。`02_pydantic_vs_typeddict.py` 演示单 schema；Union 的错误路径见上一节官方说明。

## 8. 与工具组合的实战

先用天气工具拿原始事实，再用 response_format 收成业务对象（出行卡片）。前端可以直接渲染字段。

**对应示例：** `04_weather_structured_report.py`

**限制 / 注意：** 既要调工具又要结构化，取决于模型是否支持同时使用。不行就分成「先 Agent 查天气，再 with_structured_output」。

## 9. 模型层 vs Agent 层

**官方文档：** 本页是 Agent。单独调模型用 `model.with_structured_output`。

**对应示例：** `Models/08_model_structured_output.py`

百炼上模型层同样优先 `method="function_calling"`。

---

## 知识点目录（本目录代码）

```text
01_agent_response_format.py         Agent 结构化出口
02_pydantic_vs_typeddict.py         三种 schema 对比
03_provider_vs_tool_strategy.py     两种策略对比
04_weather_structured_report.py     实战：真实天气 → 出行卡片
05_error_handling.py                校验失败与重试
```

## 学习顺序

```text
response_format 基本用法
↓
选择 schema 类型
↓
选择 Strategy
↓
错误处理 / Union / tool_message_content
↓
和天气工具组合
```

## 常见问题

1. 只看了最后一条 AIMessage，没找 `structured_response`。
2. JSON Schema 没包 Strategy：官方要求必须显式包裹。
3. ProviderStrategy 在兼容接口上报错：改 ToolStrategy。
4. 既要调工具又要结构化：确认模型支持同时使用；不行就分两步。
5. Union 时模型一次填了两个 schema：默认会报错并重试。

## Provider vs Tool

| | ProviderStrategy | ToolStrategy |
| --- | --- | --- |
| 是什么 | 厂商原生约束 | 强制工具参数即 schema |
| 优点 | 更严、更省一轮 | 兼容性好、可重试、支持 Union |
| 何时选 A | OpenAI / Anthropic / Gemini 官方接口 | |
| 何时选 B | 兼容模式、能力不明、需要 handle_errors | |

## 与其他组件的关系

```text
Tools 提供原始事实
    ↓
Models 生成结构化字段
    ↓
Structured output 校验
    ↓
业务代码直接使用对象
```
