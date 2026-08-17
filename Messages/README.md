# Messages

官方文档：https://docs.langchain.com/oss/python/langchain/messages

Messages 是模型上下文的基本单位。输入、输出、工具结果、系统提示都以消息对象流动。

## 这个组件是什么

一条消息包含：

- **Role**：谁在说话（system / human / ai / tool）
- **Content**：文本、图片、推理块等
- **Metadata**：id、token 用量、厂商附加字段

LangChain 提供跨厂商的标准消息类型，保证换模型时行为一致。

## 为什么要设计它

不同厂商的聊天协议细节不同。统一消息类型后，多轮对话、工具调用、流式输出能共用同一套状态。

---

# 官方知识点讲解

## 1. 基本用法：三种输入

**官方文档：**

| 形态 | 何时用 |
| --- | --- |
| 字符串 | 单轮、无历史、代码最少。内部等于一条 HumanMessage |
| 消息对象列表 | 多轮、多模态、要带 SystemMessage |
| OpenAI 风格 dict | `{"role","content"}`，方便对接已有 OpenAI 代码 |

**对应示例：** `02_text_vs_messages_vs_dict.py`

## 2. SystemMessage

**官方文档：** 初始指令，用来定人设、语气、规则。可以很短，也可以写详细 persona。

**对应示例：** `01_message_types.py`、`06_multiturn_conversation.py`

**限制 / 注意：** 有的厂商会弱化 system 权重，或要求历史以 user 开头。删历史时不要把对话裁成非法序列。

## 3. HumanMessage

**官方文档：** 用户输入。可以是纯文本，也可以带图/音/文件。可选 `name`（区分用户）、`id`（追踪）。`name` 是否生效因厂商而异。

**对应示例：** `01_message_types.py`

**限制 / 注意：** `model.invoke("...")` 就是单条 HumanMessage 的快捷写法。

## 4. AIMessage

**官方文档：** 模型输出。常见字段：

| 字段 | 含义 |
| --- | --- |
| `.content` | 文本视图 |
| `.content` | 原始 payload（str 或 list） |
| `.content_blocks` | 跨厂商标准化块 |
| `.tool_calls` | 工具调用；没有则为空 |
| `.id` | 消息 id |
| `.usage_metadata` | token 用量 |
| `.response_metadata` | 厂商原始元数据 |

可以手动构造 AIMessage 插入历史，假装「模型曾经这么说过」。

**对应示例：** `03_ai_message_metadata.py`；tool_calls 见 `04_tool_messages.py`

## 5. Token usage

**官方文档：** `usage_metadata` 典型结构：`input_tokens` / `output_tokens` / `total_tokens`，以及 audio、cache_read、reasoning 等明细。

**对应示例：** `03_ai_message_metadata.py`、`Models/06_model_parameters.py`

**限制 / 注意：** 不是所有厂商每次都返回完整明细。流式场景有的接口要额外 opt-in。

## 6. Streaming 与 AIMessageChunk

**官方文档：** `stream()` 产出 `AIMessageChunk`，可相加成完整 AIMessage，之后可当普通消息放进历史。

**对应示例：** `Models/03_model_stream.py`

## 7. ToolMessage

**官方文档：** 把单次工具执行结果交回模型。关键字段：

- `content`：给模型看的文本
- `tool_call_id`：**必须**匹配对应 tool call 的 id
- `name`：工具名
- `artifact`：给程序看、不塞进模型上下文的附加数据（如 document_id、原始 JSON）

**对应示例：** `04_tool_messages.py`（真实天气；content 用摘要，artifact 存原始 JSON）

**限制 / 注意：** id 对不上，多数厂商会报错或忽略结果。

## 8. Message content

**官方文档：** `content` 是松散类型：字符串、厂商原生块列表、或 LangChain 标准块。这是为了兼容多模态和厂商私有结构。

指定 `content_blocks` 初始化时仍会填充 `content`，但类型更安全。

**对应示例：** `05_content_blocks.py`

**限制 / 注意：** 业务展示优先用 `.content`。不要假设 `content` 永远是 str。

## 9. Standard content blocks

**官方文档：** `content_blocks` 会把 Anthropic `thinking`、OpenAI `reasoning` 等懒解析成统一的 `reasoning` / `text` 等类型。它**不是**替换 `content`，而是标准化视图。

若应用需要把标准块写进 `content` 以便对外序列化：设环境变量 `LC_OUTPUT_VERSION=v1`，或 `init_chat_model(..., output_version="v1")`。

**对应示例：** `05_content_blocks.py`

常见块类型（官方参考）：

| type | 用途 |
| --- | --- |
| `text` | 普通文本，可带 annotations |
| `reasoning` | 思考过程 |
| `image` / `audio` / `video` / `file` | 多模态 |
| `tool_call` / `tool_call_chunk` | 客户端工具调用 |
| `server_tool_call` / `server_tool_result` | 厂商服务端工具 |

## 10. Multimodal

**官方文档：** 标准块可用 url、base64+mime_type、或厂商 file_id。图片、PDF、音频、视频写法类似。额外字段可放顶层或 `extras`。OpenAI 对 PDF 可能要求 filename。

**对应示例：** `05_content_blocks.py`（结构演示，不真正拉图）

**限制 / 注意：** 不是所有模型支持所有类型。当前教学模型是文本模型。

## 11. Serialization

**官方文档：** `dumpd(message)` 序列化成普通 dict，`load(serialized)` 还原。用于持久化会话。

**对应示例：** 无独立文件。持久化会话的产品做法见 `Short-term memory/` 的 checkpointer。

**限制 / 注意（官方强调）：** `load()` 会实例化 Python 对象，可能有副作用。**不要对不信任来源调用 `load()`。**

## 12. 与 Chat Model 一起用

**官方文档：** Chat Model 输入消息序列，输出 AIMessage。交互默认无状态，多轮就是不断变长的 list。更完整的历史管理见 Short-term memory（持久化、裁剪、摘要）。

**对应示例：** `06_multiturn_conversation.py`

---

## 知识点目录（本目录代码）

```text
01_message_types.py              四种核心消息
02_text_vs_messages_vs_dict.py   三种输入形态对比
03_ai_message_metadata.py        usage 与 response_metadata
04_tool_messages.py              ToolMessage + 真实天气
05_content_blocks.py             content / 多模态 / 标准块
06_multiturn_conversation.py     手动维护多轮历史
```

## 学习顺序

```text
消息类型
↓
三种输入写法
↓
AIMessage 元数据
↓
ToolMessage
↓
content_blocks / 多模态
↓
多轮历史 → Short-term memory
```

## 常见问题

1. **ToolMessage 对不上**：`tool_call_id` 必须等于对应 tool call 的 `id`。
2. **模型「忘了」刚才说的话**：没有把历史再传回去。
3. **content 有时是 str 有时是 list**：合法的松散类型，展示用 `.content`。
4. **不要 `load()` 来路不明的序列化消息**：官方安全警告。

## 与其他组件的关系

```text
Messages
   ↓ 作为输入
Models.invoke / stream
   ↓ 产出 AIMessage
Tools 执行后写回 ToolMessage
   ↓ 累积
Short-term memory（checkpointer）
```
