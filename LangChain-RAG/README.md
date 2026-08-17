# LangChain-RAG

官方文档：
- [Retrieval](https://docs.langchain.com/oss/python/langchain/retrieval)
- [Document loaders](https://docs.langchain.com/oss/python/integrations/document_loaders)
- [Text splitters](https://docs.langchain.com/oss/python/integrations/splitters)
- [OpenAI Embeddings（兼容接口同样适用）](https://docs.langchain.com/oss/python/integrations/embeddings/openai)
- [Milvus Vector Store](https://docs.langchain.com/oss/python/integrations/vectorstores/milvus)

模型有两个硬限制：**上下文装不下整库**，**训练知识会过时**。Retrieval 在提问时把相关外部知识取回来，这就是 RAG 的底座。

本目录把这条流水线拆成可运行单元，再用两种官方架构做成企业知识问答：

```text
文件源
  → Document Loader
  → Text Splitter
  → Embedding
  → Vector Store（Milvus）
  → Retriever
  → Chat Model 生成答案
```

Embedding 走阿里百炼 `text-embedding-v1`（`utils.get_embedding_model()`）。向量库走 `config.py` 里的 Milvus。文档样本在 `knowledge_docs/`。

## 这个组件是什么

RAG = **检索增强生成**：先从自己的知识库取出相关片段，再让模型基于片段回答，而不是靠参数里记住的世界知识。

LangChain 把入库和问答拆成可替换积木：换 PDF 解析器、换分块策略、换向量库，不必重写问答逻辑。

## 为什么要设计它

没有检索时，模型会编造制度条文和价格。有了检索，答案可以溯源到 `机房安全与出入环境管理制度.txt` 或 `faq.txt`。

官方把落地形态分成三种：

| 架构 | 行为 | 本目录 |
| --- | --- | --- |
| **2-Step RAG** | 每次必先检索再生成 | `05`、`06` |
| **Agentic RAG** | 模型决定何时、如何检索 | `07` |
| **Hybrid RAG** | 查询改写 + 检索/答案校验循环 | 无独立脚本，见下方说明 |

---

# 官方知识点讲解

## 1. Document Loader

**官方文档：** Loader 把 Google Drive / PDF / 本地文件等读成统一的 `Document`（`page_content` + `metadata`）。共同接口是 `load()` 和 `lazy_load()`。

**对应示例：** `01-加载不同格式文档.py`

**限制 / 注意：**

- 格式不同，Loader 不同。`TextLoader` 读不了 PDF。
- `langchain-community` 正在 sunset，现阶段这些 Loader 仍是常用入口。
- 扫描版 PDF 可能抽出空字符串，不会报错。
- 当前 `knowledge_docs/` 主要是 txt / md；pdf / docx 的 Loader 已写好，把文件放进去即可。

## 2. Text Splitter

**官方文档：** 长文档要切成可检索、能塞进上下文的小块。通用文本的推荐起点是 `RecursiveCharacterTextSplitter`：按分隔符从粗到细递归切。中日文等没有空格的文字，要自己加入 `。！？，`。

**对应示例：** `02-文本分块.py`

**限制 / 注意：**

- `split_text` 返回 `list[str]`，丢掉来源。入库用 `split_documents`，才能把 `source` 带到每一块。
- `chunk_size` 在 Recursive/Character 里是**字符**；`TokenTextSplitter` 才是 token。
- overlap 通常取 chunk_size 的 10%～20%。

## 3. Embedding

**官方文档：** Embedding 把文本变成固定维向量。向量库建索引用 `embed_documents`，查询用 `embed_query`。两者必须同一模型。

**对应示例：** `03-文本Embedding.py`

**限制 / 注意：**

- 百炼走 OpenAI 兼容接口，用 `OpenAIEmbeddings` + `base_url`，不是 `ChatOpenAI`。
- 必须关闭 `check_embedding_ctx_length`：默认按 OpenAI tiktoken 检查，百炼会误报。
- `text-embedding-v1` 单次 batch 上限 **25**。`utils.get_embedding_model()` 已把 `chunk_size` 设为 25，否则 `from_documents` 会 400。
- 维数由模型决定（本项目 1536）。换模型后旧向量全部作废。

## 4. Vector Store / Retriever

**官方文档：** Vector Store 存向量并做近邻搜索。Retriever 是「给一句查询，返回 Document 列表」的接口。`VectorStore.as_retriever()` 是最常见接法。

**对应示例：** `04-milvus向量数据库示例.py`

**限制 / 注意：**

- 连接信息来自 `config.py` 的 `milvus`，不要把主机写进每个示例。
- `04` 使用 collection `langchain_demo`，且 `drop_old=True`，每次运行会清空。只适合教学。
- 加载已有库时也必须传 `embedding_function`，否则问题无法向量化。
- `index_params` 与 `search_params` 的 `metric_type` 必须一致。本目录用 **COSINE**，分数越大越像。
- 元数据过滤用 Milvus `expr`，不要只靠 Python 里事后 `if`。

## 5. 2-Step RAG（LCEL 最短链）

**官方文档：** 检索永远发生在生成之前。控制力强、时延可预期，适合 FAQ / 制度问答。

**对应示例：** `05-构建完整的RAG系统1.py`

**限制 / 注意：** 本文件**不入库**。先跑 `04`，再问「机房门禁怎么管？」。它演示链怎么接，不负责阈值、溯源、交互。

## 6. 2-Step RAG（企业知识库）

**官方文档：** 用自己的语料建知识库，再跑 retrieve-then-generate。

**对应示例：** `06-企业级知识问答系统Chain.py`

会加载 `knowledge_docs/` 下 txt/md/pdf/docx，写入独立 collection `enterprise_kb`。检索带余弦阈值（默认 0.35），低于阈值不调用模型。

## 7. Agentic RAG

**官方文档：** Agent 只需要能取外部知识的 **tool**。模型在推理过程中决定何时检索、问句怎么写；多轮可以改写查询再搜。

**对应示例：** `07-企业级知识问答系统Agent.py`

用 `create_agent` + `search_knowledge_base` 工具，入库与 `06` 相同（`enterprise_kb`）。交互模式带 `InMemorySaver`，可以追问「那专业版呢」。

## 8. Hybrid RAG（本目录没有独立脚本）

**官方文档：** 在 2-Step 与 Agent 之间加查询改写、检索是否充分、答案是否 grounded 等校验，不够就重写查询再搜。

**对应示例：** 无。`06` 的分数阈值只是最简「检索校验」；完整 Hybrid 还要 query rewrite、二次检索、答案校验，适合单独做评测项目，不塞进教学主路径。

进阶（BM25 混合检索、partition key 多租户、Deep Agents 文档问答）见官方 Retrieval 页，本目录不展开。

---

# 知识点目录

| 文件 | 学习什么 |
| --- | --- |
| `01-加载不同格式文档.py` | Loader → `Document`；按后缀选解析器 |
| `02-文本分块.py` | Recursive / Character / Token；`split_text` vs `split_documents`；中文分隔符 |
| `03-文本Embedding.py` | `embed_query` / `embed_documents`；余弦相似度 |
| `04-milvus向量数据库示例.py` | 写入、搜索、带分数、`expr`、`as_retriever` |
| `05-构建完整的RAG系统1.py` | 最短 2-Step LCEL 链（依赖 04 的数据） |
| `06-企业级知识问答系统Chain.py` | 完整 2-Step：入库 + 阈值 + 溯源 + 交互 |
| `07-企业级知识问答系统Agent.py` | Agentic RAG：`create_agent` 决定是否检索 |
| `knowledge_docs/` | 机房制度 + 公司产品/FAQ 样本 |

---

# 学习顺序

```text
Loader（01）
    ↓
分块（02）
    ↓
Embedding（03）
    ↓
Vector Store（04）
    ↓
最短 2-Step 链（05）
    ↓
企业知识库 Chain（06）
    ↓
企业知识库 Agent（07）
```

建议先跑通 01→04，再对比 05/06/07 三种「检索之后怎么生成」。

---

# 运行

在项目根目录：

```text
uv sync
uv run python LangChain-RAG/01-加载不同格式文档.py
uv run python LangChain-RAG/02-文本分块.py
uv run python LangChain-RAG/03-文本Embedding.py
uv run python LangChain-RAG/04-milvus向量数据库示例.py
uv run python LangChain-RAG/05-构建完整的RAG系统1.py
uv run python LangChain-RAG/06-企业级知识问答系统Chain.py --rebuild --demo
uv run python LangChain-RAG/07-企业级知识问答系统Agent.py --demo
```

`06` / `07` 不加 `--rebuild` 时，若 `enterprise_kb` 已存在则直接连接。

`.env` 需要 `DASHSCOPE_API_KEY` / `DASHSCOPE_API_URL` / `DASHSCOPE_CHAT_MODEL_NAME` / `DASHSCOPE_EMBEDDING_MODEL_NAME`。  
Milvus 主机和 token 在 `config.py`。

---

# 常见问题

1. **跑 06/07 报 batch size > 25：** 百炼 embedding 单次最多 25 条。应走 `get_embedding_model()`（已设 `chunk_size=25`），不要自己 `OpenAIEmbeddings()` 用默认 1000。
2. **跑 05 搜不到 / 答「不知道」：** 05 读的是 `langchain_demo`。先跑 04。不要和 06 的 `enterprise_kb` 搞混。
3. **跑完 04 再跑 06，知识库空了：** 以前两者曾共用 `langchain_demo`。现在 04 用 `langchain_demo`（可清空），06/07 用 `enterprise_kb`。
4. **加载已有 Milvus 库检索失败：** 构造 `Milvus(...)` 时漏了 `embedding_function`。
5. **中文被从词中间切开：** 默认 separators 偏英文。见 02，加上 `。！？，`。
6. **COSINE 分数越大越像，不要拿 L2 的「越小越好」来设阈值。**
7. **`drop_old=True` 只适合 04 这种演示。** 生产重建要用显式 `--rebuild`。
8. **Agent 版「今天股市怎么样？」可能不调工具：** 这是 Agentic RAG 的预期行为；Chain 版每次都会搜。
9. **Windows 下 `from utils import` 失败：** 必须从项目根 `uv run python LangChain-RAG/xx.py`，文件里已插入 `sys.path`。
10. **密钥不要写进示例。** Chat / Embedding 走 `.env`，Milvus 走 `config.py`。

---

# 2-Step vs Agent

| | 06 Chain（2-Step） | 07 Agent（Agentic） |
| --- | --- | --- |
| 是什么 | 每问必检索，再生成 | 模型决定是否调用检索工具 |
| 优点 | 可预期、易溯源、时延稳定 | 能改写问句、可多轮、无关问题可不搜 |
| 何时选 A | FAQ、制度、必须引用原文 | |
| 何时选 B | 助手还要调天气等其它工具、允许追问 | |

---

# 与其他 LangChain 组件的关系

```text
Models（Chat Model + Embedding）
    ↓
Messages / Prompt（把片段和问题拼进上下文）
    ↓
Document Loader → Splitter → Vector Store → Retriever
    ↓
    ├─ 2-Step：LCEL 链 / 06
    └─ Agentic：Tools + create_agent / 07
```

入库用 Embedding；生成答案用 Chat Model。两者都是模型，接口完全不同，不要把 `qwen3.6-35b-a3b` 传给 `OpenAIEmbeddings`。
