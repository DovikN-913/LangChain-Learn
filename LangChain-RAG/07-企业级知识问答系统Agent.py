"""
实战：企业知识库问答（Agent 版，create_agent）

和 Chain 版走同一条入库链路（Loader → 分块 → Milvus），差别在问答：
- Chain：每次必检索，再把片段塞进 Prompt。
- Agent：模型自己决定是否调用检索工具，可多轮追问。

官方文档：
https://docs.langchain.com/oss/python/langchain/agents
https://docs.langchain.com/oss/python/langchain/retrieval

运行：
    uv run python LangChain-RAG/07-企业级知识问答系统Agent.py
    uv run python LangChain-RAG/07-企业级知识问答系统Agent.py --rebuild --demo
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# 工程实践：本文件在 LangChain-RAG/ 子目录，插入项目根才能 import utils / config。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware, ToolRetryMiddleware
from langchain_community.document_loaders import (
    DirectoryLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver
from pymilvus import MilvusClient

import config
from utils import get_chat_model, get_embedding_model, print_stage

# 与 Chain 版共用 enterprise_kb。不要用 04 的 langchain_demo，那里会被 drop_old 清空。
COLLECTION_NAME = "enterprise_kb"
DEFAULT_DOCS_DIR = Path(__file__).resolve().parent / "knowledge_docs"
DEFAULT_LOG_PATH = Path(__file__).resolve().parent / "query_log.json"
MIN_COSINE_SCORE = 0.35
INTERACTIVE_THREAD = "enterprise-kb-chat"

# 官方文档：system_prompt 塑形 Agent「何时调工具、如何回答」。
# 工程实践：事实类问题强制先搜；寒暄放行。否则模型会用训练数据编价格/制度。
SYSTEM_PROMPT = """你是企业知识库助手。

规则：
1. 凡是涉及公司产品、定价、福利、FAQ、机房制度、门禁、运维的事实问题，必须先调用 search_knowledge_base。
2. 只根据工具返回的片段回答，禁止用训练数据补全制度条文或价格。
3. 工具返回「没有足够相关信息」时，明确告诉用户知识库没有，不要编造。
4. 回答里点出来源文件名。
5. 多轮追问时，用户省略主语（例如「那专业版呢」）要结合上文改写成完整检索问句再调用工具。
6. 纯寒暄可以不调工具。"""


def milvus_uri() -> str:
    return f"http://{config.milvus['host']}:{config.milvus['port']}"


def milvus_connection_args() -> dict[str, str]:
    return {"uri": milvus_uri(), "token": config.milvus["token"]}


def source_name(doc: Document) -> str:
    """取出文档来源的文件名，供答案溯源展示。

    逻辑：metadata['source'] 通常是绝对路径，只留文件名。
    为什么：用户要的是「出自哪份制度」，不是本机磁盘路径。
    """
    return Path(str(doc.metadata.get("source", "未知"))).name


def format_docs(docs: list[Document]) -> str:
    """把命中片段收成工具返回值（字符串），交给模型当证据。

    逻辑：编号 + 文件名 + 正文。模型看不到 Document 对象，只能读这段文字。
    为什么必须 return 字符串：create_agent 的工具结果会写成 ToolMessage.content；
    print 只出现在终端，下一轮模型根本看不到。
    """
    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(f"【片段 {i} | {source_name(doc)}】\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def message_text(msg: Any) -> str:
    """从 Agent 最后一条消息里取出给人看的正文。

    逻辑：
    1. 优先 .text（LangChain 1.x 的文本视图）。
    2. 没有再用 .content；若是 content block 列表，降级成 str。

    为什么：str(msg) 会把 id、tool_calls、response_metadata 混进来，看起来不像「答案」。
    """
    text = getattr(msg, "text", None)
    if text:
        return str(text)
    content = getattr(msg, "content", "")
    return content if isinstance(content, str) else str(content)


def load_knowledge_documents(directory: Path) -> list[Document]:
    """入库第一步：按后缀选 Loader，把文件变成 Document。与 01 / 06 同一思想。

    逻辑：txt/md/pdf/docx 分开解析，再拼成一张列表。
    为什么：Agent 和 Chain 必须吃同一批语料，否则两边答案对不上，无法对比架构差异。
    """
    if not directory.is_dir():
        return []
    specs: list[tuple[str, type, dict[str, Any]]] = [
        ("**/*.txt", TextLoader, {"encoding": "utf-8"}),
        ("**/*.md", TextLoader, {"encoding": "utf-8"}),
        ("**/*.pdf", PyPDFLoader, {}),
        ("**/*.docx", Docx2txtLoader, {}),
    ]
    documents: list[Document] = []
    for glob, loader_cls, kwargs in specs:
        loader = DirectoryLoader(
            path=str(directory),
            glob=glob,
            loader_cls=loader_cls,
            loader_kwargs=kwargs,
            silent_errors=True,
        )
        documents.extend(loader.load())
    return documents


def collection_exists(name: str) -> bool:
    """判断 enterprise_kb 是否已建过。连不上当作不存在，后面走重建。

    为什么：避免每次启动都重新 Embedding；也避免和 04 的 langchain_demo 搞混。
    """
    try:
        client = MilvusClient(uri=milvus_uri(), token=config.milvus["token"])
        return bool(client.has_collection(name))
    except Exception:
        return False


class KnowledgeBase:
    """Agentic RAG：入库与 06 相同，问答改成「模型决定是否检索」。

    思想差异（相对 06）：
    - 06 每问必搜，时延稳定，适合 FAQ / 制度。
    - 07 把检索做成工具。模型可以不搜（寒暄、时事）、也可以改写问句再搜（「那专业版呢」）。
    代价：多一轮模型调用，是否检索变得不可完全预期。
    """

    def __init__(
        self,
        docs_directory: Path = DEFAULT_DOCS_DIR,
        *,
        k: int = 3,
        min_score: float = MIN_COSINE_SCORE,
    ) -> None:
        self.docs_directory = Path(docs_directory)
        self.k = k
        self.min_score = min_score
        self.embeddings = get_embedding_model()
        self.model = get_chat_model()
        # 入库参数与 06 保持一致，两个入口才能共用 enterprise_kb。
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        )
        self.vectorstore: Milvus | None = None
        self.agent: Any = None
        # last_hits：工具执行时把 Document 暂存在实例上，query() 再拿去展示来源。
        # 不从 ToolMessage 里解析，是因为展示层要的是对象，工具只能返回字符串。
        self.last_hits: list[tuple[Document, float]] = []
        self.query_log: list[dict[str, Any]] = []

    def _store_kwargs(self, *, drop_old: bool) -> dict[str, Any]:
        """与 06 同一套 Milvus 配置。加载已有库时同样要带 embedding_function。"""
        return {
            "embedding_function": self.embeddings,
            "connection_args": milvus_connection_args(),
            "collection_name": COLLECTION_NAME,
            "consistency_level": "Strong",
            "drop_old": drop_old,
            "auto_id": True,
            "index_params": {
                "metric_type": "COSINE",
                "index_type": "AUTOINDEX",
                "params": {},
            },
            "search_params": {
                "metric_type": "COSINE",
                "params": {},
            },
        }

    def build_index(self) -> bool:
        """离线建库，逻辑与 06 相同：加载 → 分块 → from_documents。

        建完立刻 _build_agent：没有 vectorstore 就装配 Agent，工具调用时会空转。
        """
        print_stage("构建知识库索引")
        docs = load_knowledge_documents(self.docs_directory)
        print(f"加载 {len(docs)} 个 Document，目录: {self.docs_directory}")
        if not docs:
            print("没有可加载的 txt/md/pdf/docx，请把文件放到 knowledge_docs/")
            return False

        split_docs = self.text_splitter.split_documents(docs)
        print(f"分块 {len(split_docs)} 条，开始写入 Milvus {COLLECTION_NAME} ...")
        try:
            # from_documents 的参数是 embedding=，构造函数是 embedding_function=。
            self.vectorstore = Milvus.from_documents(
                documents=split_docs,
                embedding=self.embeddings,
                **{
                    k: v
                    for k, v in self._store_kwargs(drop_old=True).items()
                    if k != "embedding_function"
                },
            )
        except Exception as exc:
            print(f"写入 Milvus 失败: {type(exc).__name__}: {exc}")
            return False
        self._build_agent()
        print("索引构建完成")
        return True

    def load_index(self) -> bool:
        """连已有库，再装配 Agent。问答侧的差异从这里才开始。"""
        print_stage("连接已有知识库")
        try:
            self.vectorstore = Milvus(**self._store_kwargs(drop_old=False))
        except Exception as exc:
            print(f"加载失败: {type(exc).__name__}: {exc}")
            return False
        self._build_agent()
        print(f"已连接 collection={COLLECTION_NAME}")
        return True

    def _build_agent(self) -> None:
        """把「检索」封装成工具，交给 create_agent 循环调用。这是本文件相对 06 的核心。

        逻辑：
        1. 定义 search_knowledge_base(query)：内部仍是向量检索 + 分数阈值。
        2. 嵌套函数闭包 self，这样模型看到的工具参数只有 query，不会出现 self。
        3. create_agent(tools=[...])：模型决定调不调、问句怎么写；框架负责执行并回填 ToolMessage。
        4. checkpointer 按 thread_id 记住 messages，交互里才能追问。

        为什么做成工具而不是 06 那样强制检索：
        官方 Agentic RAG 的要点是「何时检索」由推理决定。寒暄不必搜；
        「那专业版呢」要结合上文改写成完整问句再搜。2-Step 做不到改写。

        工具必须 return 字符串：print 给人看，模型读返回值。
        """
        def search_knowledge_base(query: str) -> str:
            """检索企业知识库中的制度、产品说明和 FAQ。

            用户问机房管理、门禁、公司产品、定价、福利、技术支持等事实时必须调用。
            query 写成完整问句，不要只丢一两个关键词。
            """
            if self.vectorstore is None:
                return "知识库未初始化。"
            scored = self.vectorstore.similarity_search_with_score(query, k=self.k)
            hits = [(doc, score) for doc, score in scored if score >= self.min_score]
            # 展示层要 Document 对象；工具通道只能走字符串，所以来源暂存在实例上。
            self.last_hits = hits
            if not hits:
                return "知识库中没有足够相关的信息。"
            return format_docs([doc for doc, _ in hits])

        self.agent = create_agent(
            model=self.model,
            tools=[search_knowledge_base],
            checkpointer=InMemorySaver(),
            middleware=[
                ModelRetryMiddleware(max_retries=2),
                ToolRetryMiddleware(max_retries=2),
            ],
            system_prompt=SYSTEM_PROMPT,
            name="enterprise_kb_agent",
        )

    def query(self, question: str, *, thread_id: str = "demo") -> dict[str, Any]:
        """把用户问题交给 Agent 循环，而不是自己先检索。

        逻辑：
        1. 清空 last_hits，避免本轮没调工具时还展示上一轮来源。
        2. invoke(messages, config)：config.thread_id 决定是否续上历史。
        3. 最后一条消息是最终回答；过程中若出现 type==tool，说明检索发生过。
        4. 来源从 last_hits 取（工具执行时写入），不是从答案文本里正则抠。

        为什么示例用 demo-0、demo-1：五道题互不干扰，避免「股市」被上一题机房上下文带偏。
        为什么交互用固定 INTERACTIVE_THREAD：同一会话才能理解「那专业版呢」。
        InMemorySaver 只活在本进程，重启即丢，和 Short-term memory 目录结论相同。
        """
        if self.agent is None:
            return {"error": "知识库未初始化", "question": question}

        self.last_hits = []
        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            {"configurable": {"thread_id": thread_id}},
        )
        messages = result.get("messages") or []
        answer = message_text(messages[-1]) if messages else ""
        tool_called = any(getattr(msg, "type", "") == "tool" for msg in messages)
        payload = {
            "question": question,
            "answer": answer,
            "sources": [doc for doc, _ in self.last_hits],
            "scores": [float(score) for _, score in self.last_hits],
            "tool_called": tool_called,
        }
        self._log(payload)
        return payload

    def _log(self, result: dict[str, Any]) -> None:
        """记录问题、答案、是否调了工具、来源文件名。对比 Chain 日志时重点看 tool_called。"""
        self.query_log.append(
            {
                "question": result["question"],
                "answer": result["answer"],
                "sources": [source_name(doc) for doc in result.get("sources", [])],
                "scores": result.get("scores", []),
                "tool_called": result.get("tool_called"),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
        )

    def export_logs(self, path: Path = DEFAULT_LOG_PATH) -> None:
        path = Path(path)
        path.write_text(
            json.dumps(self.query_log, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"查询日志已导出: {path}")


def print_result(result: dict[str, Any]) -> None:
    """先打印「本轮有没有检索」，再打印答案和来源。这是观察 Agent 决策的入口。"""
    if result.get("error"):
        print(f"错误: {result['error']}")
        return
    if result.get("tool_called"):
        print("本轮调用了 search_knowledge_base")
    else:
        print("本轮未调用检索工具")
    print(f"答案:\n{result['answer']}")
    sources: list[Document] = result.get("sources") or []
    scores: list[float] = result.get("scores") or []
    if not sources:
        return
    print("参考来源:")
    for i, doc in enumerate(sources):
        score = scores[i] if i < len(scores) else None
        score_text = f"  score={score:.4f}" if score is not None else ""
        preview = doc.page_content.replace("\n", " ")[:80]
        print(f"  {i + 1}. {source_name(doc)}{score_text}")
        print(f"     {preview}...")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="企业知识库问答（Agent）")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="删除并重建 Milvus collection",
    )
    parser.add_argument("--demo", action="store_true", help="只跑示例问题")
    parser.add_argument("--k", type=int, default=3, help="检索返回块数")
    parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS_DIR)
    return parser.parse_args()


def main() -> None:
    """启动：复用或重建知识库 → 每道示例题独立 thread → TTY 下进入可追问的交互。"""
    args = parse_args()
    print_stage("企业知识库问答（Agent）")
    print(f"文档目录: {args.docs}")
    print(f"Milvus: {milvus_uri()}  collection={COLLECTION_NAME}")

    kb = KnowledgeBase(docs_directory=args.docs, k=args.k)
    exists = collection_exists(COLLECTION_NAME)
    if args.rebuild or not exists:
        if exists and args.rebuild:
            print("按 --rebuild 重建索引，旧向量会被删除。")
        if not kb.build_index():
            print("构建失败，退出。")
            return
    elif not kb.load_index():
        print("加载失败，尝试重建...")
        if not kb.build_index():
            print("构建失败，退出。")
            return

    demo_questions = [
        "外人能不能进机房？",
        "公司主要产品有哪些？",
        "AI 开发平台的定价是多少？",
        "如何联系技术支持？",
        "今天股市怎么样？",
    ]
    print_stage("示例查询（每问一条独立 thread，互不干扰）")
    for i, question in enumerate(demo_questions):
        print(f"\n问题: {question}")
        print_result(kb.query(question, thread_id=f"demo-{i}"))
        print("-" * 60)

    if args.demo or not sys.stdin.isatty():
        kb.export_logs()
        return

    print_stage("交互模式（同一 thread，可追问）")
    print("命令: quit 退出 | export 导出日志")
    while True:
        try:
            question = input("\n请输入问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            kb.export_logs()
            print("已退出。")
            return

        lowered = question.lower()
        if lowered in {"quit", "exit", "q"}:
            kb.export_logs()
            print("已退出。")
            return
        if lowered == "export":
            kb.export_logs()
            continue
        if not question:
            continue
        print_result(kb.query(question, thread_id=INTERACTIVE_THREAD))


if __name__ == "__main__":
    main()
