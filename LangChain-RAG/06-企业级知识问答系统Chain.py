"""
实战：企业知识库问答（Loader → 分块 → Milvus → 检索生成）

把本目录 01～04 串成可交互系统：
- 多格式文档入库（txt / md / pdf / docx）
- 中文分块后写入独立 Milvus collection
- 按余弦相似度检索，低于阈值不调用模型
- 基于命中片段生成答案，并回过来源文件

官方文档：
https://docs.langchain.com/oss/python/integrations/vectorstores/milvus
https://docs.langchain.com/oss/python/langchain/retrieval

运行：
    uv run python LangChain-RAG/06-企业级知识问答系统Chain.py
    uv run python LangChain-RAG/06-企业级知识问答系统Chain.py --rebuild --demo
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

from langchain_community.document_loaders import (
    DirectoryLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_milvus import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import MilvusClient

import config
from utils import get_chat_model, get_embedding_model, print_stage

# 不要复用 04 的 langchain_demo：那个 collection 每次演示都会 drop_old 清空。
COLLECTION_NAME = "enterprise_kb"
DEFAULT_DOCS_DIR = Path(__file__).resolve().parent / "knowledge_docs"
DEFAULT_LOG_PATH = Path(__file__).resolve().parent / "query_log.json"
# 03 里无关句大约 0.13、相关句 0.64+。0.35 用来挡住明显跑题的块。
MIN_COSINE_SCORE = 0.35

# 官方文档：2-Step 把检索结果当成 Prompt 的 context，而不是让模型自由发挥。
# 工程实践：Prompt 在模块级建一次，不要在 query() 里每次 new，避免重复解析模板。
ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """你是企业知识库问答助手。只根据「文档片段」回答，禁止用片段以外的知识补全。

文档片段：
{context}

用户问题：{question}

要求：
1. 片段里没有答案时，明确说「知识库中没有相关信息」，不要编造。
2. 多段都相关时综合回答，并指出分别来自哪类制度/文档。
3. 回答准确、完整、用中文。"""
)


def milvus_uri() -> str:
    return f"http://{config.milvus['host']}:{config.milvus['port']}"


def milvus_connection_args() -> dict[str, str]:
    return {"uri": milvus_uri(), "token": config.milvus["token"]}


def source_name(doc: Document) -> str:
    """取出文档来源的文件名，供答案溯源展示。

    逻辑：metadata['source'] 通常是绝对路径，只留文件名，避免把本机目录打给用户。
    为什么：RAG 的可信度来自「这句话出自哪份制度」，不是来自模型自信程度。
    """
    return Path(str(doc.metadata.get("source", "未知"))).name


def format_docs(docs: list[Document]) -> str:
    """把检索到的 Document 列表收成一段给模型看的上下文。

    逻辑：按顺序编号，每块带上文件名 + 正文，块与块之间用分隔线切开。
    为什么：模型没有「这条是第几段」的结构感；把来源写进文本里，它才能在答案里引用文件名。
    只传 page_content、不传来源，模型就只能说「根据资料」，无法溯源。
    """
    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(f"【片段 {i} | {source_name(doc)}】\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def load_knowledge_documents(directory: Path) -> list[Document]:
    """入库第一步：把目录里的原始文件读成统一的 Document 列表。

    逻辑：
    1. 按后缀分别用 TextLoader / PyPDFLoader / Docx2txtLoader，再把结果拼起来。
    2. silent_errors=True：单个坏文件跳过，不让整批入库失败。

    为什么：
    不同格式的解析器不同（见 01）。一个 Unstructured 吃所有文件，失败点和抽字质量都不可控。
    先变成 Document，后面的分块 / Embedding 才不用关心「这原来是 PDF 还是 txt」。
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
    """判断知识库是否已经建过，决定走「重建」还是「直接连」。

    逻辑：连上 Milvus，问有没有这个 collection；网络失败当作不存在，后面会走 build_index。
    为什么：Embedding 要花钱。库已经在就不要每次启动都重新 embed 一遍。
    """
    try:
        client = MilvusClient(uri=milvus_uri(), token=config.milvus["token"])
        return bool(client.has_collection(name))
    except Exception:
        return False


class KnowledgeBase:
    """2-Step RAG 的完整闭环：入库一次，之后每问都是「检索 →（可选）生成」。

    思想：模型自己不知道你们公司的制度。可靠答案只能来自检索到的片段。
    所以生成模型被限制在 context 里；检索不到就拒绝回答，而不是让它补全。
    """

    def __init__(
        self,
        docs_directory: Path = DEFAULT_DOCS_DIR,
        *,
        k: int = 3,
        min_score: float = MIN_COSINE_SCORE,
    ) -> None:
        # k：每次召回几块。太大噪声高、费 token；太小容易漏关键条款。
        # min_score：余弦下限。低于它视为跑题，宁可不答。
        self.docs_directory = Path(docs_directory)
        self.k = k
        self.min_score = min_score
        self.embeddings = get_embedding_model()
        self.model = get_chat_model()
        # 中文 separators 见 02；overlap 取 chunk_size 的 10%，避免一句话被切在两块边界上。
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        )
        # 生成链故意不含 Retriever。
        # 逻辑上检索必须先发生、先过滤，调用方还要留着 Document 做溯源。
        # 05 把 retriever 接进 LCEL 后，invoke 只剩 str，来源就丢了。
        self.answer_chain = ANSWER_PROMPT | self.model | StrOutputParser()
        self.vectorstore: Milvus | None = None
        self.query_log: list[dict[str, Any]] = []

    def _store_kwargs(self, *, drop_old: bool) -> dict[str, Any]:
        """Milvus 连接与索引的公共参数，build / load 共用，避免两套配置漂开。

        逻辑：COSINE + AUTOINDEX；drop_old 只在重建时为 True。
        为什么：index 和 search 的度量必须相同，否则分数不可比。
        加载已有库也要带 embedding_function，否则问题无法变成同一空间里的向量。
        """
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
        """离线建库：文件 → 分块 → 向量 → 写入 Milvus。问答前只需做一次。

        逻辑：
        1. load_knowledge_documents：格式差异在这一步消掉。
        2. split_documents：切小块并复制 source，检索粒度才够用（见 02）。
        3. from_documents：内部 embed_documents + insert；drop_old=True 先删旧表。

        为什么要分块再 embed：整篇制度变成一条向量，关键句会被平均掉，问「外人进机房」对不上。
        为什么不每次提问都建库：Embedding 按条计费，语料没变就该复用 collection。
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
            # 官方文档：from_documents = 建表 + embed_documents + insert。
            # 参数名是 embedding=，和构造函数的 embedding_function= 不是同一个关键字。
            # drop_old=True 会删掉同名 collection，所以重建必须显式 --rebuild。
            self.vectorstore = Milvus.from_documents(
                documents=split_docs,
                embedding=self.embeddings,
                **{k: v for k, v in self._store_kwargs(drop_old=True).items() if k != "embedding_function"},
            )
        except Exception as exc:
            print(f"写入 Milvus 失败: {type(exc).__name__}: {exc}")
            return False
        print("索引构建完成")
        return True

    def load_index(self) -> bool:
        """连上已经建好的 collection，跳过重新 Embedding。

        逻辑：drop_old=False，用同一套 embedding 模型去 embed 问题。
        为什么：换模型后向量空间对不上，旧库必须重建；同一模型则可直接搜。
        """
        print_stage("连接已有知识库")
        try:
            self.vectorstore = Milvus(**self._store_kwargs(drop_old=False))
        except Exception as exc:
            print(f"加载失败: {type(exc).__name__}: {exc}")
            return False
        print(f"已连接 collection={COLLECTION_NAME}")
        return True

    def query(self, question: str, *, k: int | None = None) -> dict[str, Any]:
        """在线问答：2-Step RAG 的核心。先检索，再决定是否生成。

        逻辑：
        1. similarity_search_with_score：把问题 embed_query，在库里取 Top-K 及余弦分数。
        2. 丢掉低于 min_score 的块——它们只是「相对最像」，不一定真相关。
        3. 过滤后为空：直接返回拒答，不调用 Chat Model（省 token，也堵住幻觉）。
        4. 有命中：把片段格式化进 Prompt，模型只根据 context 写答案。
        5. 把 Document 和分数一并返回，界面才能做溯源。

        为什么检索和生成要拆开：
        05 把 Retriever 藏进 LCEL 后，外面拿不到来源，也无法做阈值。
        为什么先过滤再生成：把跑题块塞给模型，它仍会「综合」出像那么回事的假制度。
        """
        if self.vectorstore is None:
            return {"error": "知识库未初始化", "question": question}

        top_k = k if k is not None else self.k
        scored = self.vectorstore.similarity_search_with_score(question, k=top_k)
        hits = [(doc, score) for doc, score in scored if score >= self.min_score]

        if not hits:
            answer = "知识库中没有足够相关的信息，无法基于现有文档回答。"
            result = {
                "question": question,
                "answer": answer,
                "sources": [],
                "scores": [float(score) for _, score in scored],
            }
            self._log(result)
            return result

        docs = [doc for doc, _ in hits]
        answer = self.answer_chain.invoke(
            {"context": format_docs(docs), "question": question}
        )
        result = {
            "question": question,
            "answer": answer,
            "sources": docs,
            "scores": [float(score) for _, score in hits],
        }
        self._log(result)
        return result

    def _log(self, result: dict[str, Any]) -> None:
        """记下本轮问题、答案、来源文件名。只留文件名，避免把本机路径写进日志。"""
        self.query_log.append(
            {
                "question": result["question"],
                "answer": result["answer"],
                "sources": [source_name(doc) for doc in result.get("sources", [])],
                "scores": result.get("scores", []),
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
    """把答案和来源打到终端。学习时先看来源是否对，再看答案写得对不对。"""
    if result.get("error"):
        print(f"错误: {result['error']}")
        return
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
    parser = argparse.ArgumentParser(description="企业知识库问答")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="删除并重建 Milvus collection（会丢掉该库已有向量）",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="只跑内置示例问题，不进入交互",
    )
    parser.add_argument("--k", type=int, default=3, help="每次检索返回的块数")
    parser.add_argument(
        "--docs",
        type=Path,
        default=DEFAULT_DOCS_DIR,
        help="知识库文档目录",
    )
    return parser.parse_args()


def main() -> None:
    """启动流程：有库则连，无库或 --rebuild 则建；然后跑示例，TTY 下再进交互。

    逻辑：collection 已存在且未指定 --rebuild → load_index；否则 build_index。
    为什么默认不重建：重复 embed 同一批文档浪费额度，也拖慢启动。
    """
    args = parse_args()
    print_stage("企业知识库问答")
    print(f"文档目录: {args.docs}")
    print(f"Milvus: {milvus_uri()}  collection={COLLECTION_NAME}")

    kb = KnowledgeBase(docs_directory=args.docs, k=args.k)
    exists = collection_exists(COLLECTION_NAME)

    # 默认复用已有库。只有 --rebuild 或 collection 不存在才重新 embed，省百炼额度。
    if args.rebuild or not exists:
        if exists and args.rebuild:
            print("按 --rebuild 重建索引，旧向量会被删除。")
        if not kb.build_index():
            print("构建失败，退出。")
            return
    else:
        if not kb.load_index():
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
    print_stage("示例查询")
    for question in demo_questions:
        print(f"\n问题: {question}")
        print_result(kb.query(question))
        print("-" * 60)

    if args.demo or not sys.stdin.isatty():
        kb.export_logs()
        return

    print_stage("交互模式")
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
        print_result(kb.query(question))


if __name__ == "__main__":
    main()
