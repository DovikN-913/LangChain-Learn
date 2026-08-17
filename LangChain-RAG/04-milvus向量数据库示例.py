"""
知识点：Vector Store 把「文本 + 向量 + 元数据」存起来，并做相似度检索

官方文档：
https://docs.langchain.com/oss/python/integrations/vectorstores/milvus

本示例主要学习：
1. Embedding 只负责「文本 → 向量」；Vector Store 负责存、建索引、按向量找近邻。
2. add_documents 内部会调用 embed_documents，再把向量写入 Milvus collection。
3. similarity_search 内部会 embed_query(问题)，再在索引里找最接近的 k 条。
4. metadata 会一并写入，检索时可用 expr 过滤（例如只要机房制度、不要天气闲聊）。
5. as_retriever() 把 Vector Store 变成 RAG 链里的检索器。

连接信息来自项目根 config.py 的 milvus 字典，不要把主机写死在示例逻辑里。
本示例 drop_old=True，每次运行会删掉同名 collection，只适合教学。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 工程实践：本文件在 LangChain-RAG/ 子目录，插入项目根才能 import utils / config。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document
from langchain_milvus import Milvus

import config
from utils import get_embedding_model, print_stage


def milvus_uri() -> str:
    """把 host/port 拼成 langchain-milvus 要的 uri。

    为什么不写死在示例里：换环境只改 config.py；Lite 用本地 .db 路径，Server 用 http://host:port。
    """
    host = config.milvus["host"]
    port = config.milvus["port"]
    return f"http://{host}:{port}"


def main() -> None:
    print_stage("0. Embedding + Milvus 连接")
    # 官方文档：Milvus(...) 需要一个实现了 embed_query / embed_documents 的 Embeddings。
    # 工程实践：这里必须和以后查询时用同一 get_embedding_model()，否则向量空间对不上。
    embeddings = get_embedding_model()
    uri = milvus_uri()
    print(f"Milvus URI: {uri}")
    print(f"Embedding 模型: {embeddings.model}")

    # 官方文档：connection_args 的 uri 可以是本地 milvus.db（Lite）或 http://host:19530（Server）。
    # 源码分析：构造时就会连服务器、按 collection_name 取或建表；embedding 维数用来定义向量字段。
    # drop_old=True：若已有同名 collection 先删再建。教学方便，生产会丢掉全部已入库向量。
    # auto_id=True：主键由 Milvus 生成。若要可更新/可删除指定文档，应自己传 ids。
    # consistency_level="Strong"：写入立刻可搜到。演示刚 insert 就 search 必须用强一致；
    # 生产高吞吐可降到 Bounded/Eventually，但可能短暂搜不到刚写入的数据。
    # index/search 的 metric_type 必须一致。COSINE 与 03 里手算的余弦同一套几何。
    try:
        vectorstore = Milvus(
            embedding_function=embeddings,
            connection_args={"uri": uri, "token": config.milvus["token"]},
            collection_name="langchain_demo",
            consistency_level="Strong",
            drop_old=True,
            auto_id=True,
            index_params={
                "metric_type": "COSINE",
                "index_type": "AUTOINDEX",
                "params": {},
            },
            search_params={
                "metric_type": "COSINE",
                "params": {},
            },
        )
    except Exception as exc:
        print(
            "连不上 Milvus。请确认 config.py 里的 host/port 可访问，"
            f"服务已启动。\n{type(exc).__name__}: {exc}"
        )
        return
    print()
    print_stage("1. add_documents：文本入库（内部会 Embedding）")
    # 官方文档：写入的是 Document，不是裸向量。page_content 被嵌入，metadata 原样存。
    # 源码分析：add_documents → embed_documents(page_content 列表) → 插入 collection。
    # 和 03 自己 embed 再手算余弦的区别：这里向量落在磁盘/服务端，进程重启后还能搜。
    docs = [
        Document(
            page_content="人工智能正在改变世界",
            metadata={"source": "百科", "topic": "ai"},
        ),
        Document(
            page_content="自然语言处理是人工智能的一个子领域",
            metadata={"source": "百科", "topic": "ai"},
        ),
        Document(
            page_content="机器学习是人工智能的一个子领域",
            metadata={"source": "百科", "topic": "ai"},
        ),
        Document(
            page_content="机房实行严格的权限分级准入制度，采用门禁刷卡与人脸核验双重认证。",
            metadata={"source": "机房制度", "topic": "datacenter"},
        ),
        Document(
            page_content="外来人员进入机房必须提前申请，并由内部运维全程陪同。",
            metadata={"source": "机房制度", "topic": "datacenter"},
        ),
        Document(
            page_content="今天天气很好，适合外出",
            metadata={"source": "天气", "topic": "weather"},
        ),
    ]
    ids = vectorstore.add_documents(docs)
    print(f"写入 {len(docs)} 条，返回主键 {ids}")
    print()
    print_stage("2. similarity_search：只返回文档，不带分数")
    # 官方文档：query 是自然语言，k 是返回条数。
    # 源码分析：先 embed_query(query)，再在 COSINE 索引上取 Top-K。
    # 和关键词搜索的区别：问句里可以没有「人工智能」四个字的完全匹配，仍能命中近义块。
    query = "人工智能包括哪些方向？"
    results = vectorstore.similarity_search(query=query, k=2)
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content}")
        print(f"   metadata: {doc.metadata}")
    print()
    print_stage("3. similarity_search_with_score：带上度量值")
    # 官方文档：返回 (Document, score) 二元组。
    # 工程实践：score 的含义取决于 metric_type，不能拿 L2 的阈值套到 COSINE 上。
    # 本示例用 COSINE：langchain-milvus 传出的是距离/相似度原始值，数值越大通常越像。
    # 换 L2 时越小越像。先打印真实分数，再决定过滤阈值。
    scored = vectorstore.similarity_search_with_score(query=query, k=3)
    for doc, score in scored:
        print(f"score={score:.4f}  {doc.page_content}  [{doc.metadata.get('source')}]")
    print()
    print_stage("4. expr 元数据过滤：只在机房制度里搜")
    # 官方文档：expr 是 Milvus 布尔表达式，在向量召回前/时过滤字段。
    # 源码分析：metadata 里的 source、topic 会映射成 collection 的标量字段，才能写进 expr。
    # 和「搜完再在 Python 里 if」的区别：过滤下推到数据库，候选集更小、也更安全（多租户）。
    room_query = "外人能不能进机房？"
    filtered = vectorstore.similarity_search(
        query=room_query,
        k=3,
        expr='topic == "datacenter"',
    )
    print(f"问题: {room_query}")
    if not filtered:
        print("没有命中。检查 expr 字段名是否与 metadata 一致。")
    for doc in filtered:
        print(f"- {doc.page_content}  [{doc.metadata}]")
    print()
    print_stage("5. as_retriever：交给后续 RAG 链 / Agent")
    # 官方文档：VectorStore.as_retriever() 得到 Runnable，invoke(问题) → list[Document]。
    # search_type="similarity" 就是上面的近邻检索；"mmr" 会降低结果之间的重复。
    # 和直接 similarity_search 的区别：Retriever 能接 create_agent / 检索链，接口统一。
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    retrieved = retriever.invoke("机房门禁怎么管？")
    for doc in retrieved:
        print(f"- {doc.page_content}")

    print()
    print_stage("知识点总结")
    print(
        "Embedding 算向量；Milvus 存向量并检索。两者缺一不可。\n"
        "drop_old=True 只适合演示。生产改 False，并用自己的 ids 做更新删除。\n"
        "index 与 search 的 metric_type 必须相同；过滤条件写 expr，不要只靠事后 if。"
    )


if __name__ == "__main__":
    main()
