"""
知识点：2-Step RAG = 先检索，再生成（LCEL 最短链路）

官方文档：
https://docs.langchain.com/oss/python/langchain/retrieval

本示例主要学习：
1. 官方把 RAG 分成三种：2-Step（必检索）、Agentic（模型决定是否检索）、Hybrid（检索/答案校验循环）。
2. 2-Step 最简单：Retriever 取出片段 → Prompt 拼上下文 → Chat Model 生成。
3. LCEL 写法：{"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | model。
4. 本文件不负责入库。先跑 04 把 toy 文档写入 langchain_demo，再来这里问一句。

完整产品形态（阈值过滤、溯源、交互）见 06；模型自己决定是否检索见 07。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 工程实践：本文件在 LangChain-RAG/ 子目录，插入项目根才能 import utils / config。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_milvus import Milvus

import config
from utils import get_chat_model, get_embedding_model, print_stage


def milvus_uri() -> str:
    """04 / 05 共用 langchain_demo；06 / 07 用 enterprise_kb，避免被 04 的 drop_old 清掉。"""
    return f"http://{config.milvus['host']}:{config.milvus['port']}"


def main() -> None:
    print_stage("连接 04 写入的 collection（drop_old=False）")
    # 官方文档：Retriever 是「给一句自然语言，返回 Document 列表」的接口。
    # VectorStore.as_retriever() 就是最常见实现；底层仍是 embed_query + 近邻搜索。
    # 工程实践：这里必须带上 embedding_function，否则无法把问题编成向量。
    # drop_old=False：只连已有表，不重建。库是空的就先跑 04。
    embeddings = get_embedding_model()
    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": milvus_uri(), "token": config.milvus["token"]},
        collection_name="langchain_demo",
        consistency_level="Strong",
        drop_old=False,
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    print_stage("拼一条 2-Step 链")
    # 官方文档：2-Step RAG 每次都先检索再生成，控制力强、时延可预期，适合 FAQ / 制度问答。
    # 源码分析：RunnablePassthrough 把用户原句原样交给 question 槽；
    # retriever | format_docs 把 Document 列表收成一段字符串再填 context。
    # ChatPromptTemplate 的 {context}/{question} 必须和左边 dict 的 key 同名。
    # StrOutputParser：把 AIMessage 收成 str。不加的话 invoke 返回的是消息对象。
    # 和 06 的区别：检索藏在链里，调用方拿不到 Document 列表，也就没法打印来源和分数。
    # 和 07 Agent 的区别：这里没有「要不要搜」的判断，也不会多轮调工具。
    prompt = ChatPromptTemplate.from_template(
        """你是问答助手。只根据上下文回答，不要编造。
上下文没有相关信息时回答「不知道」。

上下文：
{context}

问题：{question}

回答："""
    )

    def format_docs(docs) -> str:
        """Retriever 吐出的是 Document 列表，模型只能读字符串。

        逻辑：按顺序把 page_content 拼起来。
        为什么这里故意不带来源文件名：本文件只讲「链怎么接」；溯源放到 06，对比更清楚。
        """
        return "\n\n".join(
            f"文档{i + 1}:\n{doc.page_content}" for i, doc in enumerate(docs)
        )

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | get_chat_model()
        | StrOutputParser()
    )
    # 执行顺序（思想）：用户一句话进来 → 先检索再填 {context} → 同时原句填 {question}
    # → Prompt 渲染成 messages → Chat Model 生成 → Parser 收成纯文本。
    # 提问方只看到最终 str，这是 2-Step 最短形态，也是它不好做溯源的原因。

    question = "机房门禁怎么管？"
    print_stage(f"提问：{question}")
    print(rag_chain.invoke(question))
    print(
        "\n这条链拿不到带分数的来源列表。"
        "要溯源、阈值过滤、交互，请看 06；要模型决定是否检索，请看 07。"
    )


if __name__ == "__main__":
    main()
