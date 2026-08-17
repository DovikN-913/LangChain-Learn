"""
知识点：Embedding 把文本变成向量，用「语义距离」代替关键词匹配

官方文档：
https://docs.langchain.com/oss/python/integrations/embeddings/openai

本示例主要学习：
1. Chat Model 负责生成文字；Embedding 负责把文字映射到固定长度的浮点向量。
2. 语义相近的句子，向量夹角小、余弦相似度高；字面完全不同也可以很近。
3. embed_query 一次一条（检索时的问题）；embed_documents 一次一批（入库的文档块）。
4. 向量库内部就是这么调的：建索引走 embed_documents，查询走 embed_query。
5. 入库和查询必须用同一个 Embedding 模型。换模型后旧向量全部作废，要重嵌。

本项目模型：阿里百炼 text-embedding-v1（OpenAI 兼容接口，见 utils.get_embedding_model）。
下一步：把向量交给 Milvus 做近似最近邻检索，见 04-milvus向量数据库示例.py。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 工程实践：本文件在 LangChain-RAG/ 子目录，插入项目根才能 import utils。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from utils import get_embedding_model, print_stage


def cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """用手算余弦，看清「语义相近」在数值上长什么样。

    逻辑：点积 / (模长乘积)。1=同向，0=正交，-1=反向。零向量返回 0，避免除零。
    为什么先自己算一遍：还没接 Milvus 时，就能验证「外人进机房」更靠近「陪同」而不是「天气」。
    向量库的 COSINE 度量，思想就是这个公式（实现上可能用 1-相似度当距离）。
    """
    v1 = np.asarray(vector1, dtype=float)
    v2 = np.asarray(vector2, dtype=float)
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)


def main() -> None:
    print_stage("0. 初始化 Embedding 模型")
    # 官方文档：OpenAIEmbeddings 走 /embeddings 接口，不是 Chat Completions。
    # 工程实践：百炼兼容网关用同一套 api_key / base_url，但 model 必须是 embedding 名。
    # 不要把 qwen3.6-35b-a3b 传给 OpenAIEmbeddings，接口和向量维数都不对。
    embeddings = get_embedding_model()
    print("模型:", embeddings.model)
    print()

    print_stage("1. embed_query：单条文本 → 一条向量")
    # 官方文档：检索时的用户问题用 embed_query。
    # 源码分析：返回 list[float]，长度由模型决定，调用方不能改 text-embedding-v1 的维数。
    # 和 Chat Model 的区别：这里没有 content、没有 token 用量对话；只有一组坐标。
    query = "外人能不能进机房？"
    query_vector = embeddings.embed_query(query)
    print(f"文本: {query}")
    print(f"向量维数: {len(query_vector)}  （同一模型恒定，建库时索引维数必须与此一致）")
    print(f"前 5 维: {[round(x, 6) for x in query_vector[:5]]}")
    print()

    print_stage("2. embed_documents：批量文本 → 多条向量")
    # 官方文档：给语料建索引用 embed_documents，不要在 for 循环里反复 embed_query。
    # 源码分析：内部会按批次打 HTTP，自带重试；循环 embed_query 更慢、更容易撞限流。
    # RAG 里这些字符串通常来自 02 的 split_documents，而不是整份制度原文。
    texts = [
        "机房实行严格的权限分级准入制度，采用门禁刷卡与人脸核验双重认证。",
        "外来人员进入机房必须提前申请，并由内部运维全程陪同。",
        "精密空调把机房温度控制在 18℃-24℃。",
        "今天贵阳多云，适合出门跑步。",
    ]
    doc_vectors = embeddings.embed_documents(texts)
    print(f"写入 {len(texts)} 条，得到 {len(doc_vectors)} 条向量")
    print(f"每条维数: {len(doc_vectors[0])}  （必须等于 query 的 {len(query_vector)}）")
    print()

    print_stage("3. 余弦相似度：问题 vs 各条文档")
    # 工程实践：这就是向量检索在内存里的朴素版——对每条文档算一次相似度再排序。
    # Milvus 做的是同一件事的近似最近邻加速，数据量大时不能这样全表扫描。
    scored = [
        (cosine_similarity(query_vector, vec), texts[i])
        for i, vec in enumerate(doc_vectors)
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    for score, text in scored:
        print(f"  {score:.4f}  {text}")
    print("预期：准入 / 陪同 应排在天气前面——字面上都没出现「外人」，靠的是语义。")
    print()

    print_stage("4. 对照：同一句、近义句、无关句")
    # 同一句应接近 1.0（浮点误差会略小于 1）。
    # 近义改写仍然高；话题完全不同则明显掉下去。阈值没有万能数字，要在自己的语料上标定。
    same = embeddings.embed_query(texts[0])
    print(f"同一句 vs 自己: {cosine_similarity(doc_vectors[0], same):.4f}")
    print(f"准入 vs 陪同（近义）: {cosine_similarity(doc_vectors[0], doc_vectors[1]):.4f}")
    print(f"准入 vs 天气（无关）: {cosine_similarity(doc_vectors[0], doc_vectors[3]):.4f}")
    print()

    print_stage("知识点总结")
    print(
        "embed_query 问句，embed_documents 文档块，两者必须同一模型。\n"
        "维数由模型决定：text-embedding-v1 建的库，不能拿另一套 embedding 去搜。\n"
        "自己算余弦只适合小样本；生产把向量交给向量数据库，见 04。"
    )


if __name__ == "__main__":
    main()
