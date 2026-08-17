"""
知识点：stream() 流式输出

官方文档：
https://docs.langchain.com/oss/python/langchain/models#stream

本示例主要学习：
1. stream 返回 AIMessageChunk 迭代器，边生成边产出；invoke 则等全文。
2. chunk 相加规则：`full = chunk if full is None else full + chunk`。
   底层是 AIMessageChunk.__add__：拼接 content，合并 tool_call_chunks / usage 等。
   第一个 chunk 不能用 `None + chunk`，所以要用「空则取当前，否则相加」。
3. print(..., flush=True)：终端默认按行缓冲，不 flush 就看不到打字机效果。
4. 累加后的对象可当普通 AIMessage 用；content_blocks 是跨厂商标准化块视图，
   可从中读 reasoning / tool_call_chunk，不只是纯文本。

流式适合前端打字机效果；下游必须等全文才能解析时，用 invoke。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 子目录默认找不到项目根，插入后才能 import utils
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.messages import AIMessageChunk

from utils import get_chat_model, print_stage


def main() -> None:
    model = get_chat_model()

    print_stage("逐 token 打印")
    full: AIMessageChunk | None = None
    for chunk in model.stream("用三句话介绍鹦鹉为什么会说话。"):
        # 官方文档：chunk.content 是当前增量，不是到目前为止的全文。
        # flush=True：工程实践。Python 在无换行时会缓冲 stdout，
        # 不立刻刷出的话，「边生成边显示」会退化成一次性蹦出。
        print(chunk.content, end="", flush=True)
        # 源码分析：AIMessageChunk 重载了 +。相加会拼接文本，并合并
        # tool_call_chunks、usage_metadata 等。第一片不能 None + chunk。
        full = chunk if full is None else full + chunk
    print("\n")

    print_stage("累加后的完整消息")
    assert full is not None
    print("类型:", type(full).__name__)
    print("全文:", full.content)
    # 官方文档：content_blocks 把厂商私有结构懒解析成统一块（text / reasoning 等）。
    # 它不是替换 .content，而是标准化视图；需要推理块或工具调用增量时看这里。
    print("content_blocks:", full.content_blocks)

    print_stage("知识点总结")
    print(
        "invoke：等全部生成完再返回 AIMessage。\n"
        "stream：持续返回 AIMessageChunk，适合实时展示。\n"
        "累加规则：full = chunk if full is None else full + chunk"
    )


if __name__ == "__main__":
    main()
