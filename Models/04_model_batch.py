"""
知识点：batch() / batch_as_completed() 批量调用

官方文档：
https://docs.langchain.com/oss/python/langchain/models#batch

本示例主要学习：
1. batch 是客户端并行发多个独立 HTTP 请求，不是 OpenAI/Anthropic 的厂商 Batch API
   （厂商 Batch 是提交离线任务、异步回调，计费和 SLA 都不同）。
2. config.max_concurrency 限制同时在飞的请求数，避免打满限流。
3. batch 结果顺序与输入一致；batch_as_completed 谁先完成谁先返回，可能乱序。
   乱序时每项带原始下标，才能把答案对回原来的问题。

适用：多条互不依赖的问题。后一个问题依赖前一个答案时，应走多轮 messages，不要 batch。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 子目录默认找不到项目根，插入后才能 import utils
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import get_chat_model, print_stage


def main() -> None:
    model = get_chat_model()
    questions = [
        "用一句话解释什么是 token。",
        "用一句话解释什么是 temperature。",
        "用一句话解释什么是 tool calling。",
    ]

    print_stage("batch：按输入顺序返回")
    # 官方文档：batch 会在客户端并行处理，但返回列表与输入下标一一对应。
    # max_concurrency=3：最多同时 3 个请求。源码分析：它走 Runnable 的并发配置，
    # 不是厂商 API 参数，不会出现在 HTTP JSON 里。
    # 工程实践：并发过大容易 429；过小则吞吐没提升。先按配额设一个上限。
    responses = model.batch(
        questions,
        config={"max_concurrency": 3},
    )
    for q, r in zip(questions, responses, strict=True):
        print(f"Q: {q}\nA: {r.content}\n")

    print_stage("batch_as_completed：完成一个返回一个")
    # 官方文档：谁先完成谁先 yield，顺序可能打乱。
    # 每项是 (原始下标, AIMessage)：乱序时用 index 对回 questions[index]。
    # 和 batch 的区别：batch 等全部结束再按序返回；这里可以边完成边处理。
    for item in model.batch_as_completed(questions):
        index, response = item
        print(f"index={index}: {response.content}")

    print_stage("知识点总结")
    print(
        "适用：多个互不依赖的问题，想提高吞吐。\n"
        "不适用：后一个问题依赖前一个答案——这时应走多轮 messages。"
    )


if __name__ == "__main__":
    main()
