"""
知识点：invoke vs stream vs batch 对比

官方文档：
https://docs.langchain.com/oss/python/langchain/models#invocation

本示例主要学习：
1. 三种调用方式走同一套 Chat Model 接口，只是交付形态不同。
2. 选择场景：
   - invoke：一次拿完整 AIMessage。脚本、后台任务、必须等全文再解析（JSON/工具结果）。
   - stream：边生成边产出 chunk。聊天窗口、终端演示、降低「等待感」。
   - batch：客户端并行多条独立请求。同一时刻有多道互不依赖的题，要提高吞吐。
3. 不要用 batch 模拟多轮对话——多轮必须把历史 messages 串起来（见 Messages/06）。
4. 异步对应物是 ainvoke / astream / abatch，选择逻辑相同，只是不阻塞事件循环。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 子目录默认找不到项目根，插入后才能 import utils
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import get_chat_model, print_stage


def main() -> None:
    model = get_chat_model()

    print_stage("invoke：一次拿完整答案")
    # 官方文档：阻塞到生成结束，返回完整 AIMessage。下游要全文时首选它。
    print(model.invoke("用一句话说明 invoke。").content)

    print_stage("stream：边生成边展示")
    # 官方文档：迭代 AIMessageChunk。和 invoke 的区别是首 token 延迟更低，但下游每步都要能处理增量。
    for chunk in model.stream("用一句话说明 stream。"):
        print(chunk.content, end="", flush=True)
    print()

    print_stage("batch：多条独立请求并行")
    # 官方文档：客户端并行，不是厂商离线 Batch API。结果按输入顺序返回。
    # 这两问互不依赖，才适合 batch；若第二问需要第一问的答案，必须走多轮 messages。
    for response in model.batch(
        ["用四个字概括 batch。", "用四个字概括 stream。"]
    ):
        print(response.content)

    print_stage("如何选择")
    print(
        "选 invoke：脚本、批处理后台任务、必须拿到全文再解析。\n"
        "选 stream：聊天窗口、终端演示、降低「等待感」。\n"
        "选 batch：同一时刻有多条互不依赖的问题。\n"
        "不要用 batch 模拟多轮对话——多轮必须把历史 messages 串起来。"
    )


if __name__ == "__main__":
    main()
