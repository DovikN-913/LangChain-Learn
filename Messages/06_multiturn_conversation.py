"""
知识点：用消息列表维护多轮对话

官方文档：
https://docs.langchain.com/oss/python/langchain/messages

本示例主要学习：
1. Chat Model 本身没有记忆：每次 invoke 都是无状态 HTTP 请求。
   「记住」完全来自你传入的 messages。
2. 每一轮都必须把历史完整（或裁剪后）再送回去，包括 System、过往 Human、过往 AI。
   只发最新一句，模型会忘掉名字、偏好、工具结果。
3. 这是 Short-term memory 的最底层形态。对话变长后改用 checkpointer / trim / 摘要，
   但原理相同：下一次请求里要有足够的历史消息。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 子目录默认找不到项目根，插入后才能 import utils
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.messages import HumanMessage, SystemMessage

from utils import get_chat_model, print_stage


def main() -> None:
    model = get_chat_model()
    # System 只放一次，但之后每一轮都会随 history 一起回传，否则人设会丢。
    history: list = [
        SystemMessage("你是助教。记住用户说过的名字，回答尽量短。"),
    ]

    turns = ["我叫林夏。", "我喜欢贵阳的天气。", "我叫什么？我喜欢哪里的天气？"]
    for text in turns:
        print_stage(f"用户：{text}")
        # 官方文档：交互默认无状态。必须先 append 本轮 Human，再把整段 history 交给 invoke。
        history.append(HumanMessage(text))
        ai = model.invoke(history)
        # 模型回复也要 append：下一轮问「我叫什么」时，请求里得有「我叫林夏」那一轮的问答。
        # 源码分析：厂商 API 看不到你本地的 list；没放进本次 messages 的内容等于不存在。
        history.append(ai)
        print("助手：", ai.content)

    print_stage("当前完整历史")
    for msg in history:
        print(f"[{msg.type}] {getattr(msg, 'text', msg.content)}")

    print_stage("知识点总结")
    print(
        "自己维护 list：最透明，适合短对话。\n"
        "对话变长后：用 checkpointer / trim / 摘要，见 Short-term memory。"
    )


if __name__ == "__main__":
    main()
