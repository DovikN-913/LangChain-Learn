"""
知识点：有记忆 vs 无记忆

官方文档：
https://docs.langchain.com/oss/python/langchain/short-term-memory

本示例主要学习：
1. 不挂 checkpointer 时，每次 invoke 都是独立会话。
2. 挂上 checkpointer 并复用 thread_id 才有短期记忆。
3. 这和「在代码里自己 append messages」是同一问题的两种解法。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 工程实践：本文件在「Short-term memory」子目录。Python 默认只从该目录找模块，
# 把项目根插入 sys.path 后才能 `from utils import ...`。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from utils import get_chat_model, print_stage


def ask(agent, text: str, config=None):
    result = agent.invoke({"messages": [{"role": "user", "content": text}]}, config)
    return result["messages"][-1].content


def main() -> None:
    model = get_chat_model()

    print_stage("无 checkpointer：第二轮通常不记得名字")
    # 官方文档：不挂 checkpointer 时每次 invoke 独立。模型本身无状态，看不到上一轮。
    # 源码分析：compile 时 checkpointer=None，图跑完状态就丢，第二次 invoke 从空 messages 开始。
    stateless = create_agent(model=model, tools=[], system_prompt="尽量短。")
    print(ask(stateless, "我叫林夏。"))
    print(ask(stateless, "我叫什么？"))

    print_stage("有 checkpointer：第二轮应能记住")
    stateful = create_agent(
        model=model,
        tools=[],
        checkpointer=InMemorySaver(),
        system_prompt="尽量短。",
    )
    # 必须同时满足：挂了 checkpointer，且两次 invoke 传同一个 thread_id。
    cfg = {"configurable": {"thread_id": "cmp-1"}}
    print(ask(stateful, "我叫林夏。", cfg))
    print(ask(stateful, "我叫什么？", cfg))

    print_stage("如何选择")
    # 工程实践：自己维护 messages list 逻辑完全透明，适合单进程脚本；
    # checkpointer 适合多请求、要可恢复。两者解决的是同一问题。
    print(
        "无记忆：分类、抽取、一次性问答。\n"
        "有记忆：客服、助手、多轮任务。\n"
        "自己维护 messages：逻辑完全透明。\n"
        "checkpointer：多请求、多进程、要可恢复时更合适。"
    )


if __name__ == "__main__":
    main()
