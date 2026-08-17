"""
知识点：Checkpointer + thread_id 实现线程级短期记忆

官方文档：
https://docs.langchain.com/oss/python/langchain/short-term-memory

本示例主要学习：
1. Agent 默认不跨调用记住对话，除非挂上 checkpointer。
2. 同一 thread_id 共享一份 messages 状态。
3. 不同 thread_id 互相隔离，像两封不同的邮件会话。
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


def build_agent():
    return create_agent(
        model=get_chat_model(),
        tools=[],
        # 官方文档：checkpointer 把图状态按 thread 存下来，下一步 invoke 再读出。
        # 源码分析：InMemorySaver 用 defaultdict 以 thread_id 为键存在进程内存里。
        # 工程实践：只适合本地演示。进程一重启，所有线程历史都丢；生产换 Postgres/SQLite。
        checkpointer=InMemorySaver(),
        system_prompt="记住用户告诉你的信息。回答简短。",
    )


def main() -> None:
    agent = build_agent()
    # 官方文档：thread 比作「一封邮件会话」。config.configurable.thread_id 是会话主键。
    # 源码分析：checkpointer 按这个 id 读写 checkpoint；没传或不一致，等于新开一条空历史。
    # 和 context 的区别：thread_id 管跨轮 messages；context 是本轮只读配置（user_id 等）。
    thread_a = {"configurable": {"thread_id": "user-a"}}
    thread_b = {"configurable": {"thread_id": "user-b"}}

    print_stage("线程 A：自我介绍")
    # invoke 的第二个位置参数是 config，不是又一份 input。
    r1 = agent.invoke(
        {"messages": [{"role": "user", "content": "我叫林夏，住在贵阳。"}]},
        thread_a,
    )
    print(r1["messages"][-1].content)

    print_stage("线程 A：追问（应能记住）")
    # 同一 thread_id：checkpointer 先加载上次 checkpoint，再把本轮 user 消息追加进去。
    r2 = agent.invoke(
        {"messages": [{"role": "user", "content": "我叫什么？住哪？"}]},
        thread_a,
    )
    print(r2["messages"][-1].content)

    print_stage("线程 B：同一 Agent，另一条线程（不应知道林夏）")
    # 同一 agent 实例、不同 thread_id：存储命名空间隔离，B 看不到 A 的 messages。
    r3 = agent.invoke(
        {"messages": [{"role": "user", "content": "我叫什么？"}]},
        thread_b,
    )
    print(r3["messages"][-1].content)

    print_stage("知识点总结")
    print(
        "InMemorySaver 适合本地开发和单进程演示。\n"
        "生产环境换 Postgres / SQLite checkpointer，进程重启后仍能恢复线程。\n"
        "thread_id 由你生成：通常用 user_id + session_id。"
    )


if __name__ == "__main__":
    main()
