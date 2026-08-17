"""
知识点：裁剪过长的消息历史

官方文档：
https://docs.langchain.com/oss/python/langchain/short-term-memory#trim-messages

本示例主要学习：
1. 上下文窗口有限，不能无限堆积 messages。
2. 用 before_model 中间件在调用模型前裁剪。
3. 删除时要用 RemoveMessage，并保证剩余历史对厂商仍然合法。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# 工程实践：本文件在「Short-term memory」子目录。Python 默认只从该目录找模块，
# 把项目根插入 sys.path 后才能 `from utils import ...`。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import before_model
from langchain.messages import RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

from utils import get_chat_model, print_stage


@before_model
def keep_recent(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """只保留系统提示之后最近几条消息。"""
    # 官方文档：@before_model 在每次调模型前改 state。流程：start → before_model → model → tools → 再回到 before_model。
    # 返回 None 表示不改；返回 dict 会按 reducer 合并进 state。
    messages = state["messages"]
    if len(messages) <= 6:
        return None
    kept = messages[-6:]
    # 源码分析：messages 带 add_messages reducer，直接赋值不会替换，只会追加。
    # 必须先发 RemoveMessage(id=REMOVE_ALL_MESSAGES)（哨兵值 "__remove_all__"），
    # reducer 会丢掉旧列表，只保留这条之后的 kept。
    # 和 RemoveMessage(id=某条id) 的区别：后者只删一条；REMOVE_ALL_MESSAGES 清空再重建。
    # 工程实践：删完后历史必须合法——有 tool_calls 的 AIMessage 后面必须跟对应 ToolMessage。
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *kept]}


def main() -> None:
    agent = create_agent(
        model=get_chat_model(),
        tools=[],
        middleware=[keep_recent],
        checkpointer=InMemorySaver(),
        system_prompt="回答尽量短。如果记不住就直说。",
    )
    config = {"configurable": {"thread_id": "trim-demo"}}

    for text in [
        "我叫林夏。",
        "我有一只猫叫糯米。",
        "我喜欢爬山。",
        "我在学 LangChain。",
        "刚才说的猫叫什么？",
    ]:
        print_stage(text)
        result = agent.invoke({"messages": [{"role": "user", "content": text}]}, config)
        print(result["messages"][-1].content)
        print("当前消息条数:", len(result["messages"]))

    print_stage("知识点总结")
    print(
        "裁剪简单、便宜，但会丢掉被删掉的事实。\n"
        "需要保留关键信息时用 SummarizationMiddleware。\n"
        "删除后不要留下「有 tool_calls 却没有 ToolMessage」的残缺历史。"
    )


if __name__ == "__main__":
    main()
