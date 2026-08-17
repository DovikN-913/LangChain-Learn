"""
知识点：invoke 时传入 thread_id 与 context

官方文档：
https://docs.langchain.com/oss/python/langchain/agents#invocation

本示例主要学习：
1. invoke 本质是给 Agent State 一次更新；要续聊必须带同一 thread_id。
2. 本地持久化必须显式传 checkpointer；LangSmith 部署会自动配。
3. thread_id 管会话历史；context 管本轮只读配置（user_id、开关），两者经常一起传。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

# 工程实践：本文件在 Agents/ 子目录。Python 默认找不到项目根的 utils.py，
# 必须把根目录插入 sys.path，后面才能 `from utils import ...`。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver

from utils import get_chat_model, get_weather, print_stage


# 官方文档：context_schema 定义本轮只读配置的形状。工具经 runtime.context 读取。
@dataclass
class UserContext:
    user_id: str # 用户唯一标识
    default_city: str # 默认查询城市


@tool
def query_weather(city: str) -> str:
    """按城市名查询天气。"""
    return get_weather(city)


@tool
def who_am_i(runtime: ToolRuntime[UserContext]) -> str:
    """查看当前调用方身份和默认城市。"""
    ctx = runtime.context
    return f"user_id={ctx.user_id}, default_city={ctx.default_city}"


def main() -> None:
    # 官方文档：本地必须显式 checkpointer=InMemorySaver()；LangSmith 部署会自动提供。
    # thread_id 管「哪一次会话的历史」；context 管「这一跑工具能读到的身份」。生产里通常一起传。
    agent = create_agent(
        model=get_chat_model(),
        tools=[query_weather, who_am_i],
        context_schema=UserContext,
        checkpointer=InMemorySaver(),
        system_prompt=(
            "先在需要时调用工具。用户省略城市时沿用对话里最近的城市。"
            "问身份时调用 who_am_i。"
        ),
    )
    # 源码分析：thread_id 放在 config["configurable"] 里，checkpointer 按它存/取 checkpoint。
    config = {"configurable": {"thread_id": str(uuid7())}}
    ctx = UserContext(user_id="user-123", default_city="贵阳")

    print_stage("第一轮：查天气")
    r1 = agent.invoke(
        {"messages": [{"role": "user", "content": "帮我查贵阳现在的天气。"}]},
        config=config,
        context=ctx,
    )
    print(r1["messages"][-1].content)

    print_stage("第二轮：同一 thread，追问明天")
    # 工程实践：同一 thread_id 才能续上 messages。「那明天呢」才能知道还是贵阳。
    # 换了 thread_id 就是新会话，Agent 不记得上一轮。
    r2 = agent.invoke(
        {"messages": [{"role": "user", "content": "那明天呢？还要带伞吗？"}]},
        config=config,
        context=ctx,
    )
    print(r2["messages"][-1].content)

    print_stage("同一 thread：读取 context")
    r3 = agent.invoke(
        {"messages": [{"role": "user", "content": "我是谁？默认城市是哪？"}]},
        config=config,
        context=ctx,
    )
    print(r3["messages"][-1].content)


if __name__ == "__main__":
    main()
