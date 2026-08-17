"""
知识点：create_agent vs bind_tools 手工循环

官方文档：
https://docs.langchain.com/oss/python/langchain/agents
https://docs.langchain.com/oss/python/langchain/models#tool-calling

本示例主要学习：
1. 两者都能调同一个天气工具。
2. 区别在于谁负责循环、记忆、中间件和流式。
3. 什么时候该选 Agent，什么时候该停在 bind_tools。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 工程实践：本文件在 Agents/ 子目录。Python 默认找不到项目根的 utils.py，
# 必须把根目录插入 sys.path，后面才能 `from utils import ...`。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool

from utils import get_chat_model, get_weather, print_stage


@tool
def query_weather(city: str) -> str:
    """查询指定城市的天气。"""
    return get_weather(city)


def run_manual(question: str) -> str:
    # 官方文档：bind_tools 只绑定 schema。执行、回填、再调用都要自己写。
    # 适合：一轮工具、要精确控流、不想引入图。没有 checkpointer / 中间件 / 自动多轮。
    model = get_chat_model().bind_tools([query_weather])
    messages: list = [HumanMessage(question)]
    ai = model.invoke(messages)
    messages.append(ai)
    for call in ai.tool_calls:
        messages.append(query_weather.invoke(call))
    if ai.tool_calls:
        ai = model.invoke(messages)
    return ai.content


def run_agent(question: str) -> str:
    # 官方文档：create_agent 自动循环，并可挂记忆、流式、结构化出口、重试中间件。
    # 适合：未知步数、多工具、产品助手。代价是多一层图，排错不如手工循环直观。
    agent = create_agent(
        model=get_chat_model(),
        tools=[query_weather],
        system_prompt="必须调用工具查询真实天气。",
    )
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].content


def main() -> None:
    question = "贵阳现在天气怎么样？"

    print_stage("手工 bind_tools 循环")
    print(run_manual(question))

    print_stage("create_agent")
    print(run_agent(question))

    print_stage("如何选择")
    print(
        "选 bind_tools：只要一轮工具、要精确控流、不想引入图。\n"
        "选 create_agent：未知步数、多工具、要记忆/中间件/流式/结构化出口。"
    )


if __name__ == "__main__":
    main()
