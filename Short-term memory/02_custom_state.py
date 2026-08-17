"""
知识点：扩展 AgentState，在短期记忆里存自定义字段

官方文档：
https://docs.langchain.com/oss/python/langchain/short-term-memory#customizing-agent-memory

本示例主要学习：
1. 默认短期记忆主要是 messages。
2. 继承 AgentState 可以加业务字段，例如偏好城市。
3. 这些字段随 checkpointer 一起按 thread 持久化。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 工程实践：本文件在「Short-term memory」子目录。Python 默认只从该目录找模块，
# 把项目根插入 sys.path 后才能 `from utils import ...`。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.agents import AgentState, create_agent
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.memory import InMemorySaver

from utils import get_chat_model, get_weather, print_stage


class TravelState(AgentState):
    # 官方文档：默认 AgentState 主要是 messages（add_messages 只追加）。
    # 继承后加的字段随 checkpoint 按 thread 持久化；不挂 checkpointer 则只活在这一次 invoke。
    # 和 messages 的区别：自然语言历史走 messages；结构化偏好/计数器走自定义字段。
    home_city: str


@tool
def query_home_weather(runtime: ToolRuntime) -> str:
    """查询用户常驻城市的天气。城市从会话状态读取，不需要用户再提供。"""
    # 官方文档：runtime 对模型隐藏，不进工具 schema。runtime.state 是当前 thread 的图状态。
    # 源码分析：ToolRuntime.state 就是 AgentState；.get 避免字段尚未写入时 KeyError。
    city = runtime.state.get("home_city") or "贵阳"
    return get_weather(city)


def main() -> None:
    agent = create_agent(
        model=get_chat_model(),
        tools=[query_home_weather],
        # 官方文档：state_schema 替换默认 AgentState，作为与中间件 state 合并的基座。
        state_schema=TravelState,
        checkpointer=InMemorySaver(),
        system_prompt="用户问天气且没提城市时，调用 query_home_weather。",
    )
    config = {"configurable": {"thread_id": "travel-1"}}

    print_stage("写入自定义状态并问天气")
    result = agent.invoke(
        {
            "messages": [{"role": "user", "content": "我出门要不要带伞？"}],
            # invoke 的 input 就是对 state 的一次更新；自定义字段和 messages 一起传入。
            "home_city": "贵阳",
        },
        config,
    )
    print(result["messages"][-1].content)

    print_stage("知识点总结")
    print(
        "messages 适合自然语言历史。\n"
        "自定义字段适合结构化偏好、计数器、当前任务状态。\n"
        "工具通过 ToolRuntime.state 读取，不必让模型再猜。"
    )


if __name__ == "__main__":
    main()
