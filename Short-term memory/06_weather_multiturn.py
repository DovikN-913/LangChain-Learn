"""
知识点：带短期记忆的多轮天气助手

官方文档：
https://docs.langchain.com/oss/python/langchain/short-term-memory

本示例主要学习：
1. 把 checkpointer、天气工具、多轮追问组合到一起。
2. 用户先说城市，后问「那明天呢」时应沿用同一城市。
3. 这是客服/助手类产品里最常见的短期记忆形态。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 工程实践：本文件在「Short-term memory」子目录。Python 默认只从该目录找模块，
# 把项目根插入 sys.path 后才能 `from utils import ...`。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from utils import get_chat_model, get_weather, print_stage


def query_weather(city: str) -> str:
    """查询指定城市的实时天气和预报。"""
    return get_weather(city)


def main() -> None:
    agent = create_agent(
        model=get_chat_model(),
        tools=[query_weather],
        # 官方文档：同一 thread_id 共享 messages。第二轮「那明天呢」能看到第一轮的城市。
        # 没有 checkpointer 时，第二轮是失忆的，模型不知道「那」指贵阳。
        checkpointer=InMemorySaver(),
        system_prompt=(
            "你是天气助手。必须调用工具查询真实天气。"
            "用户省略城市时，沿用对话里最近提到的城市。"
        ),
    )
    config = {"configurable": {"thread_id": "weather-session"}}

    questions = [
        "帮我查贵阳现在的天气。",
        "那明天呢？还要带伞吗？",
    ]
    for q in questions:
        print_stage(q)
        result = agent.invoke({"messages": [{"role": "user", "content": q}]}, config)
        print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
