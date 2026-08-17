"""
知识点：Harness 三件套 — model / tools / system_prompt

官方文档：
https://docs.langchain.com/oss/python/langchain/agents#core-components

本示例主要学习：
1. model 可以是已初始化的实例（本项目用 DashScope ChatOpenAI）。
2. tools 可以是普通函数、@tool、或 tool dict。
3. system_prompt 接受 str 或 SystemMessage；运行时动态提示要用 middleware。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 工程实践：本文件在 Agents/ 子目录。Python 默认找不到项目根的 utils.py，
# 必须把根目录插入 sys.path，后面才能 `from utils import ...`。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.agents import create_agent
from langchain.messages import SystemMessage
from langchain.tools import tool

from utils import get_chat_model, get_weather, print_stage


@tool
def query_weather(city: str) -> str:
    """查询指定城市的实时天气。"""
    return get_weather(city)


# 官方文档：tools 可混用 @tool 和普通函数。有类型注解 + docstring 即可。
def add(a: int, b: int) -> int:
    """计算两个整数之和。"""
    return a + b


def main() -> None:
    # 官方文档：model 可传 "provider:model" 字符串，或已初始化实例。
    # 工程实践：本项目是百炼兼容接口，不能写 model="openai:gpt-5.5"，必须走 get_chat_model()。
    model = get_chat_model()

    print_stage("1. 同一模型，相同工具，同一问题，不同 system_prompt，行为会变，结果也会变")
    # 官方文档：system_prompt 接受 str 或 SystemMessage，静态塑形「怎么做事」。
    # 运行时按用户身份改提示要用 middleware（@dynamic_prompt），不能每次重建 Agent 才算动态。
    concise = create_agent(
        model=model,
        tools=[query_weather, add],
        system_prompt=SystemMessage("只输出结论，不超过 40 字。天气必须查工具。"),
    )
    verbose = create_agent(
        model=model,
        tools=[query_weather, add],
        system_prompt="你是详细的气象解说员。先调用工具，再分点说明体感、穿衣和是否带伞。",
    )

    question = {"messages": [{"role": "user", "content": "贵阳今天出门要带伞吗？另外 2+3=?"}]}
    print("简洁人设：", concise.invoke(question)["messages"][-1].content)
    print()
    print("详细人设：", verbose.invoke(question)["messages"][-1].content)
    print()
    print_stage("2. 知识点总结")
    print(
        "换模型、换工具、换提示，都不必手写循环。\n"
        "动态按用户身份改提示：用 @dynamic_prompt 中间件，见 Short-term memory README。"
    )


if __name__ == "__main__":
    main()
