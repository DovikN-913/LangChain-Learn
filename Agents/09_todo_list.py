"""
知识点：Planning — TodoListMiddleware 让 Agent 拆任务

官方文档：
https://docs.langchain.com/oss/python/langchain/agents#planning-and-delegation

本示例主要学习：
1. 复杂任务超出单次上下文时，先规划再执行。
2. TodoListMiddleware 是 langchain 内置的规划中间件，不依赖 deepagents。
3. 子 Agent / 文件系统属于 Deep Agents 扩展，本文件不安装额外包。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 工程实践：本文件在 Agents/ 子目录。Python 默认找不到项目根的 utils.py，
# 必须把根目录插入 sys.path，后面才能 `from utils import ...`。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware

from utils import get_chat_model, get_weather, print_stage


def query_weather(city: str) -> str:
    """查询指定城市的天气。"""
    return get_weather(city)


def packing_tip(condition: str) -> str:
    """根据天气现象给出一句打包建议。"""
    if "雨" in condition:
        return "带折叠伞和防水外套。"
    if "晴" in condition:
        return "带防晒和薄外套。"
    return "带一件可增减的外套，包里备折叠伞。"


def main() -> None:
    # 官方文档：TodoListMiddleware 是 langchain 内置规划中间件，给 Agent 一张待办清单。
    # 复杂任务先拆步再执行，主上下文不容易漏工具。子 Agent / 文件系统要 deepagents，本文件不装。
    # 源码分析：中间件会向模型暴露写待办的工具；system_prompt 里「先规划再调用」才能真正用上。
    agent = create_agent(
        model=get_chat_model(),
        tools=[query_weather, packing_tip],
        middleware=[TodoListMiddleware()],
        system_prompt=(
            "用户要出行时：先规划待办（查天气、给打包建议），"
            "再按顺序调用工具，最后汇总。不要编造天气。"
        ),
    )

    print_stage("带规划的多工具 Agent")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "我明天去贵阳玩一天，帮我安排查天气和打包。"}]}
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
