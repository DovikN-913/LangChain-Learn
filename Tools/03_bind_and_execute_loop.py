"""
知识点：bind_tools 之后的手动执行循环

官方文档：
https://docs.langchain.com/oss/python/langchain/models#tool-calling
https://docs.langchain.com/oss/python/langchain/tools

本示例主要学习：
1. 没有 Agent 时，你必须自己写「调用 → 执行 → 回填」循环。
2. 一次回复里可能有多个 tool_calls，可以逐个或并行执行。
3. 这是理解 Agent 内部在做什么的最佳入口。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 工程实践：本文件在 Tools/ 子目录。Python 默认找不到项目根的 utils.py，
# 必须把根目录插入 sys.path，后面才能 `from utils import ...`。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.messages import HumanMessage
from langchain.tools import tool

from utils import get_chat_model, get_weather, print_stage


@tool
def query_weather(city: str) -> str:
    """查询指定城市的实时天气和预报。"""
    return get_weather(city)


@tool
def add(a: int, b: int) -> int:
    """计算两个整数之和。"""
    return a + b


# 工程实践：用 name → Tool 映射，循环里按模型给出的 call["name"] 分发。
TOOLS = {t.name: t for t in (query_weather, add)}


def run_until_text(question: str, max_rounds: int = 4) -> str:
    # 官方文档：bind_tools 只把 schema 绑到模型上。模型返回的是 tool_calls 请求，不会执行函数。
    # 和 create_agent 的区别：这里循环、回填、停机条件都要自己写。
    model = get_chat_model().bind_tools(list(TOOLS.values()))
    messages: list = [HumanMessage(question)]

    # 工程实践：必须限制 max_rounds。模型可能反复调工具空转，没有 Agent 的内置停机策略。
    for round_idx in range(max_rounds):
        print_stage(f"第 {round_idx + 1} 轮模型输出")
        ai = model.invoke(messages)
        messages.append(ai)
        if not ai.tool_calls:
            return ai.content

        # 官方文档：一次 AIMessage 里可以有多个 tool_calls（本例：天气 + 加法）。
        # tool.invoke(call) 传入完整 call（含 name/args/id），会生成带 tool_call_id 的 ToolMessage。
        # 源码分析：只传 call["args"] 也能执行函数，但拿不到对齐 id 的 ToolMessage，下一轮厂商常会报错。
        for call in ai.tool_calls:
            print("调用:", call["name"], call["args"])
            tool = TOOLS[call["name"]]
            messages.append(tool.invoke(call))

    return "超过最大轮次，停止。"


def main() -> None:
    answer = run_until_text("贵阳今天天气如何？另外 3+5 等于多少？")
    print_stage("最终回答")
    print(answer)

    print_stage("知识点总结")
    print(
        "这就是 Agent 循环的手工版。\n"
        "生产中应限制 max_rounds，防止模型反复调工具空转。"
    )


if __name__ == "__main__":
    main()
