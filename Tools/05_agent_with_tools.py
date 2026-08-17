"""
知识点：create_agent 自动编排工具

官方文档：
https://docs.langchain.com/oss/python/langchain/tools
https://docs.langchain.com/oss/python/langchain/models#tool-calling

本示例主要学习：
1. Agent 会自动完成工具调用循环，不必手写 bind_tools 回填。
2. 普通函数只要有类型注解和 docstring，也能直接当工具。
3. 天气查询走 utils.get_weather，保证和手工循环示例同源。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 工程实践：本文件在 Tools/ 子目录。Python 默认找不到项目根的 utils.py，
# 必须把根目录插入 sys.path，后面才能 `from utils import ...`。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.agents import create_agent

from utils import get_chat_model, get_weather, print_stage


# 官方文档：普通函数只要有类型注解和 docstring，也能直接放进 tools=。
# 不必先 @tool；create_agent 内部会包装成 Tool。对比 03 的手工 bind_tools 循环。
def query_weather(city: str) -> str:
    """查询指定城市的实时天气、空气质量和预报。用户问天气时必须调用。"""
    return get_weather(city)


def main() -> None:
    # 官方文档：create_agent 自动「想 → 调工具 → 回填 ToolMessage → 再想」，直到不再调工具。
    # bind_tools 只让模型发出 tool_calls；谁执行、怎么停，都要你自己写（见 03）。
    # system_prompt 塑形「必须查真实天气」，减少模型凭训练数据编数字。
    agent = create_agent(
        model=get_chat_model(),
        tools=[query_weather],
        system_prompt="你是天气助手。必须调用工具获取真实天气，禁止编造数字。",
    )

    print_stage("Agent 自动查天气")
    # 源码分析：invoke 更新 AgentState。结果在 result["messages"] 里，最后一条通常是最终 AIMessage。
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "贵阳今天适不适合带伞出门？"}]}
    )
    print(result["messages"][-1].content)

    print_stage("对比")
    print(
        "手工循环：你能看清每一步，适合排错和教学。\n"
        "create_agent：少写样板代码，适合产品功能。\n"
        "同一个 query_weather 两种用法都可以。"
    )


if __name__ == "__main__":
    main()
