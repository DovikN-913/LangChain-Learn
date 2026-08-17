"""
知识点：ToolRuntime 读取短期状态

官方文档：
https://docs.langchain.com/oss/python/langchain/tools#access-context

本示例主要学习：
1. runtime 参数会被注入，模型在 schema 里看不到它。
2. runtime.state 是当前对话的短期记忆。
3. 不要把参数命名为 config 或 runtime，这两个名字被保留。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 工程实践：本文件在 Tools/ 子目录。Python 默认找不到项目根的 utils.py，
# 必须把根目录插入 sys.path，后面才能 `from utils import ...`。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.agents import AgentState, create_agent
from langchain.tools import ToolRuntime, tool

from utils import get_chat_model, print_stage


# 官方文档：继承 AgentState 才能加业务字段。默认 state 几乎只有 messages。
class WeatherPrefState(AgentState):
    home_city: str


# 官方文档：ToolRuntime 由运行时注入，不会出现在发给模型的 schema 里。
# 模型看到的是「无参工具」；真正执行时才能读 runtime.state / context / store。
# 源码分析：参数名不能叫 config 或 runtime（保留名，会和注入机制冲突）。类型用 ToolRuntime。
# 旧写法 InjectedState / get_runtime() 应迁到 ToolRuntime。
@tool
def remember_home_city(runtime: ToolRuntime) -> str:
    """查看当前会话里记录的常驻城市。"""
    city = runtime.state.get("home_city") or "未设置"
    return f"当前记录的常驻城市是：{city}"


def main() -> None:
    # state_schema= 告诉 create_agent：state 里除了 messages 还有 home_city。
    agent = create_agent(
        model=get_chat_model(),
        tools=[remember_home_city],
        state_schema=WeatherPrefState,
        system_prompt="如果用户问常驻城市，调用工具读取状态，不要猜测。",
    )

    print_stage("把 home_city 放进初始 state")
    # 工程实践：home_city 走 state，不让模型猜、也不占工具参数。
    # 对比 context：context 是本轮只读配置（user_id）；state 是本轮可变数据。
    result = agent.invoke(
        {
            "messages": [{"role": "user", "content": "我的常驻城市是哪？"}],
            "home_city": "贵阳",
        }
    )
    print(result["messages"][-1].content)

    print_stage("知识点总结")
    print(
        "state：本轮对话可变数据。\n"
        "context：调用时传入的只读配置，例如 user_id。\n"
        "store：跨会话长期记忆，本目录不展开。"
    )


if __name__ == "__main__":
    main()
