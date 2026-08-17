"""
知识点：AgentState 是 Agent 的执行上下文

官方文档：
https://docs.langchain.com/oss/python/langchain/agents#agent-state

本示例主要学习：
1. 内置字段 messages 是当前 thread 的完整对话，只追加不替换。
2. before_model / after_model 等钩子收到的就是这份 state，返回 dict 会合并回去。
3. 自定义字段通过继承 AgentState + state_schema= 传入。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 工程实践：本文件在 Agents/ 子目录。Python 默认找不到项目根的 utils.py，
# 必须把根目录插入 sys.path，后面才能 `from utils import ...`。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.agents import AgentState, create_agent
from langchain.tools import ToolRuntime, tool

from utils import get_chat_model, get_weather, print_stage


# 官方文档：AgentState 是 TypedDict。内置 messages: list[BaseMessage]，只追加不替换。
# 自定义字段：子类化后经 state_schema= 传入。before_model / after_model 钩子收到的就是这份 state。
class TravelState(AgentState):
    home_city: str
    call_count: int


@tool
def query_home_weather(runtime: ToolRuntime) -> str:
    """查询用户常驻城市天气。城市从 state.home_city 读取。"""
    # 官方文档：ToolRuntime 对模型隐藏；城市从 state 读，不必再占工具参数。
    city = runtime.state.get("home_city") or "贵阳"
    return get_weather(city)


def main() -> None:
    agent = create_agent(
        model=get_chat_model(),
        tools=[query_home_weather],
        state_schema=TravelState,
        system_prompt="用户没提城市时，调用 query_home_weather。回答简短。",
    )

    print_stage("invoke 时写入自定义 state")
    # 源码分析：传入的 messages 追加进历史；home_city / call_count 合并进 state。
    # 本文件没挂 checkpointer，这次 invoke 结束 state 就丢了。跨多次调用见 04。
    result = agent.invoke(
        {
            "messages": [{"role": "user", "content": "我出门要不要带伞？"}],
            "home_city": "贵阳",
            "call_count": 1,
        }
    )
    print(result["messages"][-1].content)
    print("state.home_city =", result.get("home_city"))
    print("messages 条数 =", len(result["messages"]))

    print_stage("知识点总结")
    print(
        "messages：对话历史。\n"
        "自定义字段：给工具/中间件用的结构化数据。\n"
        "要跨多次 invoke 记住它们，还得加 checkpointer，见 04_invoke_thread_context.py。"
    )


if __name__ == "__main__":
    main()
