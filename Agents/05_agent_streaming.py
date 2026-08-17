"""
知识点：Agent 流式输出中间进度

官方文档：
https://docs.langchain.com/oss/python/langchain/agents#streaming

本示例主要学习：
1. invoke 只在全部跑完后返回；多步工具调用时用户会感觉卡住。
2. 官方 Agents 页推荐 stream_events(version="v3") 看状态快照。
3. 更细的投影（token、工具生命周期）见 Event streaming 目录。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 工程实践：本文件在 Agents/ 子目录。Python 默认找不到项目根的 utils.py，
# 必须把根目录插入 sys.path，后面才能 `from utils import ...`。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage

from utils import get_chat_model, get_weather, print_stage


def query_weather(city: str) -> str:
    """查询指定城市的天气。"""
    return get_weather(city)


def main() -> None:
    agent = create_agent(
        model=get_chat_model(),
        tools=[query_weather],
        system_prompt="必须调用工具查询真实天气。",
        name="weather_agent",
    )

    print_stage("按 values 快照观察循环")
    # 官方文档：Agents 页示例用 stream_events(..., version="v3")。
    # version="v3" 返回带投影的 run 对象；stream.values 是每次状态更新后的完整快照。
    # 和 invoke 的区别：invoke 跑完才返回；values 能看到「正在调工具 / 已有最终回答」。
    # 更细的 token / 工具生命周期见 Event streaming 目录，不要和 stream_mode="updates" 混用。
    stream = agent.stream_events(
        {"messages": [{"role": "user", "content": "贵阳今天天气怎样？"}]},
        version="v3",
    )
    for snapshot in stream.values:
        latest = snapshot["messages"][-1]
        if latest.tool_calls:
            print("Calling tools:", [tc["name"] for tc in latest.tool_calls])
        elif isinstance(latest, HumanMessage):
            print("User:", latest.content)
        elif isinstance(latest, AIMessage) and latest.content:
            print("Agent:", latest.content)

    print_stage("最终回答")
    # 官方文档：stream.output 是跑完后的最终 state，等价于 invoke 的返回值。
    print(stream.output["messages"][-1].content)


if __name__ == "__main__":
    main()
