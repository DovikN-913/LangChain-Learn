"""
知识点：values / output / interleave

官方文档：
https://docs.langchain.com/oss/python/langchain/event-streaming#state-and-final-output
https://docs.langchain.com/oss/python/langchain/event-streaming#multiple-projections

本示例主要学习：
1. stream.values 是过程中的状态快照。
2. stream.output 是跑完后的最终 state。
3. 同步代码用 interleave 同时消费多个投影。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 工程实践：本文件在「Event streaming」子目录。Python 默认只从该目录找模块，
# 把项目根插入 sys.path 后才能 `from utils import ...`。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.agents import create_agent

from utils import get_chat_model, get_weather, print_stage


def query_weather(city: str) -> str:
    """查询指定城市的天气。"""
    return get_weather(city)


def main() -> None:
    agent = create_agent(
        model=get_chat_model(),
        tools=[query_weather],
        system_prompt="必须调用工具查询真实天气，回答简短。",
    )
    stream = agent.stream_events(
        {"messages": [{"role": "user", "content": "贵阳气温多少？"}]},
        version="v3",
    )

    print_stage("interleave messages + tool_calls + values")
    # 官方文档：values = 过程快照；output = 最终 state。同步用 interleave 同时订多个投影。
    # 源码分析：三个投影共用同一条拉取泵。先 for messages 再 for values 会把图泵干，values 为空。
    # 和 Streaming 的 updates 相近的是 values（整份 state），不是节点增量。
    for name, item in stream.interleave("messages", "tool_calls", "values"):
        if name == "messages":
            text = str(item.content)
            if text:
                print("[messages]", text[:80])
        elif name == "tool_calls":
            print("[tool_calls]", item.tool_name, item.input)
        elif name == "values":
            msgs = item.get("messages") if isinstance(item, dict) else None
            print("[values] messages=", len(msgs) if msgs is not None else item)

    print_stage("output")
    # 读 output 会把尚未泵完的图跑完。本例 interleave 已经耗尽，这里拿到的就是终态。
    print(stream.output["messages"][-1].content)


if __name__ == "__main__":
    main()
