"""
知识点：stream.tool_calls 观察工具生命周期

官方文档：
https://docs.langchain.com/oss/python/langchain/event-streaming#tool-calls

本示例主要学习：
1. message.tool_calls 是模型「正在写出参数」。
2. stream.tool_calls 是工具「正在执行 / 已返回」。
3. 同步代码要同时看两个投影时，必须用 interleave，不能先后两个 for。
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
        system_prompt="必须调用工具查询真实天气。",
    )
    stream = agent.stream_events(
        {"messages": [{"role": "user", "content": "贵阳现在下雨吗？"}]},
        version="v3",
    )

    print_stage("同步交错消费：参数分片 + 工具执行")
    # 官方文档：同步场景用 interleave；先后迭代两个投影会漏掉后一个。
    # 源码分析：v3 是拉取式。一个投影的 for 会把整张图泵完，另一个投影的队列已经关了。
    # 异步才用 asyncio.gather 并行消费。interleave 按到达时间戳交错，不是 round-robin。
    for name, item in stream.interleave("messages", "tool_calls"):
        if name == "messages":
            for chunk in item.tool_calls:
                print("[model chunk]", chunk)
            finalized = item.tool_calls.get()
            if finalized:
                print("[model finalized]", finalized)
            text = str(item.content)
            if text:
                print("[model text]", text)
        elif name == "tool_calls":
            print(f"[exec] {item.tool_name}({item.input})")
            # 官方文档：先消费 output_deltas，再读最终 output。
            # 源码分析：v3 投影是拉取式 StreamChannel。不先把 deltas 迭代完，
            # 泵可能卡在未消费的通道上，item.output 会一直是 None 或读不到终值。
            for delta in item.output_deltas:
                print("[exec delta]", str(delta)[:200])
            print("[exec output]", str(item.output)[:240])
            print("[exec error]", item.error)


if __name__ == "__main__":
    main()
