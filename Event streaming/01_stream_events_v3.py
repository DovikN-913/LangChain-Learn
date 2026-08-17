"""
知识点：stream_events(version="v3") 入门

官方文档：
https://docs.langchain.com/oss/python/langchain/event-streaming

本示例主要学习：
1. v3 Event Streaming 返回带投影的 run 对象，不必自己拆 stream_mode 元组。
2. stream.messages 专门消费模型输出。
3. stream.output 是最终状态。
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
        system_prompt="必须调用工具查询真实天气，最终用中文简要回答。",
    )

    print_stage("v3：按 message 投影流式打印")
    # 官方文档：大多数应用/前端应使用 Event Streaming。返回 run 对象，投影可独立消费。
    # 源码分析：version="v3" 返回 GraphRunStream，不后台开线程；迭代投影会拉动底层图。
    # 类上标了 @beta，运行时可能看到 BetaWarning：v3 协议目前仍是实验性的。
    # 不要把 v2 事件字典循环和 v3 投影混用；v3 也不接受 stream_mode= 参数。
    stream = agent.stream_events(
        {"messages": [{"role": "user", "content": "贵阳今天天气怎样？"}]},
        version="v3",
    )
    # stream.messages：每次 LLM 调用一条消息流。message.content 是文本增量。
    # 和 Streaming 的 stream_mode="messages" 比：这里不用拆 type/knowledge_docs 信封。
    for message in stream.messages:
        node = getattr(message, "node", "?")
        print(f"\n[{node}] ", end="")
        for delta in message.content:
            print(delta, end="", flush=True)
    print()

    print_stage("最终状态")
    # 官方文档：stream.output 是跑完后的最终 Agent state。
    # 源码分析：读 .output 会把剩余图执行泵完（_drive_until_done），再返回最后一份 state。
    final_state = stream.output
    last = final_state["messages"][-1]
    print(getattr(last, "text", last.content))


if __name__ == "__main__":
    main()
