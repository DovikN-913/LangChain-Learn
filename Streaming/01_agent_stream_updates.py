"""
知识点：Agent stream_mode="updates" 看每一步状态

官方文档：
https://docs.langchain.com/oss/python/langchain/streaming#agent-progress

本示例主要学习：
1. Streaming 页讲的是 Pregel 底层 stream 模式。
2. updates 在每个节点结束后推送状态增量。
3. 一次工具调用通常会看到 model → tools → model 三步。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 工程实践：本文件在 Streaming 子目录。Python 默认只从该目录找模块，
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

    print_stage("按节点打印 updates")
    # 官方文档：updates = 每个图节点跑完后的状态增量，不是 token。
    # 一次工具调用通常看到：model（带 tool_calls）→ tools（ToolMessage）→ model（最终文本）。
    # 源码分析：version="v2" 时无论订几种 mode，chunk 都是 StreamPart：type / ns / data。
    # 不传 version 默认 v1，可能是 (mode, data) 元组，按 chunk["type"] 去读会对不上。
    # 和 messages 的区别：updates 按步骤，适合进度条/调试节点；messages 才是打字机。
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "贵阳现在天气如何？"}]},
        stream_mode="updates",
        version="v2",
    ):
        payload = chunk["data"] if isinstance(chunk, dict) and "data" in chunk else chunk
        print("chunk type:", type(chunk).__name__, list(payload) if isinstance(payload, dict) else payload)
        if isinstance(payload, dict):
            for node, update in payload.items():
                messages = update.get("messages") if isinstance(update, dict) else None
                if not messages:
                    continue
                last = messages[-1]
                print(f"node={node} type={last.type} text={getattr(last, 'text', last.content)[:120]}")
        print("-" * 40)


if __name__ == "__main__":
    main()
