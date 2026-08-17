"""
知识点：stream_mode="messages" 流式输出 token

官方文档：
https://docs.langchain.com/oss/python/langchain/streaming#llm-tokens

本示例主要学习：
1. messages 模式产出 (token, metadata) 元组。
2. 可以看到工具参数 JSON 是一点点拼出来的。
3. 适合做聊天窗口的打字机效果。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 工程实践：本文件在 Streaming 子目录。Python 默认只从该目录找模块，
# 把项目根插入 sys.path 后才能 `from utils import ...`。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.agents import create_agent
from langchain.messages import AIMessageChunk

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

    print_stage("流式 token / tool_call_chunk")
    # 官方文档：messages 产出 (token, metadata)；metadata 带 langgraph_node。
    # 源码分析：v2 信封是 {"type": "messages", "ns": ..., "data": (token, metadata)}。
    # 先认 type 再解包 data，不要把整个 chunk 当元组。
    # 和 updates 的区别：这里是生成过程中的增量；updates 要等整个节点跑完。
    # 和 Event streaming 的区别：这里自己拆信封；v3 用 stream.messages 投影。
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "贵阳今天天气怎么样？"}]},
        stream_mode="messages",
        version="v2",
    ):
        if not isinstance(chunk, dict) or chunk.get("type") != "messages":
            continue
        token, metadata = chunk["data"]
        node = metadata.get("langgraph_node")
        if isinstance(token, AIMessageChunk):
            if token.content:
                print(f"[{node} text] {token.content}", end="", flush=True)
            if token.tool_call_chunks:
                print(f"\n[{node} tool_chunk] {token.tool_call_chunks}")
    print()


if __name__ == "__main__":
    main()
