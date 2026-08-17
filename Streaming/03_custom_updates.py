"""
知识点：工具内 get_stream_writer 推送自定义进度

官方文档：
https://docs.langchain.com/oss/python/langchain/streaming#custom-updates

本示例主要学习：
1. custom 模式用来推「查到一半了」这类业务进度。
2. writer 只能在 LangGraph 运行中使用。
3. 可以同时订阅 updates + custom。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 工程实践：本文件在 Streaming 子目录。Python 默认只从该目录找模块，
# 把项目根插入 sys.path 后才能 `from utils import ...`。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.agents import create_agent
from langgraph.config import get_stream_writer

from utils import get_chat_model, get_weather, print_stage


def query_weather(city: str) -> str:
    """查询指定城市的天气。执行过程中会推送进度。"""
    # 官方文档：writer 把任意数据推到 stream_mode="custom"。
    # 源码分析：get_stream_writer 从 contextvar 里取当前图的 runtime.stream_writer。
    # 图外单独 invoke 这个工具时 contextvar 为空，会报错——工具一旦用了 writer，就不能当普通函数测。
    writer = get_stream_writer()
    writer(f"开始查询 {city} 天气…")
    result = get_weather(city)
    writer(f"{city} 天气数据已返回")
    return result


def main() -> None:
    agent = create_agent(
        model=get_chat_model(),
        tools=[query_weather],
        system_prompt="必须调用工具查询真实天气。",
    )

    print_stage("同时看节点更新和自定义进度")
    # 官方文档：stream_mode 可传 list，同时订多种模式。
    # 源码分析：v2 每个 chunk 是 {type, ns, knowledge_docs}，用 chunk["type"] 区分，不必解包 (mode, knowledge_docs)。
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "查一下贵阳天气"}]},
        stream_mode=["updates", "custom"],
        version="v2",
    ):
        kind = chunk.get("type") if isinstance(chunk, dict) else None
        data = chunk.get("knowledge_docs") if isinstance(chunk, dict) else chunk
        if kind == "custom":
            print("[custom]", data)
        elif kind == "updates":
            print("[updates]", list(data) if isinstance(data, dict) else data)


if __name__ == "__main__":
    main()
