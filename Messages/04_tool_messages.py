"""
知识点：ToolMessage 如何把工具结果接回对话

官方文档：
https://docs.langchain.com/oss/python/langchain/messages#tool-message

本示例主要学习：
1. 模型发起 tool_calls 后，必须回一条对应的 ToolMessage，多轮历史才能合法。
2. tool_call_id 必须等于对应 tool call 的 id；对不上，多数厂商会报错或忽略结果。
3. 字段分工：
   - content：给模型看的文本（建议摘要，避免把超大 JSON 塞进上下文）
   - name：工具名
   - artifact：给程序看、不塞进模型上下文的附加数据（原始 JSON、document_id 等）
4. 本文件用 invoke(call["args"]) 拿到原始字符串，再手工构造 ToolMessage，
   是为了演示 artifact。若只想自动带 id，可用 tool.invoke(tool_call)，见 Models/07。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# 子目录默认找不到项目根，插入后才能 import utils
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import tool

from utils import get_chat_model, get_weather, print_stage


@tool
def query_weather(city: str) -> str:
    """查询指定城市的实时天气。"""
    return get_weather(city)


def main() -> None:
    model = get_chat_model().bind_tools([query_weather])
    messages: list = [HumanMessage("查一下贵阳现在的天气，用两句话总结。")]

    print_stage("模型发出工具调用")
    ai_msg = model.invoke(messages)
    # 带 tool_calls 的 AIMessage 必须留在历史里，否则下一条 ToolMessage 没有对应调用。
    messages.append(ai_msg)
    print("tool_calls:", ai_msg.tool_calls)

    if not ai_msg.tool_calls:
        print("未触发工具。模型原文：", ai_msg.content)
        return

    print_stage("执行工具并构造 ToolMessage")
    for call in ai_msg.tool_calls:
        # 只传 args → 返回工具原始 str，不会自动变成 ToolMessage。
        # 和 Models/07 里 tool.invoke(tool_call) 的区别：那边会包成带 tool_call_id 的消息。
        raw = query_weather.invoke(call["args"])
        # 工程实践：content 给模型看摘要；artifact 给日志/前端看原始 JSON。
        # 源码分析：artifact 留在 Python 对象上，序列化给厂商时通常不会进入 messages 正文，
        # 因此不会占用模型上下文窗口。
        parsed = json.loads(raw)
        summary = (
            f"{parsed.get('city')} {parsed.get('weather')} "
            f"{parsed.get('temperature')}°C"
            if "error" not in parsed
            else raw
        )
        tool_msg = ToolMessage(
            content=summary,
            tool_call_id=call["id"],
            name=call["name"],
            artifact=parsed,
        )
        print("给模型的 content:", tool_msg.content)
        print("artifact.city:", (tool_msg.artifact or {}).get("city"))
        messages.append(tool_msg)

    print_stage("模型基于工具结果作答")
    # 完整回传：Human → AI(tool_calls) → Tool(对齐 id)。缺一段模型看不到天气。
    final = model.invoke(messages)
    print(final.content)


if __name__ == "__main__":
    main()
