"""
知识点：Model 层的 Tool Calling（bind_tools）

官方文档：
https://docs.langchain.com/oss/python/langchain/models#tool-calling

本示例主要学习：
1. bind_tools 只是把工具 schema（名字、描述、参数）附到模型请求上。
   模型返回的是「调用请求」AIMessage.tool_calls，不会自动执行 Python 函数。
2. 自己执行后必须把 ToolMessage 回传，模型才能基于真实结果写自然语言答案。
3. tool.invoke(tool_call) 传入完整 tool_call 字典（含 id / name / args）时，
   LangChain 会执行函数并把返回值包成带匹配 tool_call_id 的 ToolMessage。
   若只传 args，得到的是原始返回值（通常是 str），要自己拼 ToolMessage。
4. Agent（create_agent）才会自动完成「调用 → 执行 → 再思考」循环；本文件是手工版。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 子目录默认找不到项目根，插入后才能 import utils
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.messages import HumanMessage
from langchain.tools import tool

from utils import get_chat_model, get_weather, print_stage


@tool
def query_weather(city: str) -> str:
    """查询指定城市的实时天气、空气质量和未来几天预报。"""
    return get_weather(city)


def main() -> None:
    model = get_chat_model()
    # 官方文档：bind_tools 返回一个「绑定了 tools 参数」的新 Runnable，不改原 model。
    # 源码分析：请求体会多出 tools / tool_choice 字段；模型最多生成 tool_calls，
    # 运行时不会 import 或调用这个 Python 函数。执行是调用方的责任。
    model_with_tools = model.bind_tools([query_weather])

    print_stage("1. 模型决定是否调用工具")
    messages: list = [HumanMessage("贵阳今天天气怎么样？简要告诉我。")]
    ai_msg = model_with_tools.invoke(messages)
    # 必须把带 tool_calls 的 AIMessage 留在历史里，下一步 ToolMessage 才能对上 id。
    messages.append(ai_msg)
    print("content:", ai_msg.content)
    print("tool_calls:", ai_msg.tool_calls)

    if not ai_msg.tool_calls:
        print("模型没有发起工具调用，直接给出了文本回答。")
        return

    print_stage("2. 执行工具（模型此时还没真正查天气）")
    for tool_call in ai_msg.tool_calls:
        # 官方文档：tool.invoke(tool_call) 传入完整调用块时，会自动带上 tool_call_id，
        # 生成 ToolMessage。底层：用 args 调函数，再用 call["id"] 填 ToolMessage.tool_call_id。
        # 和 query_weather.invoke(tool_call["args"]) 的区别：后者只返回 str，要自己构造消息。
        tool_msg = query_weather.invoke(tool_call)
        print("ToolMessage:", tool_msg)
        messages.append(tool_msg)

    print_stage("3. 把工具结果交回模型，生成最终回答")
    # 历史顺序必须是：Human → AI(tool_calls) → Tool(匹配 id) → 再 invoke。
    # 缺少任何一段，多数厂商会报错或忽略工具结果。
    final = model_with_tools.invoke(messages)
    print(final.content)

    print_stage("知识点总结")
    print(
        "bind_tools：让模型「会点菜」。\n"
        "手动执行：你自己「去做菜」。\n"
        "create_agent：服务员全程代劳。需要自动循环时看 Tools 目录。"
    )


if __name__ == "__main__":
    main()
