"""
知识点：Agent = 模型在循环里调用工具，直到任务完成

官方文档：
https://docs.langchain.com/oss/python/langchain/agents

本示例主要学习：
1. Agent 的本质是「模型 + harness」，harness 负责在正确时间给模型正确上下文。
2. create_agent 是官方推荐的可配置 harness。
3. 循环由框架自动跑：想 → 调工具 → 再想 → 最终回答。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 工程实践：本文件在 Agents/ 子目录。Python 默认找不到项目根的 utils.py，
# 必须把根目录插入 sys.path，后面才能 `from utils import ...`。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.agents import create_agent

from utils import get_chat_model, get_weather, print_stage


def query_weather(city: str) -> str:
    """查询指定城市的实时天气、空气质量和预报。用户问天气时必须调用。"""
    return get_weather(city)


def main() -> None:
    print_stage("1. 创建最简 Agent")
    # 官方文档：Agent = Model + Harness。create_agent 是官方推荐的 harness。
    # 最少只要 model + tools；system_prompt 塑形行为；name 在多 Agent / 子图里标识这条 harness。
    # 源码分析：返回编译后的图。内部循环：模型决定回答或调工具 → 执行并写 ToolMessage → 再交给模型。
    # 旧写法 initialize_agent / AgentExecutor 已不是 1.x 入口。
    agent = create_agent(
        model=get_chat_model(),
        tools=[query_weather],
        system_prompt="你是天气助手。必须调用工具获取真实天气，禁止编造数字。回答简短。",
        name="weather_agent",
    )

    print_stage("2. 一次 invoke：框架自动完成工具循环")
    # 官方文档：invoke 更新 Agent State。messages 只追加，不替换。和 LLM bind_tools 不同：这里不必自己执行工具、回填 ToolMessage。
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "贵阳现在适不适合出门跑步？"}]}
    )
    # 遍历 result 字典中 "messages" 键对应的消息列表
    for msg in result["messages"]:
        # 获取当前消息的类型（如 "human", "ai", "tool" 等）
        kind = msg.type

        # 生成消息内容的预览文本：优先取 msg.text 属性；若为空/不存在，则取 msg.content 属性并转为字符串，截取前80个字符作为预览
        preview = getattr(msg, "text", "") or str(getattr(msg, "content", ""))[:80]

        # 安全地获取消息附带的工具调用列表，若属性不存在则默认为空列表
        tool_calls = getattr(msg, "tool_calls", None) or []

        # 如果存在工具调用，则拼接一个包含所有工具名称的额外信息字符串；否则为空字符串
        extra = f" tool_calls={[c['name'] for c in tool_calls]}" if tool_calls else ""

        # 按格式打印每条消息的摘要：[消息类型] + 工具调用信息(可选) + 内容预览
        print(f"[{kind}]{extra} {preview}")

    print_stage("3. 知识点总结")
    print(
        "bind_tools 只让模型「点菜」，不会将工具调用结果写入 ToolMessage并回传到模型；create_agent 会把菜做完再端上来。\n"
        "name= 在多 Agent / 子图里用来标识这条 harness。"
    )


if __name__ == "__main__":
    main()
