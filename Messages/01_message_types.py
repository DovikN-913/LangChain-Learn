"""
知识点：消息类型（System / Human / AI / Tool）

官方文档：
https://docs.langchain.com/oss/python/langchain/messages#message-types

本示例主要学习：
1. Message 是模型上下文的基本单位，三条信息缺一不可：
   - role：谁在说话（system / human / ai / tool），由类型决定，也会反映在 .type 上
   - content：说了什么（文本，或块列表）
   - metadata：id、token 用量、tool_calls、厂商附加字段
2. 四种核心类型各自代表谁在说话：System 定规则，Human 是用户，AI 是模型输出，
   Tool 是工具结果。多轮对话就是按顺序堆积这些对象。
3. ToolMessage.tool_call_id 必须匹配对应 AIMessage.tool_calls[].id，对不上会报错或被忽略。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 子目录默认找不到项目根，插入后才能 import utils
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from utils import get_chat_model, print_stage


def main() -> None:
    model = get_chat_model()

    print_stage("四种消息各司其职")
    # 官方文档：SystemMessage 是初始指令（人设、语气、硬规则），不是用户原话。
    # 有的厂商会弱化 system 权重，或要求历史以 user 开头；删历史时不要裁成非法序列。
    system = SystemMessage("你是天气讲解员，只用中文，回答不超过 50 字。")
    # HumanMessage = 用户输入。model.invoke("...") 就是单条 HumanMessage 的快捷写法。
    human = HumanMessage("贵阳会下雨吗？先别查工具，根据常识简要说。")
    print("system.role:", getattr(system, "type", None), system.content)
    print("human.type:", human.type)

    messages = [system, human]
    ai = model.invoke(messages)
    # 官方文档：AIMessage 是模型输出。除 content 外还可能带 tool_calls / usage_metadata。
    print("ai.type:", ai.type)
    print("ai.content:", ai.content)

    print_stage("ToolMessage 必须对齐 tool_call_id")
    # 官方文档：ToolMessage.tool_call_id 必须匹配 AIMessage 里那次 tool call 的 id。
    # 这里手工构造一对，不发真实请求，只看字段如何对齐。
    # 源码分析：发给厂商时，tool 消息的 tool_call_id 必须能在上一条 assistant 的
    # tool_calls 里找到；name 建议带上，部分兼容接口靠它做校验。
    demo_ai = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "query_weather",
                "args": {"city": "贵阳"},
                "id": "call_demo_1",
                "type": "tool_call",
            }
        ],
    )
    demo_tool = ToolMessage(
        content="贵阳多云，23°C",
        tool_call_id="call_demo_1",
        name="query_weather",
    )
    print(demo_ai.tool_calls)
    print(demo_tool)

    print_stage("知识点总结")
    print(
        "System：定人设和规则。\n"
        "Human：用户输入。\n"
        "AI：模型输出，可能带 tool_calls。\n"
        "Tool：工具结果，回填给模型继续推理。"
    )


if __name__ == "__main__":
    main()
