"""
知识点：invoke() 同步调用

官方文档：
https://docs.langchain.com/oss/python/langchain/models#invoke

本示例主要学习：
1. invoke 一次请求，阻塞直到模型生成完，返回完整 AIMessage（不是 str）。
2. 输入可以是纯字符串、消息对象列表、OpenAI 风格 dict，底层都会变成 messages。
3. 三种输入的底层差异：
   - 字符串：快捷写法，内部包装成单条 HumanMessage。没有 System、没有历史。
   - 消息对象：LangChain 原生类型，可混 System/Human/AI/Tool，能保留 tool_calls 等字段。
   - dict：按 OpenAI role 名（system/user/assistant/tool）转换，方便对接已有 SDK 代码。
4. `.content` 是文本视图；返回类型本身是 AIMessage。若拿到 str，说明误用了旧式 LLM。

适合脚本、notebook、必须等全文再处理的下游。实时打字机效果请看 stream。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 子目录默认找不到项目根，插入后才能 import utils
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.messages import AIMessage, HumanMessage, SystemMessage

from utils import get_chat_model, print_stage


def main() -> None:
    model = get_chat_model()

    print_stage("1. 字符串输入")
    # 官方文档：字符串是单轮任务的快捷写法。
    # 源码分析：invoke 入口会把 str 转成 [HumanMessage(content=...)] 再发给厂商。
    # 和消息列表的区别：这里无法附带 SystemMessage，也无法回传上一轮 AI/Tool。
    response = model.invoke("用一句话解释什么是 Chat Model。")
    print(type(response).__name__)
    print(response.content)

    print_stage("2. 消息对象列表")
    # 官方文档：多轮、系统提示、工具结果都必须用消息列表，不能再用裸字符串。
    # 底层：对象原样序列化进 Chat Completions 的 messages 字段，role 由类型决定。
    # 故意放一条 AIMessage 进去，演示「历史里可以有模型自己说过的话」。
    conversation = [
        SystemMessage("你是简洁的中文助教，回答不超过 40 字。"),
        HumanMessage("LangChain 的 Model 组件解决什么问题？"),
        AIMessage("它把不同厂商的大模型统一成同一套调用接口。"),
        HumanMessage("那 invoke 呢？"),
    ]
    response = model.invoke(conversation)
    print(response.content)

    print_stage("3. OpenAI 风格 dict")
    # 官方文档：也可以直接传 {"role","content"} 字典，方便对接已有 OpenAI 代码。
    # 源码分析：dict 会先转成对应 Message 对象（user→HumanMessage, system→SystemMessage），
    # 再走和方式 2 相同的请求路径。role 必须用 OpenAI 名（user 而不是 human）。
    # 和消息对象的区别：dict 不好携带 tool_calls、usage_metadata 这类结构化字段。
    response = model.invoke(
        [
            {"role": "system", "content": "只输出一个单词。"},
            {"role": "user", "content": "天空通常是什么颜色？"},
        ]
    )
    print(type(response))
    print(response)
    print(response.content)

    print_stage("结果说明")
    print(
        "invoke 返回 AIMessage，不是 str。\n"
        "如果拿到的是纯字符串，说明误用了旧式 LLM，而不是 Chat Model。"
    )


if __name__ == "__main__":
    main()
