"""
知识点：字符串 vs 消息对象 vs dict

官方文档：
https://docs.langchain.com/oss/python/langchain/messages#basic-usage

本示例主要学习：
1. 三种输入最终都会变成 messages，再发给 Chat Model。
2. 底层差异：
   - 字符串：内部等于一条 HumanMessage。单轮、无历史、代码最少。
   - 消息对象：LangChain 原生类型，可带 SystemMessage、历史 AI/Tool、tool_calls、多模态块。
   - OpenAI 风格 dict：{"role","content"}，role 用 system/user/assistant/tool，
     先转成 Message 对象再走同一请求路径。方便对接已有 OpenAI SDK 代码。
3. 多轮、多模态、系统提示必须用消息列表（对象或 dict），不能再用裸字符串，
   否则上一轮内容根本不会出现在请求里。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 子目录默认找不到项目根，插入后才能 import utils
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.messages import HumanMessage, SystemMessage

from utils import get_chat_model, print_stage


def main() -> None:
    model = get_chat_model()

    print_stage("字符串：内部等于一条 HumanMessage")
    # 官方文档：str 是单轮快捷写法。源码分析：invoke 把它包装成 [HumanMessage(...)]。
    # 没有地方放 SystemMessage，也回传不了历史。
    print(model.invoke("用四个字形容 LangChain。").content)

    print_stage("消息对象：可带 SystemMessage，适合多轮")
    # 官方文档：要保留历史、加系统提示、处理 tool_calls 时用消息对象列表。
    # 和字符串的区别：role 由类型携带，字段（tool_calls、id、name）不会在转换中丢失。
    print(
        model.invoke(
            [
                SystemMessage("只输出四个字，不要标点。"),
                HumanMessage("用四个字形容 LangChain。"),
            ]
        ).content
    )

    print_stage("dict：对接已有 OpenAI 风格代码")
    # 官方文档：role 必须用 OpenAI 名（user 而不是 human，assistant 而不是 ai）。
    # 和消息对象的区别：dict 是传输格式，不好直接挂 usage_metadata / artifact；
    # 进模型前仍会转成 Message。接口层已经是 role/content 时用它最省事。
    print(
        model.invoke(
            [
                {"role": "system", "content": "只输出四个字。"},
                {"role": "user", "content": "用四个字形容 Agent。"},
            ]
        ).content
    )

    print_stage("如何选择")
    print(
        "选字符串：一次性问答、最小代码。\n"
        "选消息对象：要保留历史、加系统提示、处理 tool_calls。\n"
        "选 dict：从 OpenAI SDK 迁移，或接口层已经是 role/content。"
    )


if __name__ == "__main__":
    main()
