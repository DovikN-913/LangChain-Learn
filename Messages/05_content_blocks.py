"""
知识点：content 与标准 content_blocks

官方文档：
https://docs.langchain.com/oss/python/langchain/messages#message-content

本示例主要学习：
1. content 是松散类型：字符串、厂商原生块列表、或 LangChain 标准块。
   这是为了兼容多模态和厂商私有结构，不要假设它永远是 str。
2. content_blocks 是跨厂商的标准视图（text / image / reasoning / tool_call 等），
   不是替换 content。指定 content_blocks 初始化时仍会填充 content，但类型更安全。
3. 多模态输入可用 OpenAI 原生格式（image_url）或 LangChain 标准块（type=image）。
4. 业务展示优先用 .content；需要推理块、引用、图片时看 content_blocks。
   当前教学模型是文本模型，下面的图片只演示结构，不会真正拉图。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 子目录默认找不到项目根，插入后才能 import utils
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.messages import AIMessage, HumanMessage

from utils import get_chat_model, print_stage


def main() -> None:
    print_stage("字符串 content")
    # 最常见形态：纯文本。role=human，content 就是这句话。
    human = HumanMessage("你好")
    print(human.content)

    print_stage("OpenAI 原生多模态格式（结构演示，不一定真的拉图）")
    # 官方文档：provider-native 格式可以直接放进 content。
    # 这是 Chat Completions 的 image_url 结构；对接已有 OpenAI 多模态代码时用它。
    # 和标准块的区别：键名跟厂商走（image_url vs type=image），换厂商可能要改。
    multimodal = HumanMessage(
        content=[
            {"type": "text", "text": "描述这张图"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/demo.png"},
            },
        ]
    )
    print(multimodal.content)

    print_stage("标准 content_blocks")
    # 官方文档：指定 content_blocks 时仍会填充 content，但类型更安全。
    # 源码分析：标准块用 type="image" + url，LangChain 再按目标厂商转成原生格式。
    # 应用层写标准块，比手写 OpenAI 结构更不容易绑死一家。
    standard = HumanMessage(
        content_blocks=[
            {"type": "text", "text": "描述这张图"},
            {"type": "image", "url": "https://example.com/demo.png"},
        ]
    )
    print("content:", standard.content)
    print("content_blocks:", standard.content_blocks)

    print_stage("真实模型返回的 content_blocks")
    # 输出侧同样有 content_blocks：把思考过程、文本、工具调用块统一成标准类型。
    model = get_chat_model()
    ai: AIMessage = model.invoke("只说：收到")
    print(ai.content_blocks)

    print_stage("知识点总结")
    print(
        "读写业务文本优先用 .content。\n"
        "需要推理块、引用、图片时看 content_blocks。\n"
        "当前 Qwen 文本模型对本地图片支持有限，多模态要换视觉模型。"
    )


if __name__ == "__main__":
    main()
