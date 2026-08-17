"""
知识点：AIMessage 元数据（usage / response_metadata / id）

官方文档：
https://docs.langchain.com/oss/python/langchain/messages#ai-message

本示例主要学习：
1. AIMessage 除了文本，还携带 token 用量和厂商原始元数据。一条消息 =
   role（固定为 ai）+ content + metadata（id / usage / response_metadata / tool_calls）。
2. usage_metadata 是跨厂商标准化用量，便于计费和监控。
   典型字段：input_tokens / output_tokens / total_tokens，以及 reasoning 等明细。
3. response_metadata 内容因厂商而异，不要写死字段名；排查兼容接口差异时看它。
4. content_blocks 是把厂商格式懒解析后的标准块视图，不是替换 .content。
   业务展示用 .content；推理块、引用、图片看 content_blocks。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 子目录默认找不到项目根，插入后才能 import utils
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import get_chat_model, print_stage


def main() -> None:
    model = get_chat_model()
    response = model.invoke("用一句话介绍 AIMessage。")

    print_stage("文本内容")
    # 官方文档：.content 是文本/原始 payload 视图，类型可能是 str 或 list，不要默认它永远是 str。
    print("text:", response.content)
    print("content:", response.content)

    print_stage("标准化内容块")
    # 官方文档：content_blocks 会把厂商格式懒解析成统一结构（text / reasoning 等）。
    # 它是只读标准视图；需要跨厂商处理思考过程或工具调用块时用这个，而不是解析 content 字符串。
    print(response.content_blocks)

    print_stage("用量与响应元数据")
    # id：本条消息标识，追踪、去重、对账用。
    # usage_metadata：官方标准化用量。开启 Qwen thinking 后这里会出现 reasoning token。
    # response_metadata：厂商原始字段，键名不稳定，打印 keys 即可，不要依赖某个私有键。
    print("id:", response.id)
    print("usage_metadata:", response.usage_metadata)
    print("response_metadata keys:", list((response.response_metadata or {}).keys()))

    print_stage("知识点总结")
    print(
        "业务展示用 .content。\n"
        "账单和限流看 usage_metadata。\n"
        "排查厂商差异看 response_metadata 和 content_blocks。"
    )


if __name__ == "__main__":
    main()
