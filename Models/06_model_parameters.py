"""
知识点：模型参数与连接韧性

官方文档：
https://docs.langchain.com/oss/python/langchain/models#parameters

本示例主要学习：
1. temperature：采样随机性。越低越确定（抽取、分类、工具调用）；越高越发散（文案）。
2. max_tokens：限制本次生成长度（含可见文本；思考模型还可能另计 reasoning token）。
3. enable_thinking：百炼 Qwen 的厂商扩展，经 extra_body 传递，不是 LangChain 标准参数。
   开启后 usage_metadata 里会出现 reasoning token，一次简单问答也可能很贵。
4. usage_metadata：标准化 token 用量（input / output / total，以及 reasoning 等明细），
   用来计费和监控。response_metadata 才是厂商原始字段，不要写死键名。

连接韧性（timeout / max_retries）在 10_error_handling.py 里对照异常看。
官方默认约重试 6 次，会重试网络错误、429、5xx，不会重试 401/404。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 子目录默认找不到项目根，插入后才能 import utils
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import get_chat_model, print_stage


def main() -> None:
    prompt = "用一个词形容今天的心情，不要解释。"

    print_stage("低温度：更确定")
    # 官方文档：temperature 控制采样。0 表示几乎总选最高概率 token，同一 prompt 更稳。
    # 工程实践：工具调用、结构化抽取、分类用低温度，减少胡编参数。
    deterministic = get_chat_model(temperature=0)
    print(deterministic.invoke(prompt).content)

    print_stage("较高温度：更发散")
    # 同一 prompt、更高温度：分布更平，用词更活，但可复现性变差。
    creative = get_chat_model(temperature=0.9)
    print(creative.invoke(prompt).content)

    print_stage("限制生成长度")
    # 官方文档：max_tokens 是本次补全的上限，到顶就会截断。
    # 源码分析：它会写入请求体的 max_tokens；截断不等于「模型拒绝扩写」，只是被掐断。
    short = get_chat_model(max_tokens=16)
    print(short.invoke("请把「春眠不觉晓」扩写成 200 字。").content)

    print_stage("开启 thinking（观察 usage）")
    # 源码分析：Qwen3.6 默认带思考。utils.get_chat_model(enable_thinking=True)
    # 会把 extra_body.enable_thinking 设为 True，请求体里多这个厂商字段。
    # 官方文档：用量看 AIMessage.usage_metadata，不要去解析 response_metadata 的私有结构。
    # 开启后 output 里常含 reasoning token，账单会明显高于关 thinking 的同样问题。
    thinking = get_chat_model(enable_thinking=True)
    response = thinking.invoke("1+1 为什么等于 2？一句话回答。")
    print("text:", response.content)
    print("usage:", response.usage_metadata)

    print_stage("知识点总结")
    print(
        "temperature 低 → 抽取、分类、工具调用更稳。\n"
        "temperature 高 → 文案、头脑风暴。\n"
        "max_retries 默认约 6 次；长任务/不稳网络可调到 10-15。\n"
        "timeout 是单次请求等待上限，超时会触发重试或失败。"
    )


if __name__ == "__main__":
    main()
