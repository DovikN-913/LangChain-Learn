"""
知识点：ProviderStrategy vs ToolStrategy

官方文档：
https://docs.langchain.com/oss/python/langchain/structured-output#provider-strategy
https://docs.langchain.com/oss/python/langchain/structured-output#tool-calling-strategy

本示例主要学习：
1. ProviderStrategy 走厂商原生结构化输出，通常更严。
2. ToolStrategy 用强制工具调用模拟结构化输出，兼容面更广。
3. 百炼 OpenAI 兼容接口优先 ToolStrategy，避免原生 json_schema 不齐。
"""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic import BaseModel, Field

# 工程实践：本文件在「Structured output」子目录。Python 默认只从该目录找模块，
# 把项目根插入 sys.path 后才能 `from utils import ...`。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy

from utils import get_chat_model, print_stage


class CityFact(BaseModel):
    city: str = Field(description="城市")
    country: str = Field(description="国家")
    famous_for: str = Field(description="一句著名点")


def main() -> None:
    model = get_chat_model()
    prompt = {
        "messages": [{"role": "user", "content": "介绍贵阳：中国，以避暑和酸汤闻名。"}]
    }

    print_stage("ToolStrategy（本项目默认推荐）")
    # 官方文档：把 schema 当成强制工具。所有支持 tool calling 的模型都能用。
    # 和 Provider 的区别：多一轮「假工具」调用，但可 handle_errors 重试，并支持 Union 多 schema。
    tool_agent = create_agent(
        model=model,
        tools=[],
        response_format=ToolStrategy(CityFact),
    )
    print(tool_agent.invoke(prompt)["structured_response"])

    print_stage("ProviderStrategy（依赖厂商原生能力）")
    # 官方文档：走厂商 native structured output（OpenAI / Anthropic / Gemini 等）。
    # 工程实践：百炼兼容接口实测会走 json_object，报 400（prompt 须含 json）。失败就回退 Tool。
    try:
        provider_agent = create_agent(
            model=model,
            tools=[],
            response_format=ProviderStrategy(CityFact),
        )
        print(provider_agent.invoke(prompt)["structured_response"])
    except Exception as exc:
        print("ProviderStrategy 失败，回退 ToolStrategy 即可：")
        print(type(exc).__name__, str(exc)[:300])

    print_stage("如何选择")
    print(
        "厂商明确支持 structured output（OpenAI/Anthropic/Gemini 等）→ Provider。\n"
        "兼容接口、小模型、或 Provider 失败 → Tool。\n"
        "ToolStrategy.handle_errors 还能在校验失败时自动让模型重试。"
    )


if __name__ == "__main__":
    main()
