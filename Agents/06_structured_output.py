"""
知识点：Agent 的 response_format 结构化出口

官方文档：
https://docs.langchain.com/oss/python/langchain/agents#structured-output

本示例主要学习：
1. 把 schema 传给 create_agent(response_format=...)，结果在 structured_response。
2. 百炼兼容接口请包 ToolStrategy，不要依赖自动 ProviderStrategy。
3. 可以先调天气工具，再收成业务对象。
"""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic import BaseModel, Field

# 工程实践：本文件在 Agents/ 子目录。Python 默认找不到项目根的 utils.py，
# 必须把根目录插入 sys.path，后面才能 `from utils import ...`。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

from utils import get_chat_model, get_weather, print_stage


class OutingAdvice(BaseModel):
    city: str = Field(description="城市")
    condition: str = Field(description="天气现象")
    temperature_c: int = Field(description="气温")
    need_umbrella: bool = Field(description="是否带伞")
    advice: str = Field(description="一句建议")


def query_weather(city: str) -> str:
    """查询真实天气。填写结构化结果前必须调用，禁止编造。"""
    return get_weather(city)


def main() -> None:
    # 官方文档：response_format= 让 Agent 返回校验后的 schema，结果在 result["structured_response"]。
    # 工程实践：百炼兼容接口对原生 json_schema 支持不完整，显式 ToolStrategy，不要依赖自动 ProviderStrategy。
    # 源码分析：ToolStrategy 用强制工具调用「填表」；ProviderStrategy 走厂商原生 structured output。
    agent = create_agent(
        model=get_chat_model(),
        tools=[query_weather],
        response_format=ToolStrategy(OutingAdvice),
        system_prompt="先调用天气工具，再给出结构化出行建议。",
    )

    print_stage("structured_response")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "我待会要在贵阳出门。"}]}
    )
    print(result["structured_response"].model_dump_json(ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
