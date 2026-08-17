"""
知识点：实战 — 把真实天气收成结构化出行建议

官方文档：
https://docs.langchain.com/oss/python/langchain/structured-output

本示例主要学习：
1. 工具拿原始数据，response_format 约束最终业务对象。
2. 前端可以直接渲染 Pydantic 字段，不必再解析散文。
3. 天气数据仍然来自 utils.get_weather。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

# 工程实践：本文件在「Structured output」子目录。Python 默认只从该目录找模块，
# 把项目根插入 sys.path 后才能 `from utils import ...`。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

from utils import get_chat_model, get_weather, print_stage


class TravelWeather(BaseModel):
    """给出行 App 用的天气卡片。"""

    city: str = Field(description="城市")
    condition: str = Field(description="当前天气现象")
    temperature_c: int = Field(description="当前气温")
    need_umbrella: bool = Field(description="是否建议带伞")
    aqi: int | None = Field(description="空气质量指数，未知则为 null")
    advice: str = Field(description="一句给用户的建议")
    risk: Literal["low", "medium", "high"] = Field(description="出行风险")


def query_weather(city: str) -> str:
    """查询指定城市的实时天气和预报。必须据此填写结构化结果，禁止编造。"""
    return get_weather(city)


def main() -> None:
    agent = create_agent(
        model=get_chat_model(),
        tools=[query_weather],
        # 官方文档：既有业务工具又有 response_format 时，模型必须支持「工具 + 结构化」一起用。
        # 工程实践：百炼用 ToolStrategy——先 query_weather 拿事实，再强制填 TravelWeather。
        # 不行就拆成两步：先 Agent 查天气，再 model.with_structured_output。
        response_format=ToolStrategy(TravelWeather),
        system_prompt="先调用天气工具，再给出结构化出行建议。",
    )

    print_stage("结构化天气卡片")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "我待会要在贵阳出门，帮我做一张天气卡片。"}]}
    )
    # 前端渲染用 structured_response，不要去解析最后一条 AIMessage 的散文。
    card = result["structured_response"]
    print(card.model_dump_json(indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
