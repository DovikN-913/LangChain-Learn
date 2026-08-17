"""
知识点：实战 — 多工具天气出行助手

官方文档：
https://docs.langchain.com/oss/python/langchain/agents

本示例主要学习：
1. 把 model、tools、prompt、checkpointer、retry、结构化输出组合成一条产品向 harness。
2. 天气数据仍来自 utils.get_weather。
3. 这是「学完 Agents 页之后」最接近真实助手的形态。
"""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic import BaseModel, Field

# 工程实践：本文件在 Agents/ 子目录。Python 默认找不到项目根的 utils.py，
# 必须把根目录插入 sys.path，后面才能 `from utils import ...`。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware, ToolRetryMiddleware
from langchain.agents.structured_output import ToolStrategy
from langgraph.checkpoint.memory import InMemorySaver

from utils import get_chat_model, get_weather, print_stage


class TripCard(BaseModel):
    city: str = Field(description="城市")
    today: str = Field(description="今日天气一句话")
    need_umbrella: bool = Field(description="是否带伞")
    clothing: str = Field(description="穿衣建议")


def query_weather(city: str) -> str:
    """查询指定城市的实时天气和预报。"""
    return get_weather(city)


def main() -> None:
    # 工程实践：把本目录学过的 harness 能力叠在一起——
    # model + tools + system_prompt + name + checkpointer + 重试中间件 + ToolStrategy。
    # 官方文档：百炼用 ToolStrategy，不要依赖自动 ProviderStrategy。
    # InMemorySaver 按 thread_id 记住 messages，第二轮「还是这个城市」才能续上。
    agent = create_agent(
        model=get_chat_model(),
        tools=[query_weather],
        checkpointer=InMemorySaver(),
        middleware=[
            ModelRetryMiddleware(max_retries=2),
            ToolRetryMiddleware(max_retries=2),
        ],
        response_format=ToolStrategy(TripCard),
        system_prompt="先查真实天气，再给出结构化出行卡片。用户省略城市时沿用上文。",
        name="trip_assistant",
    )
    config = {"configurable": {"thread_id": "trip-demo"}}

    print_stage("第一轮")
    r1 = agent.invoke(
        {"messages": [{"role": "user", "content": "我今晚在贵阳吃饭，帮我做张出行卡片。"}]},
        config,
    )
    # 结构化结果在 structured_response，不是最后一条 AIMessage。
    print(r1["structured_response"].model_dump_json(ensure_ascii=False, indent=2))

    print_stage("第二轮：同一会话追问")
    r2 = agent.invoke(
        {"messages": [{"role": "user", "content": "那明天还是这个城市，还要带伞吗？"}]},
        config,
    )
    print(r2["structured_response"].model_dump_json(ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
