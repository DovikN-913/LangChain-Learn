"""
知识点：实战天气工具（改造自项目根目录 utils.py）

官方文档：
https://docs.langchain.com/oss/python/langchain/tools

本示例主要学习：
1. 真实业务工具 = 外部 API + 清晰 schema + 稳定的字符串返回。
2. 不要在工具里 print 当作返回值。
3. 把数据访问留在 utils，把 LangChain 装饰器留在工具层。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# 工程实践：本文件在 Tools/ 子目录。Python 默认找不到项目根的 utils.py，
# 必须把根目录插入 sys.path，后面才能 `from utils import ...`。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.tools import tool

from utils import get_weather, print_stage


# 官方文档：类型注解 city: str 生成参数 schema；docstring 告诉模型「问天气时必须调，禁止编造」。
# 工程实践：get_weather 必须 return JSON 字符串。print 只方便人看，模型和下一步 ToolMessage 都读不到。
# 数据访问留在 utils，@tool 只负责「何时调用、参数叫什么」，CLI 自测和 Agent 才能共用同一数据源。
@tool
def query_weather(city: str) -> str:
    """查询中国城市的实时天气、空气质量、生活指数和多天预报。

    当用户问「天气怎么样」「会不会下雨」「要不要带伞」时使用。
    不要编造气象数据。

    Args:
        city: 城市名称，例如贵阳、北京、上海。
    """
    return get_weather(city)


def main() -> None:
    print_stage("工具 schema")
    print("name:", query_weather.name)
    print("description:", query_weather.description)

    print_stage("直接调用真实 UAPI")
    raw = query_weather.invoke({"city": "贵阳"})
    data = json.loads(raw)
    if "error" in data:
        print("查询失败:", data)
        return

    print(
        f"{data.get('province')}{data.get('city')} "
        f"{data.get('weather')} {data.get('temperature')}°C "
        f"湿度{data.get('humidity')}% AQI {data.get('aqi')}"
    )
    forecast = data.get("forecast") or []
    print("预报天数:", len(forecast))
    if forecast:
        first = forecast[0]
        print(
            "明天概览:",
            first.get("date"),
            first.get("weather_day"),
            f"{first.get('temp_min')}~{first.get('temp_max')}°C",
        )

    print_stage("空城市 / 异常路径")
    # 工程实践：空字符串能通过 schema（类型是 str），失败发生在业务层。对比 07_tool_errors.py。
    print(query_weather.invoke({"city": ""})[:200])

    print_stage("知识点总结")
    print(
        "utils.get_weather：只负责调 UAPI 并返回 JSON 字符串。\n"
        "@tool query_weather：负责告诉模型何时调用、参数叫什么。\n"
        "两层拆开后，CLI 自测和 Agent 调用可以共用同一数据源。"
    )


if __name__ == "__main__":
    main()
