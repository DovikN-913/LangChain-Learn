"""
知识点：用 Pydantic / JSON Schema 定义复杂工具参数

官方文档：
https://docs.langchain.com/oss/python/langchain/tools#advanced-schema-definition

本示例主要学习：
1. 多字段、枚举、默认值时，用 Pydantic 比纯类型注解更清晰。
2. Field(description=...) 会进入发给模型的 schema。
3. JSON Schema 适合动态生成、跨语言对齐。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

# 工程实践：本文件在 Tools/ 子目录。Python 默认找不到项目根的 utils.py，
# 必须把根目录插入 sys.path，后面才能 `from utils import ...`。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.tools import tool

from utils import get_weather, print_stage


class WeatherInput(BaseModel):
    """天气查询入参。"""

    # 官方文档：Field(description=...) 会写进发给模型的 JSON schema，比只写类型更管用。
    city: str = Field(description="城市中文名，例如贵阳、北京")
    include_forecast: bool = Field(
        default=True,
        description="是否在结果中保留多天预报",
    )
    # 源码分析：Literal 会变成 JSON Schema 的 enum，模型只能填列出的值。
    units: Literal["celsius"] = Field(
        default="celsius",
        description="温度单位。当前上游接口固定摄氏度",
    )


# 官方文档：args_schema= 用完整 Pydantic 模型替代「从类型注解推断」。
# 1-2 个简单参数用类型注解就够；枚举、嵌套、校验、字段说明要上 Pydantic。
# 函数签名仍要能接住这些字段，否则运行时对不上。
@tool(args_schema=WeatherInput)
def lookup_weather(
    city: str,
    include_forecast: bool = True,
    units: str = "celsius",
) -> str:
    """查询城市天气。需要真实气象数据时必须调用本工具，不要编造。"""
    raw = get_weather(city)
    if include_forecast:
        return raw
    # 工程实践：工具内部可以裁剪字段，减少塞进模型上下文的噪音。
    import json

    data = json.loads(raw)
    data.pop("forecast", None)
    data.pop("life_indices", None)
    return json.dumps(data, ensure_ascii=False)


def main() -> None:
    print_stage("查看生成的 schema")
    # 官方文档：model_json_schema() 就是 bind_tools 时塞进请求的那份参数定义。
    print(lookup_weather.args_schema.model_json_schema())

    print_stage("按 schema 调用真实天气接口")
    # include_forecast=False 走裁剪分支；返回值仍是字符串，给模型读。
    print(lookup_weather.invoke({"city": "贵阳", "include_forecast": False})[:500])

    print_stage("知识点总结")
    print(
        "简单 1-2 个参数：函数类型注解就够。\n"
        "枚举、嵌套、校验：上 Pydantic。\n"
        "运行时才知道字段：用 JSON Schema。"
    )


if __name__ == "__main__":
    main()
