"""
知识点：Model.with_structured_output()

官方文档：
https://docs.langchain.com/oss/python/langchain/models#structured-output

本示例主要学习：
1. 结构化输出约束模型按 schema 返回，避免再正则抠文本。
2. Pydantic 带运行时校验，返回模型实例；TypedDict / JSON Schema 返回 dict。
3. method="function_calling" 的原因（工程实践 / 百炼实测）：
   默认 json_schema / json_mode 会走 json_object，兼容接口常 400，还要求 prompt 含 "json"。
   function_calling 把 schema 当成强制工具调用，百炼支持更稳。
4. include_raw=True 同时返回解析结果和原始 AIMessage，便于排错、看 token、看 tool_calls。
5. Field(description=...) 会写进发给模型的 JSON Schema，模型靠它理解字段含义，不是只给人看。

Agent 场景请用 create_agent(response_format=...)，见 Structured output 目录。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

# 子目录默认找不到项目根，插入后才能 import utils
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import get_chat_model, print_stage


class WeatherBrief(BaseModel):
    """从自然语言中抽出的天气摘要。"""

    # 官方文档：Field description 会进入 schema，随工具/结构化请求发给模型。
    # 写清楚取值约定（单位、枚举含义），比把说明堆进用户 prompt 更稳。
    city: str = Field(description="城市名")
    condition: str = Field(description="天气现象，例如多云、小雨")
    temperature_c: int = Field(description="气温，摄氏度")
    suggestion: str = Field(description="一句出行建议")
    severity: Literal["low", "medium", "high"] = Field(
        description="对出行的影响程度"
    )


def main() -> None:
    model = get_chat_model()
    # 工程实践：百炼兼容接口对 json_schema / json_object 支持不完整，
    # 会要求 prompt 里出现 "json" 字样。function_calling 更稳。
    # 源码分析：该方法把 Pydantic 转成 tool schema，强制模型「调用」这个假工具，
    # 再从 tool_calls 的 args 反序列化成 WeatherBrief，而不是解析散文 JSON。
    structured = model.with_structured_output(
        WeatherBrief,
        method="function_calling",
    )

    print_stage("Pydantic 结构化抽取")
    result = structured.invoke(
        "刚才广播说：贵阳现在多云，气温 23 度，空气不错，适合出门，但紫外线较强。"
    )
    print(type(result).__name__, result)
    print("可直接访问字段:", result.city, result.temperature_c)

    print_stage("include_raw=True 同时拿到原始 AIMessage")
    # 官方文档：include_raw=True 时返回 dict，而不是直接的 Pydantic 实例。
    # parsed = 校验后的结构；raw = 原始 AIMessage（content / tool_calls / usage）。
    # 解析失败时仍能从 raw 看到模型实际吐了什么，方便对照 schema。
    structured_raw = model.with_structured_output(
        WeatherBrief,
        method="function_calling",
        include_raw=True,
    )
    packed = structured_raw.invoke("北京今天晴，32 度，注意防暑。")
    print("parsed:", packed["parsed"])
    print("raw type:", type(packed["raw"]).__name__)

    print_stage("知识点总结")
    print(
        "method 常见取值：json_schema / function_calling / json_mode。\n"
        "DashScope 兼容接口优先 function_calling；json_mode 还要求 prompt 含 json。\n"
        "Agent 场景用 ToolStrategy，见 Structured output 目录。"
    )


if __name__ == "__main__":
    main()
