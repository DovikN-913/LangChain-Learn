"""
知识点：工具参数错误与上游 API 失败

官方文档：
https://docs.langchain.com/oss/python/langchain/tools

本示例主要学习：
1. schema 校验失败发生在真正执行前。
2. 上游天气 API 失败时应返回可读错误，而不是让 Agent 崩溃。
3. 空城市、错误城市名是真实业务里最常见的边界。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# 工程实践：本文件在 Tools/ 子目录。Python 默认找不到项目根的 utils.py，
# 必须把根目录插入 sys.path，后面才能 `from utils import ...`。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.tools import tool
from pydantic import ValidationError

from utils import get_weather, print_stage


@tool
def query_weather(city: str) -> str:
    """查询城市天气。"""
    # 工程实践：schema 只能保证「有 city 且是 str」。空格字符串能通过校验，必须业务层再挡。
    if not city.strip():
        return json.dumps({"error": "city 不能为空"}, ensure_ascii=False)
    return get_weather(city)


def main() -> None:
    print_stage("schema 校验：缺少必填参数")
    # 官方文档：invoke({}) 缺必填字段，Pydantic 在函数体运行前就 ValidationError。
    # 这和下面「city='   '」不同：后者类型合法，函数已经执行了。
    try:
        query_weather.invoke({})
    except (ValidationError, Exception) as exc:
        print(type(exc).__name__, str(exc)[:300])

    print_stage("业务校验：空字符串")
    # 源码分析：{"city": "   "} 满足 str schema，进函数体后被 strip 判空，返回 error JSON。
    print(query_weather.invoke({"city": "   "}))

    print_stage("上游可能无法识别的地名")
    # 工程实践：UAPI 失败时 get_weather 返回带 error 的 JSON，而不是 raise。
    # Agent 才能把错误读进 ToolMessage，向用户解释；traceback 会直接打断循环。
    print(query_weather.invoke({"city": "火星基地A"})[:400])

    print_stage("知识点总结")
    print(
        "参数错误：尽量让 schema 挡在门外。\n"
        "API 失败：返回 error JSON，让模型向用户解释，而不是 traceback。\n"
        "需要严格失败时再 raise，并在 Agent 层做 handle_tool_errors。"
    )


if __name__ == "__main__":
    main()
