"""
知识点：Pydantic vs TypedDict vs JSON Schema

官方文档：
https://docs.langchain.com/oss/python/langchain/structured-output#response-format

本示例主要学习：
1. 三种 schema 都能用，返回形态不同。
2. Pydantic 有校验；TypedDict / JSON Schema 得到 dict。
3. JSON Schema 必须包进显式 Strategy，不能裸传给 response_format。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TypedDict

from pydantic import BaseModel, Field

# 工程实践：本文件在「Structured output」子目录。Python 默认只从该目录找模块，
# 把项目根插入 sys.path 后才能 `from utils import ...`。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

from utils import get_chat_model, print_stage


class Movie(BaseModel):
    title: str = Field(description="片名")
    year: int = Field(description="上映年份")
    director: str = Field(description="导演")


class MovieDict(TypedDict):
    title: str
    year: int
    director: str


MOVIE_JSON_SCHEMA = {
    "title": "Movie",
    "description": "电影信息",
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "片名"},
        "year": {"type": "integer", "description": "上映年份"},
        "director": {"type": "string", "description": "导演"},
    },
    "required": ["title", "year", "director"],
}


def run(schema, label: str) -> None:
    print_stage(label)
    agent = create_agent(
        model=get_chat_model(),
        tools=[],
        # 官方文档：JSON Schema 字典必须包进显式 Strategy，不能裸传给 response_format。
        # 源码分析：裸传 dict 会进 AutoStrategy，再可能被选成 ProviderStrategy；
        # 百炼兼容接口上 Provider 会失败。三种形态都包 ToolStrategy，对比才公平。
        response_format=ToolStrategy(schema),
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "盗梦空间，2010，诺兰导演"}]}
    )
    print(type(result["structured_response"]).__name__, result["structured_response"])


def main() -> None:
    run(Movie, "Pydantic → 模型实例")
    run(MovieDict, "TypedDict → dict")
    run(MOVIE_JSON_SCHEMA, "JSON Schema → dict（必须包 Strategy）")

    print_stage("如何选择")
    print(
        "要校验、默认值、嵌套模型：Pydantic。\n"
        "只要 dict、想少依赖：TypedDict。\n"
        "schema 来自外部系统：JSON Schema + 显式 Strategy。"
    )


if __name__ == "__main__":
    main()
