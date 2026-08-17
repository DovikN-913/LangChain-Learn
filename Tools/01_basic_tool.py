"""
知识点：用 @tool 把普通函数变成 LangChain Tool

官方文档：
https://docs.langchain.com/oss/python/langchain/tools#basic-tool-definition

本示例主要学习：
1. 类型注解决定输入 schema，docstring 决定模型何时调用。
2. 工具名用 snake_case，避免空格和特殊字符。
3. 可以直接 invoke，也可以交给 bind_tools / create_agent。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 工程实践：本文件在 Tools/ 子目录。Python 默认找不到项目根的 utils.py，
# 必须把根目录插入 sys.path，后面才能 `from utils import ...`。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.tools import tool

from utils import print_stage


# 官方文档：@tool 把普通函数变成带 JSON schema 的可调用工具。
# 类型注解 a: int, b: int → 发给模型的参数 schema（模型据此填 args）。
# docstring → description（模型据此决定「要不要调、什么时候调」）。
# 源码分析：装饰后得到 StructuredTool；.name 默认取函数名 add。
@tool
def add(a: int, b: int) -> int:
    """计算两个整数的和。

    Args:
        a: 第一个整数
        b: 第二个整数
    """
    return a + b


# 官方文档：@tool("web_search") 覆盖函数名。发给模型的 name 是 web_search，不是 search。
# 工程实践：工具名用 snake_case；空格、连字符、特殊字符部分厂商会拒收。
@tool("web_search")
def search(query: str) -> str:
    """按关键词搜索内部知识（示例占位）。"""
    return f"未找到与 {query!r} 相关的真实检索后端，这是占位结果。"


def main() -> None:
    print_stage("自动生成的工具属性")
    # 官方文档：name / description / args_schema 就是发给模型的那份「菜单」。
    # 源码分析：没有显式 args_schema 时，框架从类型注解推断 Pydantic 模型。
    print("name:", add.name)
    print("description:", add.description)
    print("args_schema:", add.args_schema)

    print_stage("直接调用工具")
    # 官方文档：tool.invoke 先按 schema 校验，再执行函数体。
    # 和直接 add(3, 5) 不同：invoke 走 Tool 接口，bind_tools 循环 / Agent 也是这条路径。
    print(add.invoke({"a": 3, "b": 5}))

    print_stage("自定义工具名")
    print("search.name =", search.name)

    print_stage("知识点总结")
    print(
        "没有类型注解，模型不知道参数长什么样。\n"
        "docstring 写清楚「何时用、参数含义」，比堆砌形容词更有效。\n"
        "config / runtime 是保留参数名，不要拿来当业务字段。"
    )


if __name__ == "__main__":
    main()
