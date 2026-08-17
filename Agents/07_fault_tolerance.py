"""
知识点：Fault tolerance 中间件 — 模型/工具重试

官方文档：
https://docs.langchain.com/oss/python/langchain/agents#fault-tolerance

本示例主要学习：
1. 生产里的 429、超时、瞬时错误不该散落在每个工具的 try/except 里。
2. ModelRetryMiddleware / ToolRetryMiddleware 在 harness 层统一重试。
3. 这和 Chat Model 自己的 max_retries 互补：一个管模型 HTTP，一个管 Agent 循环里的工具失败。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 工程实践：本文件在 Agents/ 子目录。Python 默认找不到项目根的 utils.py，
# 必须把根目录插入 sys.path，后面才能 `from utils import ...`。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware, ToolRetryMiddleware

from utils import get_chat_model, get_weather, print_stage


def query_weather(city: str) -> str:
    """查询指定城市的天气。"""
    return get_weather(city)


def main() -> None:
    # 官方文档：ModelRetryMiddleware 重试模型调用失败；ToolRetryMiddleware 重试工具执行失败。
    # 和 ChatOpenAI(max_retries=...) 互补：模型 max_retries 管 HTTP 层（网络/429/5xx）；
    # 中间件管 Agent 循环层（工具抛错、模型节点失败）。两者可以同时开。
    # 工程实践：瞬时错误放 harness，不要每个工具自己写一遍 try/except 重试。
    agent = create_agent(
        model=get_chat_model(),
        tools=[query_weather],
        middleware=[
            ModelRetryMiddleware(max_retries=3),
            ToolRetryMiddleware(max_retries=2),
        ],
        system_prompt="必须调用工具查询真实天气。回答简短。",
    )

    print_stage("正常路径：中间件在失败时才会介入")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "贵阳现在气温多少？"}]}
    )
    print(result["messages"][-1].content)

    print_stage("知识点总结")
    print(
        "max_retries 写在模型上：请求级重试。\n"
        "Middleware：Agent 循环级重试，覆盖工具异常。\n"
        "两者可以同时开。"
    )


if __name__ == "__main__":
    main()
