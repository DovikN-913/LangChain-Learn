"""
知识点：结构化输出校验失败与重试

官方文档：
https://docs.langchain.com/oss/python/langchain/structured-output#tool-calling-strategy

本示例主要学习：
1. ToolStrategy.handle_errors 控制校验失败后是否让模型再试。
2. Pydantic 的 ge/le 会把越界值挡下来。
3. handle_errors=False 时异常会冒泡，适合测试和严格流水线。
"""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic import BaseModel, Field

# 工程实践：本文件在「Structured output」子目录。Python 默认只从该目录找模块，
# 把项目根插入 sys.path 后才能 `from utils import ...`。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

from utils import get_chat_model, print_stage


class Score(BaseModel):
    """1 到 5 的整数评分。"""

    score: int = Field(description="必须是 1 到 5 的整数", ge=1, le=5)
    reason: str = Field(description="理由")


def main() -> None:
    model = get_chat_model()
    prompt = {
        "messages": [
            {
                "role": "user",
                "content": "给这段评价打分：物流很快，包装一般。分数必须是 1-5。",
            }
        ]
    }

    print_stage("默认 handle_errors=True：失败会重试")
    # 官方文档：handle_errors 默认 True——捕获校验错误，把错误写进 ToolMessage，让模型改到合法为止。
    # 源码分析：True → 默认错误模板重试；str → 自定义文案；Exception 类型/元组 → 只对这些异常重试；
    # Callable → 自己生成文案；False → 不重试，异常冒泡。
    # ProviderStrategy 没有这个参数，校验失败不会自动回炉。
    agent_retry = create_agent(
        model=model,
        tools=[],
        response_format=ToolStrategy(Score),
    )
    print(agent_retry.invoke(prompt)["structured_response"])

    print_stage("handle_errors=False：校验失败直接抛错")
    agent_strict = create_agent(
        model=model,
        tools=[],
        response_format=ToolStrategy(Score, handle_errors=False),
    )
    try:
        print(agent_strict.invoke(prompt)["structured_response"])
    except Exception as exc:
        print(type(exc).__name__, str(exc)[:400])

    print_stage("知识点总结")
    print(
        "面向用户的助手：打开 handle_errors，让模型改到合法为止。\n"
        "数据管道：关闭它，失败进死信队列，避免悄悄吞掉坏数据。"
    )


if __name__ == "__main__":
    main()
