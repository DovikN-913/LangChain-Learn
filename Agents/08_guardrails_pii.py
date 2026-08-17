"""
知识点：Guardrails — 用 PIIMiddleware 确定性拦截敏感信息

官方文档：
https://docs.langchain.com/oss/python/langchain/agents#guardrails

本示例主要学习：
1. 有些策略不能只写在 prompt 里，必须在循环里强制执行。
2. PIIMiddleware 在数据流经 Agent 时检测/处理邮箱等 PII。
3. 演示数据使用虚构邮箱，不要填真实个人信息。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 工程实践：本文件在 Agents/ 子目录。Python 默认找不到项目根的 utils.py，
# 必须把根目录插入 sys.path，后面才能 `from utils import ...`。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware

from utils import get_chat_model, print_stage


def echo_note(text: str) -> str:
    """原样记录一段用户备注，供后续回复引用。"""
    return f"已记录备注：{text}"


def main() -> None:
    # 官方文档：PII 这类策略必须确定性拦截，不能只靠 prompt「请不要输出邮箱」。
    # PIIMiddleware("email") 在数据流经循环时检测/处理邮箱，不依赖模型是否听话。
    # 源码分析：默认处理输入侧；apply_to_output=True 还会在 Event streaming 流出前打码。
    agent = create_agent(
        model=get_chat_model(),
        tools=[echo_note],
        middleware=[PIIMiddleware("email")],
        system_prompt="如果用户让你记录备注，调用 echo_note。回复时不要重复完整邮箱。",
    )

    print_stage("输入含虚构邮箱")
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "请记下：联系人 demo.user@example.com，明天下午开会。",
                }
            ]
        }
    )
    print(result["messages"][-1].content)

    print_stage("知识点总结")
    print(
        "Guardrails 拦截的是数据流，不依赖模型是否「听话」。\n"
        "apply_to_output=True 时还会在 Event streaming 流出前打码。"
    )


if __name__ == "__main__":
    main()
