"""
知识点：用摘要代替硬删，保留长期要点

官方文档：
https://docs.langchain.com/oss/python/langchain/short-term-memory#summarize-messages

本示例主要学习：
1. 裁剪会丢信息；摘要用模型压缩旧对话。
2. SummarizationMiddleware 按 token / 消息条数触发。
3. 触发阈值在教学里调低，才能当场看到摘要生效。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 工程实践：本文件在「Short-term memory」子目录。Python 默认只从该目录找模块，
# 把项目根插入 sys.path 后才能 `from utils import ...`。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver

from utils import get_chat_model, print_stage


def main() -> None:
    model = get_chat_model()
    agent = create_agent(
        model=model,
        tools=[],
        checkpointer=InMemorySaver(),
        middleware=[
            SummarizationMiddleware(
                model=model,
                # 官方文档：trigger 决定何时摘要；keep 决定摘要后留多少新消息。
                # 元组 ("messages", N) 表示按条数；也可 ("tokens", 4000) 或 ("fraction", 0.8)。
                # 源码分析：挂在 before_model。达标后用 RemoveMessage(REMOVE_ALL_MESSAGES)
                # 换成「摘要消息 + 保留的最近消息」，并尽量不拆开 AI/Tool 成对。
                # 工程实践：生产按模型窗口设 token 阈值；这里把条数调低，才能当场看到效果。
                trigger=("messages", 6),
                keep=("messages", 4),
            )
        ],
        system_prompt="根据对话历史回答，尽量短。",
    )
    config = {"configurable": {"thread_id": "sum-demo"}}

    turns = [
        "我叫林夏，住在贵阳。",
        "我正在做 LangChain 1.x 的学习项目。",
        "天气工具用的是 UAPI。",
        "模型用的是阿里百炼 qwen3.6。",
        "请重复：我是谁，住哪，在做什么项目？",
    ]
    for text in turns:
        print_stage(text)
        result = agent.invoke({"messages": [{"role": "user", "content": text}]}, config)
        print(result["messages"][-1].content)

    print_stage("知识点总结")
    print(
        "trigger 决定何时摘要，keep 决定摘要后留多少新消息。\n"
        "摘要本身也耗 token，短对话没必要开。\n"
        "生产阈值按模型上下文窗口设置，例如 tokens=4000。"
    )


if __name__ == "__main__":
    main()
