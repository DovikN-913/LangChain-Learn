"""
知识点：create_agent(response_format=...) 结构化返回

官方文档：
https://docs.langchain.com/oss/python/langchain/structured-output

本示例主要学习：
1. Agent 结构化结果在 state['structured_response']，不是只在最后一条消息里。
2. 直接传 Pydantic 类时，框架按模型能力在 Provider / Tool 策略间选择。
3. 百炼兼容接口更稳妥的做法是显式 ToolStrategy。
"""

from __future__ import annotations

import sys
from pathlib import Path
import json

from pydantic import BaseModel, Field

# 工程实践：本文件在「Structured output」子目录。Python 默认只从该目录找模块，
# 把项目根插入 sys.path 后才能 `from utils import ...`。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

from utils import get_chat_model, print_stage


class ContactInfo(BaseModel):
    """从文本中抽出的联系人。"""

    name: str = Field(description="姓名")
    email: str = Field(description="邮箱")
    phone: str = Field(description="电话")


def main() -> None:
    agent = create_agent(
        model=get_chat_model(),
        tools=[],
        # 官方文档：response_format 约束最终业务对象。结果在 state["structured_response"]。
        # 源码分析：裸传 Pydantic 类会被包成 AutoStrategy，再按 model.profile 选 Provider 或 Tool。
        # ChatOpenAI 常被判成支持原生 structured output，百炼兼容接口却会走 json_object 报 400。
        # 工程实践：本项目一律显式 ToolStrategy，把 schema 当成强制工具调用。
        response_format=ToolStrategy(ContactInfo),
    )

    print_stage("抽取联系人")
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "请登记：林夏，邮箱 linxia@example.com，电话 13800001111",
                }
            ]
        }
    )
    # 官方文档：结构化结果不在最后一条 AIMessage 里，而在独立字段 structured_response。
    # 源码分析：AgentState.structured_response 标了 OmitFromInput，invoke 输入里不必也不该带它。
    structured = result["structured_response"]
    print(type(structured).__name__, structured)
    print("name=", structured.name, "email=", structured.email, "phone=", structured.phone)
    print(json.dumps(structured.model_dump(), ensure_ascii=False))


if __name__ == "__main__":
    main()
