"""
知识点：初始化 Chat Model（init_chat_model vs ChatOpenAI）

官方文档：
https://docs.langchain.com/oss/python/langchain/models#initialize-a-model

本示例主要学习：
1. LangChain 1.x 推荐用 Chat Model，而不是旧式 Completion LLM。
2. `init_chat_model` 是官方统一入口；厂商 Chat 类是更明确的写法。
3. 阿里百炼走 OpenAI 兼容接口，必须显式传入 base_url 和 api_key。
4. 三个关键参数的含义（本文件重点）：
   - model_provider：选哪套集成实现。"openai" 表示走 OpenAI Chat Completions 协议，
     不等于「一定连 OpenAI 官方」。百炼不是内置 provider，必须显式指定。
   - base_url：HTTP 请求打到哪台服务器。不设则默认 api.openai.com；
     本项目指向 DashScope 兼容网关。
   - extra_body：原样并进请求 JSON 的厂商扩展字段。Qwen 的 enable_thinking
     不是 LangChain 标准参数，只能走这条通道。

三种写法底层最终都得到 ChatOpenAI 实例，invoke 行为应一致。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 子目录默认找不到项目根，插入后才能 import utils
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI

from utils import get_chat_model, print_stage

load_dotenv()


def main() -> None:
    print_stage("方式一：项目封装 get_chat_model()")
    # 工程实践：教学项目把厂商差异（base_url、api_key、关 thinking）收口到 utils。
    # 将项目需要使用的所有模型都可以封装成一个函数，换模型时改一处即可。这样就不用在每个文件里写重复的代码。还可以给函数传参数，比如 model_name。来指定模型。
    model = get_chat_model()
    print(type(model).__name__, model.model_name)
    print()

    print_stage("方式二：官方推荐 init_chat_model")
    # 官方文档：init_chat_model 是跨厂商统一入口。
    # 传入 "gpt-5.5" 这类内置模型名时，会按名字推断 provider。
    # 源码分析：百炼不在内置 provider 表里，推断会失败，因此必须同时给：
    #   model_provider="openai"  → 选用 ChatOpenAI 这套实现（协议兼容，不是官方账号）
    #   base_url                 → 把 HTTP 从 api.openai.com 改打到 DashScope
    #   extra_body               → 把 enable_thinking 等非标准字段塞进请求体
    # 和方式三的区别：init_chat_model 按 provider 字符串动态选类，换厂商改参数即可；
    # ChatOpenAI 则把实现类写死，意图更明确，少一层分发。
    model_via_init = init_chat_model(
        model=os.getenv("DASHSCOPE_CHAT_MODEL_NAME"),
        model_provider="openai",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_API_URL"),
        temperature=0.2,
        extra_body={"enable_thinking": False},
    )
    print(type(model_via_init).__name__)
    print(model_via_init.model_name)
    print()

    print_stage("方式三：直接使用 ChatOpenAI")
    # 官方文档：ChatOpenAI 是 OpenAI Chat Completions 协议的标准实现。
    # 工程实践：本项目固定走百炼兼容接口，直接用这个类更直观，少一次 provider 查找。
    # extra_body 同样需要关 thinking：Qwen3.6 默认会产出大量 reasoning token。
    model_via_class = ChatOpenAI(
        model=os.environ["DASHSCOPE_CHAT_MODEL_NAME"],
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url=os.environ["DASHSCOPE_API_URL"],
        extra_body={"enable_thinking": False},
    )
    print(type(model_via_class).__name__)

    print_stage("三种方式调用结果应一致")
    question = "用五个字回答：1+1等于几？"
    print("get_chat_model:", model.invoke(question).content)
    print("init_chat_model:", model_via_init.invoke(question).content)
    print("ChatOpenAI:", model_via_class.invoke(question).content)

    print_stage("知识点总结")
    print(
        "旧版本写法：from langchain.llms import OpenAI  # Completion LLM，已不推荐\n"
        "当前推荐写法：init_chat_model(...) 或 ChatOpenAI(...)\n"
        "本项目统一用 DashScope OpenAI 兼容接口，密钥只从 .env 读取。"
    )


if __name__ == "__main__":
    main()
