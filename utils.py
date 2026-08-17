"""
项目公共工具。

职责：
1. 从 `.env` 读取阿里百炼 DashScope（OpenAI 兼容接口）配置，统一创建 Chat Model。
2. 把 UAPI 天气查询改造成「返回字符串」的函数，方便作为 LangChain Tool 使用。
3. 提供统一的阶段打印，方便观察示例脚本执行流程。

官方文档：
- Models: https://docs.langchain.com/oss/python/langchain/models
- Tools: https://docs.langchain.com/oss/python/langchain/tools
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from uapi import UapiClient
from uapi.errors import UapiError

# load_dotenv() 会把项目根目录 .env 读进 os.environ。
# 必须在读取 DASHSCOPE_* / UAPI_* 之前调用，否则 get_chat_model 会以为缺配置。
load_dotenv()

if sys.platform == "win32":
    # 工程实践：Windows 控制台默认 GBK，print 中文会乱码。
    # reconfigure 失败时忽略，避免在特殊 IDE 里把导入搞崩。
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def print_stage(title: str) -> None:
    """打印统一风格的阶段标题，方便观察脚本执行流程。"""
    print(f"{'=' * 20} {title} {'=' * 20}")


def get_chat_model(*, enable_thinking: bool = False, **kwargs: Any) -> ChatOpenAI:
    """创建本项目统一使用的 Chat Model。

    官方文档：LangChain 推荐用 `init_chat_model` 或各厂商 Chat Model 类。
    本项目模型走阿里百炼 OpenAI 兼容接口，因此使用 `ChatOpenAI` + `base_url`。

    源码分析：`qwen3.6-35b-a3b` 默认会产出大量 reasoning token。
    教学示例默认关闭 thinking，避免一次简单问答消耗近千 token。
    需要观察推理过程时传入 `enable_thinking=True`。
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("DASHSCOPE_API_URL")
    model_name = os.getenv("DASHSCOPE_CHAT_MODEL_NAME")
    if not api_key or not base_url or not model_name:
        raise RuntimeError(
            "缺少 DashScope 配置。请在 .env 中设置 "
            "DASHSCOPE_API_KEY / DASHSCOPE_API_URL / DASHSCOPE_CHAT_MODEL_NAME"
        )

    # extra_body 会原样传给 OpenAI 兼容接口的请求体。
    # 调用方若自己传 extra_body，这里合并而不是覆盖，以免丢掉 enable_thinking。
    extra_body = dict(kwargs.pop("extra_body", {}) or {})
    extra_body.setdefault("enable_thinking", enable_thinking)

    # temperature / timeout / max_retries 都是 Chat Model 标准参数：
    # - temperature：越低越确定，工具调用更稳
    # - timeout：单次 HTTP 等待秒数，超时会触发重试或失败
    # - max_retries：网络错误 / 429 / 5xx 会重试；401、404 不会
    params: dict[str, Any] = {
        "model": model_name,
        "api_key": api_key,
        "base_url": base_url,
        "temperature": 0.2,
        "timeout": 60,
        "max_retries": 3,
        "extra_body": extra_body,
    }
    # kwargs 放在后面，示例里可以覆盖默认温度、超时等。
    params.update(kwargs)
    return ChatOpenAI(**params)


def get_embedding_model(**kwargs: Any) -> OpenAIEmbeddings:
    """创建阿里百炼 Embedding 模型（OpenAI 兼容接口）。

    官方文档：Chat Model 生成文本；Embeddings 把文本变成固定维向量，供检索使用。
    百炼同样走兼容接口，因此仍用 OpenAIEmbeddings + base_url，而不是 ChatOpenAI。

    源码分析：OpenAIEmbeddings 默认会用 tiktoken 按 OpenAI 的 8192 token 上限做截断检查。
    百炼 text-embedding-v1 的分词和上限都不同，检查会误报，所以关闭 check_embedding_ctx_length。
    工程实践：百炼单次最多 25 条文本，chunk_size 必须压到 25 以内，否则 embed_documents 会 400。
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("DASHSCOPE_API_URL")
    model_name = os.getenv("DASHSCOPE_EMBEDDING_MODEL_NAME")
    if not api_key or not base_url or not model_name:
        raise RuntimeError(
            "缺少 DashScope Embedding 配置。请在 .env 中设置 "
            "DASHSCOPE_API_KEY / DASHSCOPE_API_URL / DASHSCOPE_EMBEDDING_MODEL_NAME"
        )
    params: dict[str, Any] = {
        "model": model_name,
        "api_key": api_key,
        "base_url": base_url,
        "check_embedding_ctx_length": False,
        # 百炼 text-embedding-v1 单次 batch 上限 25；默认 1000 会直接 400。
        "chunk_size": 25,
    }
    params.update(kwargs)
    return OpenAIEmbeddings(**params)


def get_weather(city: str) -> str:
    """查询指定城市的实时天气、空气质量和未来几天预报。

    Args:
        city: 城市名称，例如「贵阳」「北京」「上海」。

    Returns:
        JSON 字符串。成功时包含实况、空气质量、生活指数和预报；
        失败时包含 error 字段，供模型和调用方处理。

    工程实践：
    - Tool 应返回字符串，而不是 print。print 只方便人看，模型读不到。
    - 异常不要直接抛给 Agent 循环（除非你想演示错误处理），
      返回可读错误信息通常更稳妥。
    """
    token = os.getenv("UAPI_API_KEY")
    if not token:
        return json.dumps({"error": "缺少 UAPI_API_KEY"}, ensure_ascii=False)

    client = UapiClient("https://uapis.cn", token=token)
    try:
        # extended / forecast / indices 打开后，结果更长，但 Agent 做出行建议时用得上。
        # hourly / minutely 默认关掉，避免把过大 JSON 塞进模型上下文。
        result = client.misc.get_misc_weather(
            city=city,
            adcode="",
            extended=True,
            forecast=True,
            hourly=False,
            minutely=False,
            indices=True,
            lang="zh",
        )
        # ensure_ascii=False：中文不以 \uXXXX 转义，模型更好读。
        # default=str：遇到不能直接 JSON 化的对象时降级成字符串，避免工具直接崩。
        return json.dumps(result, ensure_ascii=False, default=str)
    except UapiError as exc:
        return json.dumps(
            {"error": f"天气查询失败: {exc}", "city": city},
            ensure_ascii=False,
        )


if __name__ == "__main__":
    print_stage("天气查询自测")
    print(get_weather("贵阳"))
