"""
知识点：模型调用的异常与边界

官方文档：
https://docs.langchain.com/oss/python/langchain/models#parameters
https://docs.langchain.com/oss/python/langchain/models#connection-resilience

本示例主要学习：
1. 缺少 API Key / 模型名错误时会发生什么。
2. timeout 过短会失败。
3. 401（鉴权失败）/ 404（资源不存在，含错误模型名）不会自动重试；
   网络错误、429、5xx 才会按 max_retries 指数退避。
4. 本文件把 max_retries=0：演示错误时不想干等默认约 6 次重试。
   即使不设 0，401/404 本来也不会重试；设 0 是让超时等「本会重试」的错误也立刻抛出。
5. openai_api_key / openai_api_base 复用：ChatOpenAI 把构造参数存成这两个属性。
   从已成功的 model 上取，保证鉴权和网关与正常调用相同，失败原因才是「模型名不存在」，
   而不是误配成空 key。不要把密钥写进源码。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 子目录默认找不到项目根，插入后才能 import utils
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_openai import ChatOpenAI

from utils import get_chat_model, print_stage


def main() -> None:
    print_stage("正常调用")
    model = get_chat_model()
    print(model.invoke("只回复：ok").content)

    print_stage("错误模型名（通常不会重试，直接 4xx）")
    # 官方文档：404 / 401 不在重试名单里，会立刻失败。
    # max_retries=0：工程实践。即使将来厂商把「模型不存在」映射成可重试状态码，
    # 教学演示也不该空等；生产里对确定性 4xx 同样应快速失败。
    # openai_api_key / openai_api_base：复用上面成功实例的鉴权与网关，
    # 排除「key 写错」干扰，让异常明确指向无效 model 名。
    bad = ChatOpenAI(
        model="this-model-does-not-exist-xxx",
        api_key=model.openai_api_key,
        base_url=model.openai_api_base,
        max_retries=0,
        timeout=20,
        extra_body={"enable_thinking": False},
    )
    try:
        bad.invoke("hello")
    except Exception as exc:
        print(type(exc).__name__, str(exc)[:300])

    print_stage("timeout 过短")
    # timeout 是单次 HTTP 等待上限。0.01 秒几乎必然超时。
    # 源码分析：超时默认会触发重试；这里同样 max_retries=0，避免演示被退避拖成数秒。
    impatient = get_chat_model(timeout=0.01, max_retries=0)
    try:
        impatient.invoke("写一首七言绝句")
    except Exception as exc:
        print(type(exc).__name__, str(exc)[:300])

    print_stage("知识点总结")
    print(
        "生产环境至少要处理：鉴权失败、超时、限流、模型不存在。\n"
        "不要把 API Key 写进代码；缺失时应在启动阶段快速失败。"
    )


if __name__ == "__main__":
    main()
