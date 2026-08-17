"""
知识点：ainvoke / astream / abatch 异步调用

官方文档：
https://docs.langchain.com/oss/python/langchain/models

本示例主要学习：
1. Chat Model 同时提供同步（invoke/stream/batch）和异步（ainvoke/astream/abatch）接口。
   底层都是同一套 Chat Completions 请求，差别只在于是否把事件循环让给别人。
2. 为何 Web 要用异步：FastAPI / asyncio 服务里若调用同步 invoke，会堵住整个事件循环，
   其他用户的请求只能干等这次 HTTP 完成。异步接口在等网络时可以去处理别的连接。
3. astream 是异步迭代器，必须 `async for`；写成普通 for 会报错。
4. asyncio.run() 会新建并关闭一层事件循环。Jupyter、FastAPI、已在跑的 loop 里
   再套一层会 RuntimeError。脚本入口用它可以；协程内部应 await，不要再 run。

同步适合脚本、notebook、一次性批处理；异步适合同时服务多个用户。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# 子目录默认找不到项目根，插入后才能 import utils
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import get_chat_model, print_stage


async def main() -> None:
    model = get_chat_model()

    print_stage("ainvoke")
    # 官方文档：ainvoke 是 invoke 的异步版，同样返回完整 AIMessage。
    # await 期间把控制权交还事件循环，Web 服务才能同时处理其他请求。
    response = await model.ainvoke("用一句话说明异步调用模型的好处。")
    print(response.content)

    print_stage("astream")
    # 官方文档：astream 产出 AIMessageChunk。
    # 源码分析：它是 async iterator，只能 async for；普通 for 不会去 __anext__。
    # flush=True 的原因与同步 stream 相同：无换行时立刻把增量刷到终端。
    async for chunk in model.astream("数到 5，每个数字单独一行。"):
        print(chunk.content, end="", flush=True)
    print()

    print_stage("abatch")
    # 官方文档：abatch 在异步上下文里并行多条独立请求，语义同 batch。
    responses = await model.abatch(
        [
            "给「模型」起一个两字别名。",
            "给「消息」起一个两字别名。",
        ]
    )
    for item in responses:
        print(item.content)

    print_stage("知识点总结")
    print(
        "同步：脚本、notebook、一次性批处理。\n"
        "异步：Web 服务、同时处理多个用户请求。\n"
        "不要在已运行的事件循环里再用 asyncio.run() 套一层。"
    )


if __name__ == "__main__":
    # 工程实践：脚本入口用 asyncio.run 启动唯一一层 loop。
    # 已有 loop（Jupyter / FastAPI）里应直接 await main()，不要再 run。
    asyncio.run(main())
