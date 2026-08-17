"""
知识点：updates / messages / custom 三种 stream_mode 对比

官方文档：
https://docs.langchain.com/oss/python/langchain/streaming#supported-stream-modes

本示例主要学习：
1. 三种模式分别服务进度条、打字机、业务信号。
2. 新项目更推荐 Event streaming 的投影 API。
3. 本页 API 仍然有用：你需要精确控制 Pregel stream_mode 时。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 工程实践：本文件在 Streaming 子目录。Python 默认只从该目录找模块，
# 把项目根插入 sys.path 后才能 `from utils import ...`。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import print_stage


def main() -> None:
    print_stage("是什么")
    # 官方文档：三种模式可单独订，也可 list 同时订。底层都是 Pregel.stream。
    # 源码分析：version="v2" 统一成 StreamPart（type/ns/data）；默认 v1 可能是元组。
    print(
        "updates：每个图节点跑完后的状态增量。\n"
        "messages：模型生成过程中的 token 与 metadata。\n"
        "custom：工具/节点里 writer() 推送的任意数据。"
    )

    print_stage("区别")
    print(
        "粒度：messages 最细，updates 按步骤，custom 由你定义。\n"
        "解析成本：updates/messages 要自己拆 tuple/dict；"
        "Event streaming 把这些收成 stream.messages / stream.tool_calls。"
    )

    print_stage("使用场景")
    # 工程实践：新前端优先 Event streaming；排错「走到了哪个节点」仍用 updates。
    print(
        "选 updates：调试 Agent 走到了哪一步。\n"
        "选 messages：前端逐字渲染。\n"
        "选 custom：检索进度、文件上传、天气查询阶段。\n"
        "新 UI：优先看 Event streaming 目录的 stream_events(version='v3')。"
    )


if __name__ == "__main__":
    main()
