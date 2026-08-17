"""
知识点：Event streaming vs Streaming stream_mode

官方文档：
https://docs.langchain.com/oss/python/langchain/event-streaming
https://docs.langchain.com/oss/python/langchain/streaming

本示例主要学习：
1. 两套 API 都能流式看 Agent，但消费体验不同。
2. 官方对应用/前端推荐 v3 投影。
3. 排错图节点时 stream_mode 仍然直观。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 工程实践：本文件在「Event streaming」子目录。Python 默认只从该目录找模块，
# 把项目根插入 sys.path 后才能 `from utils import ...`。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import print_stage


def main() -> None:
    print_stage("是什么")
    # 官方文档：新应用优先 stream_events(version="v3")；stream_mode 仍是理解图执行的基础。
    # 源码分析：v3 标了 @beta，可能看到 BetaWarning。v3 不接受 stream_mode=，由投影自己决定订阅哪些 channel。
    print(
        "Streaming：agent.stream(..., stream_mode=..., version='v2')。\n"
        "Event streaming：agent.stream_events(..., version='v3')。"
    )

    print_stage("区别")
    print(
        "Streaming 返回按模式分类的 chunk，要自己判断 type/knowledge_docs。\n"
        "Event streaming 给出 stream.messages / tool_calls / values / output。\n"
        "异步场景下投影还能 asyncio.gather 并行消费。"
    )

    print_stage("如何选择")
    # 工程实践：不要在同一条请求里混用两套并期待顺序一致。
    print(
        "做聊天 UI、工具进度条、多投影并行 → Event streaming。\n"
        "调试节点、兼容旧代码、精确控制 stream_mode → Streaming。\n"
        "不要在同一条请求里混用两套并期待顺序一致，选一套做主路径。"
    )


if __name__ == "__main__":
    main()
