"""
LangGraph 入门：人机协作（Human-in-the-loop）

前面 01~03 都是程序自己跑完。
这份代码会在「审核」这一步停下来，等你在终端里输入 y / n。

画出来就是：

    入口 → draft → review ─┬─► publish → END     （你输入 y，批准）
                    ▲      │
                    │      └─► revise ──┘        （你输入 n，给意见后再审）
                    └──────────┘

新概念：
- 人机协作 = 图中间插入一个「等人」的节点
- checkpointer = 把每一步的 State 存下来（可恢复、可按 thread 区分会话）
- thread_id    = 这次对话 / 这篇稿子的身份证

注意：
这份是教学简化版。生产里更常见的是 interrupt() 暂停图，
等 HTTP 接口把人的决定送回来再 resume。这里直接用 input() 阻塞进程。
"""

from langchain_core.runnables.graph import MermaidDrawMethod
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from typing import TypedDict


# ============================================================
# 1. 状态：稿子 + 人有没有批 + 人给的修改意见
# ============================================================
class State(TypedDict):
    content: str     # 当前文稿
    approved: bool   # 人是否批准
    feedback: str    # 拒绝时写下的修改意见


# ============================================================
# 2. 节点
#    draft / revise / publish 是机器做的
#    review 是人做的（用 input() 模拟审核界面）
# ============================================================
def generate_draft(state: State) -> State:
    """机器先写一版草稿。真实项目里这里通常会调 LLM。"""
    print("📝 生成草稿...")
    state["content"] = "这是一篇关于 AI 的文章草稿..."
    return state


def human_review(state: State) -> State:
    """
    人机协作的核心节点：进程会卡在 input()，直到你打字回车。

    y → approved=True，后面路由去 publish
    n → approved=False，再问一句修改意见，后面路由去 revise
    """
    print("\n⏸️  等待人工审核...")
    print(f"内容: {state['content']}")
    choice = input("您的决定 (y = 批准 / n = 拒绝): ").strip().lower()
    if choice == "y":
        state["approved"] = True
        state["feedback"] = ""
        print("✅ 已批准")
    else:
        state["approved"] = False
        state["feedback"] = input("请提供修改意见: ")
        print("📝 反馈已记录")
    return state


def revise_content(state: State) -> State:
    """按人的意见改一版。改完会回到 review，再让人看一次。"""
    print(f"🔧 根据反馈修订: {state['feedback']}")
    state["content"] = f"{state['content']}\n[已修订: {state['feedback']}]"
    return state


def publish_content(state: State) -> State:
    """批准后的收尾动作。这里只打印，真实项目可能是发到 CMS / 写文件。"""
    print("🚀 发布内容...")
    return state


# ============================================================
# 3. 路由：审核节点做完后，看 approved 决定去发布还是去修订
# ============================================================
def check_approval(state: State) -> str:
    return "publish" if state["approved"] else "revise"


# ============================================================
# 4. 建图
#    draft → review → (publish | revise)
#    revise → review   ← 形成「改完再审」的环
# ============================================================
workflow = StateGraph(State)
workflow.add_node("draft", generate_draft)
workflow.add_node("review", human_review)
workflow.add_node("revise", revise_content)
workflow.add_node("publish", publish_content)

workflow.set_entry_point("draft")
workflow.add_edge("draft", "review")
workflow.add_conditional_edges(
    "review",
    check_approval,
    {"publish": "publish", "revise": "revise"},
)
workflow.add_edge("revise", "review")
workflow.add_edge("publish", END)


# ============================================================
# 5. Checkpointer（检查点 / 存档）
#    没有它：图跑完 State 就丢了，也无法按「会话」区分多次 invoke。
#    有了它：每走完一个节点，LangGraph 都会把 State 存一份。
#
#    ":memory:" = 存在内存里的 SQLite，进程退出就没了。
#    换成 "checkpoints.db" 就会落到磁盘，重启还能接着聊同一个 thread_id。
#
#    check_same_thread=False：允许不同线程用同一个连接（LangGraph 内部会用到）。
# ============================================================
conn = sqlite3.connect(":memory:", check_same_thread=False)
checkpointer = SqliteSaver(conn)

# compile 时把存档器塞进去，之后每次 invoke 都要带上 thread_id
app = workflow.compile(checkpointer=checkpointer)

try:
    with open("04-人机协作工作流.png", "wb") as f:
        f.write(app.get_graph(xray=True).draw_mermaid_png(draw_method=MermaidDrawMethod.API))
except Exception as e:
    print(f"流程图渲染失败: {e}")
    print("\n使用 Mermaid 文本方式显示:")
    print(app.get_graph(xray=True).draw_mermaid())


# ============================================================
# 6. 执行
#    config.configurable.thread_id = 这次任务的身份证。
#    同一个 thread_id 会复用同一份存档；换一个 id 就是全新一篇稿。
# ============================================================
initial_state = {"content": "", "approved": False, "feedback": ""}
config = {"configurable": {"thread_id": "1"}}

result = app.invoke(initial_state, config)
print(f"\n最终状态: {'已发布' if result['approved'] else '待修订'}")
print("最终内容:", result["content"])
