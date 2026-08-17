"""
LangGraph 入门：循环工作流

和前面的差别：
- 01 直线：做完就结束
- 02 分叉：选一条路，做完就结束
- 03 循环：质量不够就回去再生成，直到达标或次数用尽

画出来就是：

    入口 → generate → evaluate ─┬─► END          （够好了，或次数到了）
                    ▲          │
                    │          └─► improve ──┘   （不够好，改进后再生成）
                    └──────────────┘

新概念：
- 循环不是 while True 手写出来的，而是「边指回前面的节点」
- 一定要有退出条件，否则会永远转圈
"""

from langchain_core.runnables.graph import MermaidDrawMethod
from langgraph.graph import StateGraph, END
from typing import TypedDict


# ============================================================
# 1. 状态里多了「计数器」和「是否够好」
#    循环图几乎都要有这两个东西，否则不知道何时停。
# ============================================================
class State(TypedDict):
    content: str          # 当前生成出来的内容
    iteration: int        # 现在是第几轮（从 1 开始）
    max_iterations: int   # 最多转几轮（安全阀，防止死循环）
    is_good_enough: bool  # 评估节点认为够不够好


# ============================================================
# 2. 三个节点：生成 → 评估 →（可选）改进
#    这里仍然没有 LLM，用「第 3 轮就算够好」来模拟评估。
# ============================================================
def generate_content(state: State) -> State:
    """按当前轮次生成一版内容。循环回来时会带着更大的 iteration 再跑一次。"""
    print(f"📝 生成内容（第 {state['iteration']} 次）...")
    state["content"] = f"内容版本 {state['iteration']}"
    return state


def evaluate_content(state: State) -> State:
    """评估：这里写死「第 3 次及以后就算达标」，方便你看循环怎么停。"""
    print("✅ 评估内容...")
    state["is_good_enough"] = state["iteration"] >= 3
    return state


def improve_content(state: State) -> State:
    """改进：真正要改的往往是 content；这里为了演示循环，只把轮次 +1。"""
    print("🔧 改进内容...")
    state["iteration"] += 1
    return state


# ============================================================
# 3. 路由函数：评估完之后，决定「继续转」还是「下车」
#    返回 "continue" 或 "end"，后面的字典会把它们映射到节点 / END。
# ============================================================
def should_continue(state: State) -> str:
    """循环的刹车：达标 或 次数用尽 → end；否则 → continue。"""
    if state["is_good_enough"]:
        print("✨ 内容已达标！")
        return "end"
    elif state["iteration"] >= state["max_iterations"]:
        print("⚠️  达到最大迭代次数")
        return "end"
    else:
        print("🔄 需要继续改进")
        return "continue"


# ============================================================
# 4. 建图
#    固定边：generate → evaluate
#    条件边：evaluate → improve 或 END
#    回边：  improve → generate   ← 就是这一条把图收成了环
# ============================================================
workflow = StateGraph(State)

workflow.add_node("generate", generate_content)
workflow.add_node("evaluate", evaluate_content)
workflow.add_node("improve", improve_content)

workflow.set_entry_point("generate")
workflow.add_edge("generate", "evaluate")

workflow.add_conditional_edges(
    "evaluate",
    should_continue,
    {
        "continue": "improve",  # 不够好：去改进
        "end": END,             # 够好 / 次数到了：结束
    },
)

# 改进完必须回到 generate，否则只是「改进一次就停」，构不成循环
workflow.add_edge("improve", "generate")

app = workflow.compile()


# ============================================================
# 5. 画图（失败就打印 Mermaid 文本，方便没网时也能看结构）
# ============================================================
try:
    with open("03-循环工作流.png", "wb") as f:
        f.write(app.get_graph(xray=True).draw_mermaid_png(draw_method=MermaidDrawMethod.API))
except Exception as e:
    print(f"流程图渲染失败: {e}")
    print("\n使用 Mermaid 文本方式显示:")
    print(app.get_graph(xray=True).draw_mermaid())


# ============================================================
# 6. 执行
#    初始 iteration=1，max=5，is_good_enough=False
#    实际会走 3 轮后因「第 3 次达标」停下，不会用到 max=5。
#
#    第 1 轮: generate(版本1) → evaluate(不够) → improve(变成2) → generate
#    第 2 轮: generate(版本2) → evaluate(不够) → improve(变成3) → generate
#    第 3 轮: generate(版本3) → evaluate(够了) → END
# ============================================================
result = app.invoke({
    "content": "",
    "iteration": 1,
    "max_iterations": 5,
    "is_good_enough": False,
})

print(f"\n最终内容: {result['content']}")
print(f"总迭代次数: {result['iteration']}")
