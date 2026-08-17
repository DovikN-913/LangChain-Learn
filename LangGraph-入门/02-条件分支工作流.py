"""
LangGraph 入门：条件分支工作流

和 01 的差别：
- 01 是一条直线：step1 → step2 → step3
- 02 会在 classify 之后「分叉」，根据输入内容走三条路中的一条

画出来就是：

                    ┌─► code ─────┐
    入口 → classify ─┼─► article ──┼─► END
                    └─► general ──┘

新概念：
- 普通边 add_edge          = 固定走下一家（没有选择）
- 条件边 add_conditional_edges = 先问一个路由函数「下一步去哪」
"""

from typing import TypedDict, Literal

from langchain_core.runnables.graph import MermaidDrawMethod
from langgraph.graph import StateGraph, END


# ============================================================
# 1. 定义状态 State（共用记事本）
#    category 很关键：分类节点把它写上，路由函数再读它来决定走哪条路。
# ============================================================
class State(TypedDict):
    input: str      # 用户原始输入
    category: str   # 分类结果：code / article / general
    result: str     # 最终处理结果


# ============================================================
# 2. 定义节点
#    classify 负责「看输入、打标签」
#    后面三个节点负责「按标签做不同的事」
#    注意：这里仍然没有 LLM，只是用字符串判断，方便先看懂分支。
# ============================================================
def classify(state: State) -> State:
    """分类节点：只负责给 input 打上 category，自己不产出最终答案。"""
    print("🔍 分类中...")
    if "代码" in state["input"]:
        state["category"] = "code"
    elif "文章" in state["input"]:
        state["category"] = "article"
    else:
        state["category"] = "general"
    return state


def process_code(state: State) -> State:
    """代码这条路：只有 category == code 时才会走到这里。"""
    print("💻 处理代码...")
    state["result"] = f"生成代码: {state['input']}"
    return state


def process_article(state: State) -> State:
    """文章这条路。"""
    print("📝 处理文章...")
    state["result"] = f"撰写文章: {state['input']}"
    return state


def process_general(state: State) -> State:
    """兜底这条路：既不像代码也不像文章。"""
    print("💬 处理一般请求...")
    state["result"] = f"回答: {state['input']}"
    return state


# ============================================================
# 3. 路由函数（不是节点！）
#    和节点的区别：
#    - 节点：改 State，然后 return State
#    - 路由函数：不负责干活，只返回「下一个节点的名字」
#
#    Literal[...] 是类型提示：告诉读代码的人，这里只能返回这三个字符串。
# ============================================================
def route_by_category(state: State) -> Literal["code", "article", "general"]:
    """根据 classify 写好的 category，告诉图下一步去哪个节点。"""
    return state["category"]


# ============================================================
# 4. 构建图：先放节点，再连线
# ============================================================
workflow = StateGraph(State)

workflow.add_node("classify", classify)
workflow.add_node("code", process_code)
workflow.add_node("article", process_article)
workflow.add_node("general", process_general)

# 所有请求都先分类，所以入口固定是 classify
workflow.set_entry_point("classify")


# ============================================================
# 5. 条件边：classify 做完后，不写死下一站
#    add_conditional_edges(从哪出发, 用谁决定, 返回值怎么映射到节点)
#
#    第三个参数是字典：
#      路由函数返回的字符串  →  真正要去的节点名
#    这里两边写成一样（"code" → "code"），是为了好对照。
#    也可以写成 {"code": "process_code_node"}，名字不必相同。
# ============================================================
workflow.add_conditional_edges(
    "classify",
    route_by_category,
    {
        "code": "code",
        "article": "article",
        "general": "general",
    },
)

# 三条支路做完都结束，不再汇合到同一个后续节点
workflow.add_edge("code", END)
workflow.add_edge("article", END)
workflow.add_edge("general", END)

app = workflow.compile()


# ============================================================
# 6. 画图 + 用 3 条输入分别走 3 条路
#    「帮我写一段Python代码」→ 含「代码」→ code
#    「写一篇关于AI的文章」  → 含「文章」→ article
#    「今天天气怎么样？」    → 都不含   → general
# ============================================================
with open("02-条件工作流.png", "wb") as f:
    f.write(app.get_graph(xray=True).draw_mermaid_png(draw_method=MermaidDrawMethod.API))

test_inputs = [
    "帮我写一段Python代码",
    "写一篇关于AI的文章",
    "今天天气怎么样？",
]

for inp in test_inputs:
    print(f"\n{'=' * 60}")
    print(f"输入: {inp}")
    # category / result 先给空字符串，避免 TypedDict 缺字段
    result = app.invoke({"input": inp, "category": "", "result": ""})
    print(f"结果: {result['result']}")
