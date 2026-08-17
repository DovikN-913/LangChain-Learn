"""
LangGraph 入门：顺序工作流

核心比喻：
- State     = 一本大家共用的记事本（整张图里流转的数据）
- 节点 node = 处理步骤（读记事本、改记事本、交回去）
- 边   edge = 步骤之间的箭头（规定谁做完之后轮到谁）
- END       = 终点站（做到这里就停）

这份代码没有分支、没有循环，就是一条直线：
    入口 → step1 → step2 → step3 → 结束
"""

from langgraph.graph import StateGraph, END
from langchain_core.runnables.graph import MermaidDrawMethod
from typing import TypedDict


# ============================================================
# 1. 定义状态 State
#    告诉 LangGraph：这张图里会传递哪些字段。
#    TypedDict 就是“带字段名的字典”，运行时仍是普通 dict。
# ============================================================
class State(TypedDict):
    input: str          # 一开始由调用方传入的原始输入
    step1_output: str   # 第 1 步写进去的结果
    step2_output: str   # 第 2 步写进去的结果
    step3_output: str   # 第 3 步写进去的结果（最终我们要看的）


# ============================================================
# 2. 定义节点（每个节点都是一个普通函数）
#    约定：
#      入参 = 当前整本记事本 State
#      出参 = 改过之后的 State
#    节点之间不直接互相调用，而是靠后面的“边”串起来。
# ============================================================
def step1(state: State) -> State:
    """第 1 步：拿原始 input，加工后写入 step1_output。"""
    print("执行 Step 1 ...")
    # 例：input="Hello World" → step1_output="[Step1 处理] Hello World"
    state["step1_output"] = f"[Step1 处理] {state['input']}"
    return state  # 必须返回 state，下一节点才能接着用


def step2(state: State) -> State:
    """第 2 步：不看原始 input，而是接着加工上一步的结果。"""
    print("执行 Step 2 ...")
    state["step2_output"] = f"[Step2 处理] {state['step1_output']}"
    return state


def step3(state: State) -> State:
    """第 3 步：再加工第二步的结果，这就是流水线的终点产物。"""
    print("执行 Step 3 ...")
    state["step3_output"] = f"[Step3 处理] {state['step2_output']}"
    return state


# ============================================================
# 3. 构建图
#    StateGraph(State) = 新建一张空白图纸，并声明记事本的格式是 State。
#    此时还没有节点、也没有箭头，只是一个空壳。
# ============================================================
workflow = StateGraph(State)


# ============================================================
# 4. 添加节点
#    add_node("名字", 函数)
#    - 左边的字符串是图上的节点名（连线时用这个名字）
#    - 右边的函数是真正执行的代码
#    名字和函数名可以不一样，这里故意起成一样方便对照。
# ============================================================
workflow.add_node("step1", step1)
workflow.add_node("step2", step2)
workflow.add_node("step3", step3)


# ============================================================
# 5. 添加边（规定执行顺序）
#    add_edge(起点, 终点) 表示：起点做完后，自动去跑终点。
#    END 是 LangGraph 内置常量，表示工作流结束，不要再往下走。
#
#    画出来就是：
#        step1 ──► step2 ──► step3 ──► END
# ============================================================
workflow.add_edge("step1", "step2")
workflow.add_edge("step2", "step3")
workflow.add_edge("step3", END)


# ============================================================
# 6. 设置入口点
#    图可能有很多节点，必须告诉 LangGraph：invoke 时从哪个节点开始跑。
# ============================================================
workflow.set_entry_point("step1")


# ============================================================
# 7. 编译
#    compile() 把“图纸”变成“可执行程序”。
#    编译前：只能 add_node / add_edge（还在搭积木）
#    编译后：才能 invoke（真正跑起来）
# ============================================================
app = workflow.compile()


# ============================================================
# 8. 执行
#    invoke(初始状态)：把这本记事本交给入口节点，然后按边一路跑完。
#    这里只传了 input，另外 3 个字段会在各节点里被逐步填上。
#
#    数据接力过程：
#      传入:  {"input": "Hello World"}
#      step1: step1_output = "[Step1 处理] Hello World"
#      step2: step2_output = "[Step2 处理] [Step1 处理] Hello World"
#      step3: step3_output = "[Step3 处理] [Step2 处理] [Step1 处理] Hello World"
# ============================================================
print("开始执行工作流...\n")
result = app.invoke({"input": "Hello World"})

# result 是跑完后的完整 State（一本填满的记事本）
print(f"\n最终结果: {result['step3_output']}")


# ============================================================
# 9. 把流程图保存成 PNG（这一步和工作流执行无关，只是画图）
#    app.get_graph()                         取出图的结构（节点+边）
#    xray=True                               如果有嵌套子图，把内部也画出来
#    .draw_mermaid_png(...)                  渲染成 PNG 的二进制数据
#    draw_method=MermaidDrawMethod.API       用网上的 Mermaid 服务画图（需要联网）
# ============================================================
with open("workflow-images/01-顺序工作流.png", "wb") as f:
    f.write(app.get_graph(xray=True).draw_mermaid_png(draw_method=MermaidDrawMethod.API))
