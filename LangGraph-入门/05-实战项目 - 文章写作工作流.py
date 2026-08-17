"""
LangGraph 入门：实战 —— 文章写作工作流

前面 01~04 是零件，这份把它们拼在一起，并且第一次真正接上 LLM。

整张图：

    入口 → plan → write → assemble → evaluate
                                      │
                    ┌─────────────────┼─────────────────┐
                    │ 分数 < 8 且还能改                  │ 分数够 / 改不动了
                    ▼                                   ▼
                 revise                              review（人审）
                    │                                   │
                    │                    ┌──────────────┼──────────────┐
                    │                    │ 人拒绝                         │ 人批准
                    │                    ▼                               ▼
                    └──────────────► assemble                          save → END

你会看到 4 种图技巧同时出现：
- 顺序：plan → write → assemble → evaluate
- 条件：evaluate 之后去 review 还是 revise
- 循环：revise → assemble → evaluate ...
- 人机：review 里用 input() 等人拍板

和 01~04 的最大差别：节点里会调用大模型，不再是假数据。
"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.runnables.graph import MermaidDrawMethod
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import json

# 从 .env 读 SILICONFLOW_API_KEY / SILICONFLOW_MODEL_NAME
load_dotenv()


# ============================================================
# 1. Pydantic 数据模型（给 LLM 用的「答题卡」）
#    和 State 不是一回事：
#    - State     = 整张图流转的记事本
#    - BaseModel = 某一次模型调用时，要求它按这个 JSON 结构回答
#
#    后面会用 model.with_structured_output(ArticleOutline)
#    模型就不会回一段散文，而是直接变成 ArticleOutline 对象。
# ============================================================
class OutlineSection(BaseModel):
    """大纲里的一个章节。"""
    title: str = Field(description="章节标题")
    key_points: List[str] = Field(description="关键要点")


class ArticleOutline(BaseModel):
    """整篇大纲：标题 + 引言 + 若干章节 + 结论。"""
    title: str = Field(description="文章标题")
    introduction: str = Field(description="引言")
    sections: List[OutlineSection] = Field(description="章节列表")
    conclusion: str = Field(description="结论")


class QualityScore(BaseModel):
    """质量评分。ge/le 限制分数必须在 1~10。"""
    clarity: int = Field(description="清晰度1-10", ge=1, le=10)
    coherence: int = Field(description="连贯性1-10", ge=1, le=10)
    depth: int = Field(description="深度1-10", ge=1, le=10)
    overall: int = Field(description="总分1-10", ge=1, le=10)
    feedback: str = Field(description="改进建议")


# ============================================================
# 2. 图的状态
#    Optional[...] 表示「一开始可以是 None，后面节点再填上」。
#    比如刚 invoke 时还没有大纲，plan 节点跑完才有 outline。
# ============================================================
class WritingState(TypedDict):
    topic: str                            # 写作主题（唯一一开始就必须有的）
    outline: Optional[ArticleOutline]     # 规划出来的大纲
    sections_content: List[str]           # 各段正文（引言/章节/结论）
    full_article: str                     # 拼好的完整 Markdown
    quality_score: Optional[QualityScore] # 自动评分
    revision_count: int                   # 已经改了几轮（防止死循环）
    approved: bool                        # 人是否批准发布
    human_feedback: str                   # 人的修改意见；自动修订时也会借用这个字段


# ============================================================
# 3. 创建模型
#    教学写法：每个需要 LLM 的节点里都调用一次 create_model()。
#    正常项目更常见的是：在外面创建一次，通过参数 / 依赖注入传进去，
#    避免重复初始化，也方便单测时换成假模型。
# ============================================================
def create_model():
    """连 SiliconFlow（OpenAI 兼容接口）。温度 0.7 = 允许一点发挥。"""
    return init_chat_model(
        model=os.getenv("SILICONFLOW_MODEL_NAME"),
        model_provider="openai",
        base_url="https://api.siliconflow.cn/v1",
        api_key=os.getenv("SILICONFLOW_API_KEY"),
        temperature=0.7,
    )


# ============================================================
# 4. 工作流节点（每个节点只做一件事）
# ============================================================
def plan_outline(state: WritingState) -> WritingState:
    """节点 1：让模型按 ArticleOutline 的结构吐出大纲。"""
    print(f"\n📋 规划大纲: {state['topic']}")

    model = create_model()
    # with_structured_output：强制模型按 Pydantic 模型返回，不是自由文本
    structured_model = model.with_structured_output(ArticleOutline)

    prompt = f"""请为以下主题创建详细的文章大纲：

      主题：{state['topic']}

      要求：
      1. 创建吸引人的标题
      2. 撰写引言（2-3句）
      3. 设计 3-5 个章节，每个章节列出 2-3 个要点
      4. 撰写结论（2-3句）

      请确保逻辑清晰、结构完整。
      """

    outline = structured_model.invoke(prompt)
    state["outline"] = outline

    print("✅ 大纲创建完成")
    print(f"   标题: {outline.title}")
    print(f"   章节数: {len(outline.sections)}")

    return state


def write_sections(state: WritingState) -> WritingState:
    """节点 2：按大纲逐章写正文。引言和结论直接用大纲里的句子，章节才调模型。"""
    print("\n✍️  撰写文章内容...")

    model = create_model()
    outline = state["outline"]

    sections_content = []

    print("   - 引言")
    sections_content.append(f"## 引言\n\n{outline.introduction}")

    # enumerate(..., 1) 只是为了人读时从 1 计数，循环本身不依赖这个数字
    for i, section in enumerate(outline.sections, 1):
        print(f"   - {section.title}")

        # chr(10) 就是换行符 \n，用来把要点列表拼成多行
        prompt = f"""请撰写文章的这一章节：

        章节标题：{section.title}

        关键要点：
        {chr(10).join([f'- {point}' for point in section.key_points])}

        要求：
        1. 内容详实，每个要点都要充分展开
        2. 语言流畅，逻辑清晰
        3. 字数 300-500 字
        4. 使用 Markdown 格式

        正文：
        """

        content = model.invoke(prompt).content
        sections_content.append(f"## {section.title}\n\n{content}")

    print("   - 结论")
    sections_content.append(f"## 结论\n\n{outline.conclusion}")

    state["sections_content"] = sections_content
    print(f"✅ 内容撰写完成（共 {len(sections_content)} 部分）")
    return state


def assemble_article(state: WritingState) -> WritingState:
    """节点 3：不调模型，只把标题和各段拼成一篇 Markdown。"""
    print("\n🔧 组装文章...")

    outline = state["outline"]
    sections = state["sections_content"]

    article = f"# {outline.title}\n\n"
    article += "\n\n".join(sections)
    article += "\n\n---\n\n"
    article += f"*生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"

    state["full_article"] = article
    print(f"✅ 文章组装完成（共 {len(article)} 字符）")
    return state


def evaluate_quality(state: WritingState) -> WritingState:
    """节点 4：让模型按 QualityScore 给分，并写出改进建议。"""
    print("\n📊 评估文章质量...")

    model = create_model()
    structured_model = model.with_structured_output(QualityScore)

    prompt = f"""请评估以下文章的质量：

      {state['full_article']}

      评估维度：
      1. 清晰度（1-10）：语言是否清晰易懂
      2. 连贯性（1-10）：逻辑是否连贯
      3. 深度（1-10）：内容是否有深度
      4. 总分（1-10）：综合评分

      请提供具体的改进建议。
      """

    score = structured_model.invoke(prompt)
    state["quality_score"] = score

    print("✅ 质量评估完成")
    print(f"   总分: {score.overall}/10")
    print(f"   清晰度: {score.clarity}/10")
    print(f"   连贯性: {score.coherence}/10")
    print(f"   深度: {score.depth}/10")
    return state


def human_review_node(state: WritingState) -> WritingState:
    """节点 5：人机协作。和 04 一样，用 input() 卡住等你决定。"""
    print("\n" + "=" * 70)
    print("👤 人工审核")
    print("=" * 70)

    print(f"\n文章标题: {state['outline'].title}")
    print(f"质量评分: {state['quality_score'].overall}/10")
    print("\n预览前 500 字符:\n")
    print(state["full_article"][:500])
    print("\n...")

    print("\n自动评估建议:")
    print(state["quality_score"].feedback)

    print("\n请审核:")
    print("1. 批准发布（输入 'y'）")
    print("2. 需要修订（输入 'n'）")
    print("3. 自动批准所有后续（输入 'auto'）")

    choice = input("\n您的决定: ").strip().lower()

    # 注意：auto 在这份代码里和 y 效果一样，并不会真的「跳过后面所有审核」
    if choice == "y" or choice == "auto":
        state["approved"] = True
        state["human_feedback"] = ""
        print("✅ 已批准")
    else:
        state["approved"] = False
        state["human_feedback"] = input("\n请提供修改意见: ")
        print("📝 反馈已记录")

    return state


def revise_article(state: WritingState) -> WritingState:
    """节点 6：按 human_feedback 让模型改全文。改完 revision_count + 1。"""
    print("\n🔧 根据反馈修订文章...")
    print(f"   反馈: {state['human_feedback']}")

    model = create_model()

    prompt = f"""请根据以下反馈修订文章：

      原文：
      {state['full_article']}

      修改意见：
      {state['human_feedback']}

      要求：
      1. 针对性改进
      2. 保持原有结构
      3. 提升整体质量

      修订后的文章：
      """

    revised = model.invoke(prompt).content
    state["full_article"] = revised
    state["revision_count"] += 1

    print(f"✅ 修订完成（第 {state['revision_count']} 次）")
    return state


def save_article(state: WritingState) -> WritingState:
    """节点 7：人批准后，把文章和评分元数据写到磁盘。"""
    print("\n💾 保存文章...")

    filename = f"article_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(state["full_article"])

    metadata = {
        "topic": state["topic"],
        "title": state["outline"].title,
        # .dict() 把 Pydantic 对象变成普通字典，才能 json.dump
        "quality_score": state["quality_score"].dict(),
        "revision_count": state["revision_count"],
        "generated_at": datetime.now().isoformat(),
    }

    meta_filename = filename.replace(".md", "_meta.json")
    with open(meta_filename, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("✅ 文章已保存")
    print(f"   文件: {filename}")
    print(f"   元数据: {meta_filename}")
    return state


# ============================================================
# 5. 路由函数（只负责指路，原则上不该改 State）
# ============================================================
def check_quality(state: WritingState) -> str:
    """
    evaluate 之后走哪：
    - 总分 >= 8            → 给人审
    - 已经改过 2 次        → 也给人审（别死循环）
    - 否则                 → 自动修订

    教学提醒：下面这行改了 state["human_feedback"]。
    路由函数里改状态能跑，但不是好习惯；更好的做法是单独做一个节点来写反馈。
    """
    score = state["quality_score"].overall

    if score >= 8:
        print("🌟 质量优秀，进入审核")
        return "review"
    elif state["revision_count"] >= 2:
        print("⚠️  已达最大修订次数，进入审核")
        return "review"
    else:
        print("📝 质量需改进，自动修订")
        state["human_feedback"] = state["quality_score"].feedback
        return "revise"


def check_approval(state: WritingState) -> str:
    """人审之后：批准去保存，拒绝去修订。"""
    return "save" if state["approved"] else "revise"


# ============================================================
# 6. 把节点和边组装成可执行 app
#    包进函数里，是为了 main() 调用；也方便以后单测时 create 多次。
# ============================================================
def create_writing_workflow():
    """创建写作工作流并 compile。"""
    workflow = StateGraph(WritingState)

    workflow.add_node("plan", plan_outline)
    workflow.add_node("write", write_sections)
    workflow.add_node("assemble", assemble_article)
    workflow.add_node("evaluate", evaluate_quality)
    workflow.add_node("review", human_review_node)
    workflow.add_node("revise", revise_article)
    workflow.add_node("save", save_article)

    workflow.set_entry_point("plan")

    # 前半段是固定流水线
    workflow.add_edge("plan", "write")
    workflow.add_edge("write", "assemble")
    workflow.add_edge("assemble", "evaluate")

    # 评估后：够好去人审，不够好去自动改
    workflow.add_conditional_edges(
        "evaluate",
        check_quality,
        {
            "review": "review",
            "revise": "revise",
        },
    )

    # 人审后：批了就存，没批再改
    workflow.add_conditional_edges(
        "review",
        check_approval,
        {
            "save": "save",
            "revise": "revise",
        },
    )

    # 修订后回到 assemble，而不是 write：
    # 因为 revise 已经直接改了 full_article，只需重新评估，不必按大纲重写各章。
    workflow.add_edge("revise", "assemble")
    workflow.add_edge("save", END)

    return workflow.compile()


# ============================================================
# 7. 主程序：创建图 → 画图 → 连续写两篇
# ============================================================
def main():
    print("=" * 70)
    print("📝 智能文章写作工作流（LangChain 1.0 + LangGraph）")
    print("=" * 70)

    app = create_writing_workflow()
    with open("05-文字写作工作流.png", "wb") as f:
        f.write(app.get_graph(xray=True).draw_mermaid_png(draw_method=MermaidDrawMethod.API))

    topics = [
        "人工智能在教育领域的应用与挑战",
        "如何使用 LangChain 构建 AI 应用",
    ]

    for topic in topics:
        print("\n\n" + "=" * 70)
        print(f"🎯 开始写作: {topic}")
        print("=" * 70)

        # 除了 topic，其他字段都先给「空值」，后面节点会填
        initial_state = {
            "topic": topic,
            "outline": None,
            "sections_content": [],
            "full_article": "",
            "quality_score": None,
            "revision_count": 0,
            "approved": False,
            "human_feedback": "",
        }

        result = app.invoke(initial_state)

        print("\n\n" + "=" * 70)
        print("✅ 写作完成！")
        print("=" * 70)
        print(f"主题: {result['topic']}")
        print(f"标题: {result['outline'].title}")
        print(f"质量评分: {result['quality_score'].overall}/10")
        print(f"修订次数: {result['revision_count']}")
        print(f"状态: {'已发布' if result['approved'] else '待处理'}")

        input("\n按 Enter 继续下一篇...")

    # 再存一次图（和上面重复，教学脚本里常会留下这种痕迹）
    try:
        with open("05-文字写作工作流.png", "wb") as f:
            f.write(app.get_graph(xray=True).draw_mermaid_png(draw_method=MermaidDrawMethod.API))
    except Exception as e:
        print(f"流程图渲染失败: {e}")
        print("\n使用 Mermaid 文本方式显示:")
        print(app.get_graph(xray=True).draw_mermaid())


# 只有直接 python 运行本文件时才进 main；被 import 时不会自动写文章
if __name__ == "__main__":
    main()
