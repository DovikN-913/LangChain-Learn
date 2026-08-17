"""
知识点：Text Splitter 把长 Document 切成可检索的小块

官方文档：
https://docs.langchain.com/oss/python/integrations/splitters
https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter

为什么要分块：
1. 模型有上下文上限，不能把整本制度一次塞进 Prompt。
2. 检索粒度：小块更容易和问题对齐；整篇制度的向量会被「平均」掉关键句。
3. 成本：只把命中的几块送给模型，而不是全文。

本示例主要学习：
1. 官方推荐起点是 RecursiveCharacterTextSplitter，而不是 CharacterTextSplitter。
2. split_text 返回 list[str]；split_documents / create_documents 返回 list[Document]，保留 metadata。
3. chunk_size 按 length_function 计（默认是字符数）；TokenTextSplitter 才按 token。
4. chunk_overlap 用来避免一句话被切在两块边界上，检索时丢上下文。
5. 中文默认分隔符列表以空格为主，没有空格的中文要自己加入 。！？，。

RAG 里真正入库的是 split_documents 的结果，不是 split_text。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 工程实践：本文件在 LangChain-RAG/ 子目录，插入项目根才能 import utils。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

from utils import print_stage

DATA_DIR = Path(__file__).resolve().parent / "knowledge_docs"
TXT_PATH = DATA_DIR / "机房安全与出入环境管理制度.txt"


def show_chunks(chunks: list, *, n: int = 2) -> None:
    """对照不同分块器时，看「切成几块、每块多长、有没有带上 metadata」。

    为什么看 length 和 metadata：切太碎检索噪声大；切太长关键句被稀释。
    split_text 的元素是 str（无来源），split_documents 才是带 source 的 Document。
    """
    print(f"切出 {len(chunks)} 块")
    for i, chunk in enumerate(chunks[:n]):
        text = chunk.page_content if hasattr(chunk, "page_content") else str(chunk)
        meta = getattr(chunk, "metadata", None)
        print(f"\n--- 第 {i} 块  length={len(text)} ---")
        if meta:
            print("metadata:", meta)
        print(text[:180].replace("\n", " "), "...")


def main() -> None:
    print_stage("1. 先加载，再分块")
    # 官方文档：Splitter 吃的是文本或 Document，不是文件路径。
    # 工程实践：Loader → Splitter → VectorStore，三条流水线不要反着来。
    loader = TextLoader(file_path=str(TXT_PATH), encoding="utf-8")
    docs = loader.load()
    text = docs[0].page_content
    print(f"原文 {len(text)} 字，Document 数={len(docs)}")
    print("metadata:", docs[0].metadata)
    print("预览:", text[:80].replace("\n", " "))
    print()

    print_stage("2. RecursiveCharacterTextSplitter（官方推荐）")
    # 官方文档：按 separators 从左到右试切，能在 chunk_size 内切开就停，否则换下一个分隔符。
    # 默认 separators=["\n\n", "\n", " ", ""]，对英文段落/句子/单词很合适。
    # 源码分析：最后一项 "" 表示退化为按字符硬切，保证一定能小于 chunk_size。
    # 中文几乎没有空格，若不加 。！？，很容易直接落到按字硬切，语义被撕碎。
    recursive = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        length_function=len,
    )
    rec_texts = recursive.split_text(text)
    show_chunks(rec_texts)
    print()

    print_stage("2.1 split_text vs split_documents")
    # 官方文档：split_text → list[str]；create_documents / split_documents → list[Document]。
    # 源码分析：split_documents 对每条 Document 取 page_content 去切，再把原 metadata 复制到每一块。
    # 还会写入 start_index（若 add_start_index=True）。source 因此能从检索结果追回原文件。
    # 和 split_text 的区别：字符串块不知道自己从哪来，RAG 引用出处时只能用 split_documents。
    rec_docs = recursive.split_documents(docs)
    show_chunks(rec_docs, n=1)
    print(
        "对比：split_text 的元素是",
        type(rec_texts[0]).__name__,
        "；split_documents 的元素是",
        type(rec_docs[0]).__name__,
    )
    print()

    print_stage("3. CharacterTextSplitter（只认一个分隔符）")
    # 官方文档：这是按「单一 separator」切开后再拼到 chunk_size 的长度型切法。
    # 源码分析：不会递归降级。若某一段本身就超过 chunk_size（一整段没有换行），
    # 这块会超长留下，或按实现发出警告——它不会像 Recursive 那样继续按句号切。
    # 适用：日志、CSV 行、已经用统一分隔符组织好的文本。通用中文制度文档不推荐。
    char_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
    )
    char_chunks = char_splitter.split_text(text)
    show_chunks(char_chunks, n=1)
    print(
        f"同一篇制度：Recursive={len(rec_texts)} 块，"
        f"Character(只按换行)={len(char_chunks)} 块。"
    )
    print()

    print_stage("4. TokenTextSplitter（按 token，不是按字）")
    # 官方文档：模型上下文按 token 计费/限长，按字符切只能近似。
    # 源码分析：默认走 tiktoken 的 gpt-3.5/cl100k 一类编码。中文常见 1 字 ≈ 1～2 token，
    # 和 Qwen 自己的分词并不一致，只能当「别超模型窗口」的粗估计。
    # chunk_size=200 表示约 200 个 token，不是 200 个汉字。
    token_splitter = TokenTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
    )
    token_chunks = token_splitter.split_text(text)
    show_chunks(token_chunks, n=1)
    print("不要拿 Token 的 200 和 Recursive 的 500 直接比块数，计量单位不同。")
    print()

    print_stage("5. 中文分隔符：有无 。 的差别")
    # 官方文档专门提醒：中日泰等「词之间没有空格」的文字，要覆盖标点，否则会切碎词。
    # 默认 separators 在段落/换行都用尽后，下一个是英文空格，中文对不上，就会按字符硬切。
    # 下面故意用「没有换行、没有空格」的长句，才能看见两种切法的边界差在哪。
    sample = (
        "机房实行严格的权限分级准入制度，采用门禁刷卡与人脸核验双重认证机制。"
        "所有授权人员仅可在授权时段、授权范围内进入机房。"
        "普通员工以及外包无权限人员严禁私自进入机房核心区域。"
        "运维人员不得转借门禁卡，不得协助无权限人员刷门禁进入机房。"
    )
    default_splitter = RecursiveCharacterTextSplitter(
        chunk_size=40,
        chunk_overlap=0,
    )
    chinese_splitter = RecursiveCharacterTextSplitter(
        chunk_size=40,
        chunk_overlap=0,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    print("默认分隔符切块：")
    for i, chunk in enumerate(default_splitter.split_text(sample)):
        print(f"  [{i}] {chunk}")
    print("加上中文标点后：")
    for i, chunk in enumerate(chinese_splitter.split_text(sample)):
        print(f"  [{i}] {chunk}")
    print()

    print_stage("6. 不同场景的 chunk_size / overlap（同一套算法，不同参数）")
    # 工程实践：没有万能数字。overlap 通常取 chunk_size 的 10%～20%。
    # 太小：边界句子检索不到完整条件；太大：重复入库，召回变吵、费用变高。
    faq_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    report_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    print("短文本/FAQ  200/20  →", len(faq_splitter.split_text(text)), "块")
    print("长制度/报告 1000/150 →", len(report_splitter.split_text(text)), "块")
    print()

    print_stage("7. 代码要用代码分隔符，不要用中文句号")
    # 官方文档：结构型文档（Markdown / HTML / 代码 / JSON）应按结构切。
    # RecursiveCharacterTextSplitter.from_language("python") 会换成 class/def 分隔符。
    # 这里手写一份，方便对照：先按 class/def，再按空行，避免把一个函数从中间切开。
    code = '''
class RoomGuard:
    def check_badge(self, user_id: str) -> bool:
        return user_id in self.allowed

    def record_entry(self, user_id: str) -> None:
        self.log.append(user_id)
'''
    code_splitter = RecursiveCharacterTextSplitter(
        chunk_size=120,
        chunk_overlap=20,
        separators=["\nclass ", "\ndef ", "\n\n", "\n", " "],
    )
    show_chunks(code_splitter.split_text(code), n=3)
    print()

    print_stage("知识点总结")
    print(
        "通用文本先用 RecursiveCharacterTextSplitter。\n"
        "入库走 split_documents，保留 source。\n"
        "中文 separators 加上 。！？，；。\n"
        "chunk_size 对 Recursive/Character 是字符，对 TokenTextSplitter 是 token。"
    )


if __name__ == "__main__":
    main()
