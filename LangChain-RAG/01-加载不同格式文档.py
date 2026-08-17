"""
知识点：Document Loader 把不同格式文件转成统一的 Document

官方文档：
https://docs.langchain.com/oss/python/integrations/document_loaders

本示例主要学习：
1. RAG 的第一步不是切块、也不是向量化，而是把原始文件读成 LangChain Document。
2. 所有 Loader 都实现 BaseLoader：load() 一次读完；lazy_load() 按需流式读。
3. Document 只有两块核心数据：page_content（文本）和 metadata（来源、页码等）。
4. 格式不同，Loader 不同。TextLoader 读 PDF 会得到乱码或直接失败。
5. 目录批量加载时，按后缀指定 loader_cls，比「一个 Unstructured 吃天下」更可控。

依赖（已在 pyproject.toml）：
- PDF：pypdf（PyPDFLoader）
- Word：docx2txt（Docx2txtLoader）
- 自动识别：unstructured（UnstructuredFileLoader，可选，失败不影响前面步骤）
"""

from __future__ import annotations

import sys
from pathlib import Path

# 工程实践：本文件在 LangChain-RAG/ 子目录。Python 默认找不到项目根的 utils.py。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 官方文档：Loader 集成目前仍大量在 langchain-community。
# 工程实践：1.x 会提示 community 正在 sunset，长期应迁到各格式独立包；
# 现阶段 PyPDFLoader / TextLoader / DirectoryLoader 的稳定入口仍是下面这条。
from langchain_community.document_loaders import (
    DirectoryLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredFileLoader,
)
from langchain_core.documents import Document

from utils import print_stage

# 工程实践：路径相对「本文件」而不是进程 cwd。
# 从项目根 `uv run python LangChain-RAG/01-加载不同格式文档.py` 时，
# 若写 "./knowledge_docs/..." 会去项目根找 knowledge_docs，而不是本目录下的 knowledge_docs。
DATA_DIR = Path(__file__).resolve().parent / "knowledge_docs"


def preview(docs: list[Document], *, chars: int = 80) -> None:
    """检查 Loader 有没有真的抽出字，以及 metadata 是否带上 source。

    逻辑：看条数、正文预览、metadata。空列表通常是文件不存在或 glob 没匹配到。
    为什么要先看这个：后面分块 / 向量化都建立在 Document 上，空内容会静默污染知识库。
    """
    print(f"得到 {len(docs)} 个 Document")
    if not docs:
        print("（空列表：文件不存在，或 Loader 没有匹配到任何文件）")
        return
    first = docs[0]
    text = first.page_content.replace("\n", " ")
    print(f"第一条 page_content 预览: {text[:chars]}...")
    print(f"第一条 metadata: {first.metadata}")


def load_one(path: Path, loader):
    """安全地 load 单个文件：缺失就跳过，不让整个示例中断。

    逻辑：path 不存在 → 返回空列表；存在 → loader.load()。
    为什么：教学目录里 PDF/Word 可能还没放进去，txt/md 仍应能跑完后面的阶段。
    """
    if not path.exists():
        print(f"跳过：找不到 {path.name}")
        print(f"请把文件放到：{DATA_DIR}")
        return []
    return loader.load()


def main() -> None:
    print_stage("0. Document 是什么")
    # 官方文档：Loader 的标准产物是 Document，不是 str，也不是 dict。
    # 源码分析：Document 继承自 Serializable；page_content 是检索/分块的正文，
    # metadata 是随块传递的附加信息（source、page、后续还可以自己加部门、密级）。
    # 和普通字符串的区别：切块、写入向量库时 metadata 会跟着走，检索结果才能回溯来源。
    sample = Document(
        page_content="这是一段示例正文。",
        metadata={"source": "manual", "page": 0},
    )
    print(type(sample).__name__, sample)
    print()

    print_stage("1. 加载 PDF（PyPDFLoader）")
    # 官方文档：PDF 有多种 Loader。文字型 PDF 用 PyPDFLoader（底层 pypdf）即可。
    # 源码分析：默认 mode="page"，一页一个 Document，metadata 里通常有 source 和 page。
    # 扫描件/纯图片 PDF 抽不出字，page_content 会是空字符串，不会报错——入库前要检查。
    # 和 TextLoader 的区别：PDF 是二进制，必须走 PDF 解析器。
    pdf_path = DATA_DIR / "IT软硬件资产全生命周期盘点与报废规范.pdf"
    pdf_docs = load_one(pdf_path, PyPDFLoader(file_path=str(pdf_path)))
    preview(pdf_docs)
    print("注意：打印里的「页」只对 PDF 的 page 模式成立；Word/txt 通常整份文件一个 Document。")
    print()

    print_stage("2. 加载 Word（Docx2txtLoader）")
    # 工程实践：.docx 是 ZIP+XML。docx2txt 抽纯文本，丢掉样式、页眉页脚、修订记录。
    # 和 python-docx 自己解析的区别：这里只要正文，不要排版。
    # .doc（老二进制 Word）这个 Loader 处理不了，需要额外转换。
    docx_path = DATA_DIR / "生产数据备份与容灾恢复管理制度.docx"
    docx_docs = load_one(docx_path, Docx2txtLoader(file_path=str(docx_path)))
    preview(docx_docs)
    print()

    print_stage("3. 加载纯文本（TextLoader）")
    # 官方文档：TextLoader 按文本文件读入，默认整份文件一个 Document。
    # 工程实践：Windows 上中文 txt 经常是 GBK。不写 encoding 可能 UnicodeDecodeError。
    # 本仓库样本是 UTF-8，所以显式 encoding="utf-8"。
    txt_path = DATA_DIR / "机房安全与出入环境管理制度.txt"
    txt_docs = load_one(
        txt_path,
        TextLoader(file_path=str(txt_path), encoding="utf-8"),
    )
    preview(txt_docs)

    print_stage("4. 加载 Markdown（仍用 TextLoader）")
    # 工程实践：只要后续自己按标题切，Markdown 当纯文本读就够了。
    # 若要按 # / ## 切并写入 header metadata，应改用 MarkdownHeaderTextSplitter，
    # 那是分块器，不是 Loader，见 02-文本分块.py。
    md_path = DATA_DIR / "服务器故障分级处置与应急响应规程.md"
    md_docs = load_one(
        md_path,
        TextLoader(file_path=str(md_path), encoding="utf-8"),
    )
    preview(md_docs)
    print()

    print_stage("5. DirectoryLoader：目录 + glob + 指定 Loader")
    # 官方文档：DirectoryLoader 按 glob 找文件，每个文件交给 loader_cls 去 load。
    # glob="*.txt" 只匹配 knowledge_docs/ 这一层；要含子目录用 "**/*.txt"。
    # loader_kwargs 会传给每一个 TextLoader，这里必须带 encoding，否则批量中文必炸。
    dir_txt = DirectoryLoader(
        path=str(DATA_DIR),
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    preview(dir_txt.load())
    print()

    print_stage("6. DirectoryLoader + Unstructured：自动识别（可选）")
    # 官方文档：Unstructured 能处理很多格式，DirectoryLoader 默认也常走它。
    # 工程实践：依赖重（nltk、pdf 解析等），Windows 上失败率高，结果也不稳定。
    # 生产更稳妥的做法是阶段 7：按后缀拆 Loader。这里用 try 包住，失败不阻断后面。
    try:
        auto_loader = DirectoryLoader(
            path=str(DATA_DIR),
            glob="**/*.*",
            loader_cls=UnstructuredFileLoader,
            show_progress=True,
            silent_errors=True,
        )
        preview(auto_loader.load())
    except Exception as exc:
        print(f"Unstructured 加载失败（可忽略）：{type(exc).__name__}: {exc}")
    print()

    print_stage("7. 按后缀使用不同 Loader（推荐）")
    # 工程实践：每种格式用它最合适的解析器，最后把 Document 列表拼起来。
    # 源码分析：load() 返回的是 list[Document]，用 + 合并即可；metadata.source 仍指向原文件。
    # 和阶段 6 的区别：解析行为可预期，缺某个可选依赖只影响对应后缀，不会整批失败。
    documents = (
            DirectoryLoader(
                str(DATA_DIR),
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"},
            ).load()
            + DirectoryLoader(
        str(DATA_DIR),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    ).load()
            + DirectoryLoader(
        str(DATA_DIR),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        silent_errors=True,
    ).load()
            + DirectoryLoader(
        str(DATA_DIR),
        glob="**/*.docx",
        loader_cls=Docx2txtLoader,
        silent_errors=True,
    ).load()
    )
    preview(documents)
    print("按来源统计：")
    for doc in documents:
        print(f"  - {Path(doc.metadata.get('source', '')).name}: {len(doc.page_content)} 字")
    print()

    print_stage("8. 看清 Document 结构（给下一步分块用）")
    if not documents:
        print("没有加载到任何文档，无法展示结构。")
        return
    doc = documents[0]
    print("类型:", type(doc))
    print("page_content 前 200 字:\n", doc.page_content[:200], "...", sep="")
    print("metadata:", doc.metadata)
    print(
        "\n下一步：把这份 list[Document] 交给文本分块器。"
        "不要对原始文件再切一次，否则会丢掉 source 等元数据。"
    )
    print()

    print_stage("知识点总结")
    print(
        "Loader 解决「文件 → Document」，不负责切块、不负责向量化。\n"
        "load() 适合小语料；文件很多时用 lazy_load()，避免一次塞进内存。\n"
        "扫描版 PDF 可能抽出空字符串，入库前检查 page_content.strip()。"
    )


if __name__ == "__main__":
    main()
