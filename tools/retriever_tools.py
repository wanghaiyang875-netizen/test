import chromadb
from langchain_chroma import Chroma
from langchain_core.tools import create_retriever_tool

# 1. 导入你写入时用的 embeddings 模型（必须和写入时保持一致！）
from llm_models.embeddings_model import embeddings

# 2. 用 LangChain 的 Chroma 类来加载本地数据库
vector_store = Chroma(
    collection_name="pdf_docs",       # 和你写入时的 collection 名字一致
    embedding_function=embeddings,    # 关键！必须和写入时用的 embedding 模型相同
    persist_directory="./chroma_db"   # 你的数据库文件夹路径
)

# 3. 转换为 LangChain 的 Retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}            # 检索返回最相似的 3 条
)

# 4. 创建 Agent 能用的工具
retriever_tool = create_retriever_tool(
    retriever,
    "pdf_docs_retriever",
    "用于检索本地文档库，回答关于文档内容的问题。"
)