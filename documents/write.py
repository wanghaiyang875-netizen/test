import multiprocessing
import os
from multiprocessing import Queue
import chromadb
from documents.markdown_parser import MarkdownParser
from langchain_chroma import Chroma
from llm_models.embeddings_model import embeddings
from utils.log_utils import log


# 采用分布式，多进程的方式把海量数据写入Milvus数据库

def file_parser_process(dir_path: str, output_queue: Queue, batch_size: int = 20):
    """进程1：解析目录下所有md文件并分批放入队列"""
    log.info(f"解析进程开始扫描目录: {dir_path}")

    # 获取目录下所有.md文件
    md_files = [
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if f.endswith('.md')
    ]

    if not md_files:
        log.warning("警告：未找到任何.md文件")
        output_queue.put(None)  # 发送终止信号
        return

    parser = MarkdownParser()
    doc_batch = []
    for file_path in md_files:
        try:
            docs = parser.parse_markdown_to_documents(file_path)
            if docs:
                doc_batch.extend(docs)

            # 达到批次大小时发送 到队列中
            if len(doc_batch) >= batch_size:
                output_queue.put(doc_batch.copy())
                doc_batch.clear()  # 清空当前缓冲区的所有批次数据
        except Exception as e:
            log.error(f"解析失败 {file_path}: {str(e)}")
            log.exception(e)

    # 发送剩余文档
    if doc_batch:
        output_queue.put(doc_batch)

    # 发送终止信号
    output_queue.put(None)
    log.info(f"解析完成，共处理{len(md_files)}个文件")

'''
def milvus_writer_process(input_queue: Queue):
    """进程2：从队列读取并写入Milvus"""
    log.info("Milvus写入进程启动...")


    mv = MilvusVectorSave()
    mv.create_connection()
    total_count = 0
    while True:
        try:
            datas = input_queue.get()  # 阻塞的函数
            if datas is None:  # 收到了终止的信号
                break

            if isinstance(datas, list):
                mv.add_documents(datas)
                total_count += len(datas)
                log.info(f"累计已写入: {total_count} 个文档")
        except Exception as e:
            log.error(f"写入数据是吧 ！")
            log.exception(e)

    log.info(f"写入进程结束，总计写入 {total_count} 个文档")

'''
def chromadb_writer_process(input_queue: Queue):
    """进程2：从队列读取并写入ChromaDB（使用LangChain + 通义千问Embedding）"""
    log.info("ChromaDB写入进程启动...")

    # 1. 使用 LangChain 的 Chroma 初始化向量库
    # ⚠️ 注意：collection_name 必须和检索工具里的一致，这里统一用 "pdf_docs"
    vector_store = Chroma(
        collection_name="pdf_docs",          # 集合名称
        embedding_function=embeddings,       # 关键：指定使用通义千问的嵌入模型
        persist_directory="./chroma_db"      # 持久化路径
    )

    total_count = 0
    while True:
        try:
            datas = input_queue.get()  # 阻塞的函数
            if datas is None:  # 收到了终止的信号
                break

            if isinstance(datas, list) and len(datas) > 0:
                # 2. 直接批量写入 Document 对象
                # LangChain 会自动提取 page_content 作为文本，metadata 作为元数据
                # 如果你需要指定 ID，可以在解析器里将 id 注入 metadata 的 "id" 字段
                
                # （可选）检查一下 metadata 里是否有不符合 Chroma 要求的复杂类型
                for doc in datas:
                    # 确保 metadata 是字典
                    if not isinstance(doc.metadata, dict):
                        doc.metadata = {}
                        
                vector_store.add_documents(datas)

                total_count += len(datas)
                log.info(f"累计已写入: {total_count} 个文档")

        except Exception as e:
            log.error(f"写入数据失败！")
            log.exception(e)

    log.info(f"写入进程结束，总计写入 {total_count} 个文档")
    
if __name__ == '__main__':
    '''
    # 配置参数
    md_dir = r'E:\my_project\langchain_demo01\md'  # Markdown文件目录
    queue_maxsize = 20  # 队列最大容量（防止内存溢出）

    mv = MilvusVectorSave()
    mv.create_collection()

    # 创建进程间通信队列
    docs_queue = Queue(maxsize=queue_maxsize)

    # 启动子进程
    parser_proc = multiprocessing.Process(
        target=file_parser_process,
        args=(md_dir, docs_queue)
    )
    writer_proc = multiprocessing.Process(
        target=milvus_writer_process,
        args=(docs_queue,)
    )

    parser_proc.start()
    writer_proc.start()

    # 等待进程结束
    parser_proc.join()
    writer_proc.join()

    print("系统提示：所有任务完成")
    '''
    # 配置参数
    md_dir = r'D:\RAG\RAG_PROJECT\RAG_PROJECT\datas\md'  # Markdown文件目录
    queue_maxsize = 20  # 队列最大容量（防止内存溢出）

    # 【修改点1】：删除主进程中的 Milvus 初始化代码
    # 原代码：mv = MilvusVectorSave(); mv.create_collection()
    
    # 可选：如果你希望在主进程提前确保 Collection 存在，可以加上这两行
    # 但更推荐直接交给子进程内部处理（子进程使用 get_or_create_collection）
    # client = chromadb.PersistentClient(path="./chroma_db")
    # client.get_or_create_collection(name="rag_collection")

    # 创建进程间通信队列
    docs_queue = Queue(maxsize=queue_maxsize)

    # 启动子进程
    parser_proc = multiprocessing.Process(
        target=file_parser_process,
        args=(md_dir, docs_queue)
    )
    
    # 【修改点2】：将 target 改为上一轮改好的 chromadb_writer_process
    writer_proc = multiprocessing.Process(
        target=chromadb_writer_process,
        args=(docs_queue,)
    )

    parser_proc.start()
    writer_proc.start()

    # 等待进程结束
    parser_proc.join()
    writer_proc.join()

    print("系统提示：所有任务完成")