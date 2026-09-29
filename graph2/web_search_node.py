from langchain_core.documents import Document

from llm_models.all_model import web_search_tool
from utils.log_utils import log


def web_search(state):
    """
    基于优化后的问题进行网络搜索

    Args:
        state (dict): 当前图状态，包含优化后的问题

    Returns:
        state (dict): 更新后的状态，documents字段替换为网络搜索结果
    """
    log.info("---WEB SEARCH---")  # 阶段标识
    question = state["question"]  # 获取优化后的问题

    # 不同版本的 Tavily/LangChain 可能返回字符串、字典或字典列表。
    docs = web_search_tool.invoke({"query": question})

    if isinstance(docs, str):
        text = docs
    elif isinstance(docs, dict):
        items = docs.get("results", docs.get("content", docs))
        if isinstance(items, list):
            text = "\n".join(
                item.get("content", item.get("snippet", str(item)))
                if isinstance(item, dict) else str(item)
                for item in items
            )
        else:
            text = str(items)
    elif isinstance(docs, list):
        text = "\n".join(
            item.get("content", item.get("snippet", str(item)))
            if isinstance(item, dict) else str(item)
            for item in docs
        )
    else:
        text = str(docs)

    if not text.strip():
        text = "未找到可用的网络搜索结果。"
    web_results = Document(page_content=text)

    return {"documents": web_results, "question": question}  # 返回更新状态
