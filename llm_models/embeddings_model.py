import os
from dotenv import load_dotenv
import dashscope
from http import HTTPStatus
from typing import List
from langchain_core.embeddings import Embeddings

load_dotenv(override=True)

# 1. 设置 API Key 和 Base URL
dashscope.api_key = os.getenv("QWEN_API_key")
# 注意：dashscope 2.x 以后推荐使用 base_http_api_url，如果你的库版本支持的话
if os.getenv("QWEN_BASE_URL"):
    dashscope.base_http_api_url = os.getenv("QWEN_BASE_URL")


# 2. 自定义 LangChain 兼容的 Embeddings 类
class QwenEmbeddings(Embeddings):
    def __init__(self, model_name: str = "text-embedding-v3"):
        # ⚠️ 注意：通义千问的文本嵌入模型通常是 text-embedding-v1 / v2 / v3
        # 如果你确信用 qwen3.7-text-embedding 也能调通，那也可以保留
        self.model_name = model_name

    def _get_embedding(self, text: str) -> List[float]:
        """调用 DashScope API 获取单个文本的向量"""
        resp = dashscope.TextEmbedding.call(
            model=self.model_name,
            input=text
        )
        if resp.status_code == HTTPStatus.OK:
            return resp.output['embeddings'][0]['embedding']
        else:
            raise RuntimeError(f"DashScope API 调用失败: {resp.code} - {resp.message}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文档（LangChain 写入和检索时会调用）"""
        # DashScope 支持批量传入，但为了稳定，这里可以循环调用或分批调用
        embeddings = []
        # 简单起见，每次传一条，防止超出批量限制
        for text in texts:
            embeddings.append(self._get_embedding(text))
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """嵌入单条查询（用户提问时会调用）"""
        return self._get_embedding(text)


# 3. 实例化并导出 LangChain 需要使用的 embeddings 对象
embeddings = QwenEmbeddings(model_name="text-embedding-v3")