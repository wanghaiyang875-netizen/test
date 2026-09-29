"""FastAPI HTTP 入口。

启动方式（项目根目录）：
    uvicorn graph.api:app --host 0.0.0.0 --port 8000
"""

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from graph.graph1 import graph


app = FastAPI(
    title="Graph RAG API",
    description="基于 LangGraph 的检索增强问答接口",
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    thread_id: Optional[str] = Field(
        default="default",
        min_length=1,
        max_length=100,
        description="会话 ID；相同 ID 会复用 LangGraph 检查点",
    )


class AskResponse(BaseModel):
    question: str
    answer: str
    thread_id: str


@app.get("/health")
def health():
    """服务健康检查。"""
    return {"status": "ok"}


def _invoke_graph(question: str, thread_id: str) -> dict:
    """在线程池中执行同步 LangGraph 调用。"""
    return graph.invoke(
        {"messages": [("user", question)]},
        config={"configurable": {"thread_id": thread_id}},
    )


def _extract_answer(result: dict) -> str:
    """从工作流最终消息中提取文本，兼容字符串和消息对象。"""
    messages = result.get("messages", [])
    if not messages:
        return ""
    content = messages[-1].content
    if isinstance(content, str):
        return content
    return str(content)


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """执行一次 RAG 问答。"""
    try:
        result = await run_in_threadpool(
            _invoke_graph,
            request.question,
            request.thread_id or "default",
        )
        return AskResponse(
            question=request.question,
            answer=_extract_answer(result),
            thread_id=request.thread_id or "default",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG 处理失败：{exc}") from exc
