from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

from llm_models.all_model import llm
from graph2.plain_output import parse_model_output


# 查询的动态路由： 根据用户的提问，决策采用哪种检索策略（网络检索，RAG）


# 数据模型
class RouteQuery(BaseModel):
    """将用户查询路由到最相关的数据源"""
    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="根据用户问题选择将其路由到向量知识库或网络搜索",
    )


# 带函数调用的LLM
structured_llm_router = llm | RunnableLambda(lambda value: parse_model_output(value, RouteQuery))

# 提示词模板
# system = """你是一个擅长将用户问题路由到向量知识库或网络搜索的专家。
# 向量知识库包含与半导体材料，芯片制造，光刻技术相关的文档。
# 对于这些主题的问题请使用向量知识库，其他情况使用网络搜索。"""
# route_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system),  # 系统提示词
#         ("human", "{question}"),  # 用户问题占位符
#     ]
# )
# 提示词模板 —— 关键：必须出现 "json" 字样，并给出 JSON 输出示例
system = """你是一个擅长将用户问题路由到向量知识库或网络搜索的专家。
向量知识库包含与半导体材料，芯片制造，光刻技术相关的文档。
对于这些主题的问题请使用向量知识库，其他情况使用网络搜索。

请务必以 json 格式返回结果，不要包含任何多余的文字或 Markdown 代码块。
返回的 json 结构必须严格遵循如下格式：
{{"datasource": "vectorstore"}}
或
{{"datasource": "web_search"}}

字段取值只能是 "vectorstore" 或 "web_search" 之一。"""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

# 创建问题路由器链
question_router_chain = route_prompt | structured_llm_router


#测试路由器
# print(  # 测试非技术问题（应路由到网络搜索）
#     question_router_chain.invoke(
#         {"question": "什么是EUV光刻技术?"}
#     )
# )
# print(  # 测试技术问题（应路由到向量数据库）
#     question_router_chain.invoke({"question": "今天，长沙的天气怎么样?"})
# )
