import os
from langchain_community.tools import TavilySearchResults
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import OpenAI

# 1、读取.env配置文件中的信息。相关的环境变量以.env文件中的优先
load_dotenv(override=True)
#deepseek 报错jason格式 不想改了
# llm = ChatOpenAI(  # openai的
#     model='deepseek-flash',
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     base_url=os.getenv("BASE_URL")
#     )

web_search_tool = TavilySearchResults(max_results=2)
# res = llm.invoke("你好")
# print(res.content)
# llm = ChatOpenAI(  # openai的
#     model='gpt-5.6-sol',
#     temperature=0,
#     api_key=os.getenv("GPT_API_KEY"),
#     base_url=os.getenv("GPT_URL")
#     )

# llm = ChatOpenAI(
#     temperature=0,
#     model='qwen3.8-flash',                 # 或 qwen3.8-max / qwen3.8-plus
#     api_key=os.getenv("QWEN_API_KEY"),     
#     base_url=os.getenv("QWEN_BASE_URL"),   # 必须带 /compatible-mode/v1
#     max_retries=3,
#     timeout=120
# )

llm = ChatOpenAI(  # openai的
    model='gpt-5.6-sol',
    api_key=os.getenv("my"),
    base_url=os.getenv("my_url")
    )
res = llm.invoke("你是谁")
print(res.content)
