import uuid
from typing import Literal, List

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
#科学上网画图
#from draw_png import draw_graph
from graph.agent_node import agent_node
from graph.generate_node import generate
from graph.get_human_message import get_last_human_message
from graph.graph_state1 import AgentState
from graph.rewrite_node import rewrite
from llm_models.all_model import llm
from tools.retriever_tools import retriever_tool
from utils.log_utils import log
from utils.print_utils import _print_event




def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    判断检索到的文档是否与问题相关。
    参数:
        state (messages): 当前状态
    返回:
        str: 判断结果，文档是否相关
    """
    log.info("---检查document的相关性---")
    #llm_with_structured = llm.with_structured_output(Grade, method="json_mode")

    # 提示模板
    prompt = PromptTemplate(
        template="""你是一个评估检索文档与用户问题相关性的评分器。\n
            这是检索到的文档：\n\n {context} \n\n
            这是用户的问题：{question} \n
            如果文档包含与用户问题相关的关键词或语义含义，则评为相关。\n
            给出二元评分 'yes' 或 'no' 来表示文档是否与问题相关。""",
        input_variables=["context", "question"],
    )

    # 不使用 with_structured_output：部分 DeepSeek 兼容接口不支持 response_format。
    # 直接请求普通文本，再从结果中解析 yes/no。
    chain = prompt | llm

    messages = state["messages"]
    last_message = messages[-1]

    question = get_last_human_message(messages).content
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})
    score_text = getattr(scored_result, "content", scored_result)
    score_text = str(score_text).strip().lower()
    score = "yes" if score_text.startswith("yes") else "no"

    if score == "yes":
        print("---输出：文档相关---")
        return "generate"

    else:
        print("---输出：文档不相关---")
        print(score)
        return "rewrite"

def build_graph():
    """构建并编译状态图；导入模块时不启动命令行。"""
    workflow = StateGraph(AgentState)

# 添加节点
    workflow.add_node('agent', agent_node)
    workflow.add_node('retrieve', ToolNode([retriever_tool]))
    workflow.add_node('rewrite', rewrite)
    workflow.add_node('generate', generate)

    workflow.add_edge(START, 'agent')
    workflow.add_conditional_edges(
    'agent',
    tools_condition,
    {
        'tools': 'retrieve',
        END: END
    }
    )

    workflow.add_conditional_edges('retrieve', grade_documents)

    workflow.add_edge('rewrite', 'agent')
    workflow.add_edge('generate', END)


# 检查点让状态图可以持久化其状态
# 这是整个状态图的完整内存
    memory = MemorySaver()

# 编译状态图，配置检查点为memory, 配置中断点
    return workflow.compile(checkpointer=memory)


# draw_graph(graph, 'graph_rag1-2.png')
graph = build_graph()

def run_cli():
    config = {
    "configurable": {
        # 检查点由session_id访问
        "thread_id": str(uuid.uuid4()),
    }
    }

    _printed = set()

# 执行工作流
    while True:
        question = input('用户：').strip()
        if question.lower() in ['q', 'exit', 'quit']:
            log.info('对话结束，拜拜！')
            break
        if not question:
            continue
        events = graph.stream({"messages": [("user", question)]}, config=config, stream_mode='values')
        for event in events:
            _print_event(event, _printed)

if __name__ == '__main__':
    run_cli()
