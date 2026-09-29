# Graph RAG 问答项目

基于 LangGraph、LangChain 和 ChromaDB 的检索增强问答（RAG）实验项目。主流程提供 FastAPI 问答接口与命令行入口；仓库还包含另一套带问题路由、网页搜索及回答评估的实验流程。

> 本仓库原名为 `test`，代码来自 `ragtest` 项目。它仍处于实验阶段，部分脚本包含本地绝对路径或会在导入时调用模型 API；使用前请阅读下文的注意事项。

## 功能与流程

- **主流程 `graph/`**：模型判断是否调用本地文档检索工具 → ChromaDB 检索 → 评估文档相关性 → 生成回答；若文档不相关，则改写问题后重试。
- **接口 `graph/api.py`**：`GET /health` 健康检查，`POST /ask` 提交问题；同一个 `thread_id` 使用进程内的 LangGraph 检查点保存会话状态。
- **实验流程 `graph2/`**：按问题选择本地向量库或 Tavily 网页搜索，并对生成结果进行评估和重试。目前没有接入上述 FastAPI 接口。
- **文档入库 `documents/write.py`**：解析 Markdown 文档，调用通义千问文本嵌入接口，写入本地 ChromaDB 的 `pdf_docs` 集合。

## 项目结构

```text
agent/          独立的工具调用 Agent 实验
documents/      Markdown 解析与 ChromaDB 写入脚本
graph/          主 RAG 工作流及 FastAPI 接口
graph2/         带问题路由和网页搜索的实验工作流
llm_models/     聊天模型和 Embedding 配置
tools/          ChromaDB 检索工具
utils/          日志等辅助代码
Dockerfile      API 镜像
docker-compose.yml  Docker Compose 配置
requirements.txt    Python 依赖
```

> `chroma_db/`、`logs/` 和 `.env` 是本地运行数据/配置，未纳入 Git；克隆后需要自行准备。

## 准备环境

- Python 3.12（与 Dockerfile 保持一致），或 Docker 与 Docker Compose。
- 可调用的 OpenAI 兼容聊天模型服务；文档向量化需要阿里云 DashScope 通义千问 API。
- 如果运行 `graph2/` 的网页搜索流程，还需要 Tavily API Key。

在项目根目录建立 `.env`，填入自己的真实凭据（不要提交到 Git）：

```dotenv
# 当前实际聊天模型配置：见 llm_models/all_model.py
my=你的聊天模型API密钥
my_url=https://你的OpenAI兼容服务地址/v1

# 文档向量化：变量名的 key 为小写，与代码一致
QWEN_API_key=你的DashScope_API_Key
# 非必填；仅在确实需要覆盖 DashScope 接口地址时填写
# QWEN_BASE_URL=https://你的DashScope接口地址

# 使用 graph2 网页搜索时填写
TAVILY_API_KEY=你的Tavily_API_Key

# docker-compose.yml 引用的变量（当前聊天模型实现不使用它们）
DEEPSEEK_API_KEY=你的DeepSeek_API_Key
BASE_URL=https://你的DeepSeek兼容服务地址/v1
```

`.env` 已在 `.gitignore` 和 `.dockerignore` 中排除。若无需 Compose 的 DeepSeek 变量，也应按需调整 `docker-compose.yml` 中的变量引用，避免空值配置；**当前聊天模型实际读取 `my` / `my_url`，不是 `DEEPSEEK_API_KEY` / `BASE_URL`。**

## 本地运行

在项目根目录执行：

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn graph.api:app --host 0.0.0.0 --port 8000
```

另一个终端中测试接口：

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"文档主要讲什么？","thread_id":"demo-1"}'
```

交互式主流程可运行 `python -m graph.graph1`；输入 `q`、`exit` 或 `quit` 退出。交互式实验流程可运行 `python -m graph2.graph_2`（需要配置 Tavily 搜索）。

`/health` 返回 `{"status":"ok"}` 只能说明 HTTP 服务在运行，不代表模型凭据与向量库已经可用；请通过 `/ask` 实际检验问答链路。

## Docker Compose

在项目根目录准备 `.env` 后运行：

```bash
docker compose up --build -d
curl http://localhost:8000/health
```

Compose 会将本地 `./chroma_db` 挂载到容器内 `/app/chroma_db`，并挂载 `./logs`。仓库**不包含**已建好的向量库：首次启动前需先导入文档，否则本地检索没有可用内容。`Dockerfile` 中的 `COPY chroma_db` 需要构建上下文存在该目录；如尚未导入，请先在项目根目录创建空的 `chroma_db` 目录。

## 导入自己的 Markdown 文档

将 `.md` 文件放入一个本地目录；修改 `documents/write.py` 末尾 `md_dir` 变量为该目录的实际路径，然后在项目根目录运行：

```bash
python -m documents.write
```

入库和查询都使用 `./chroma_db` 下的 `pdf_docs` 集合，以及相同的通义千问 `text-embedding-v3` 模型。脚本目前内置一个开发机器上的 `D:\RAG\...` 路径，**克隆后不修改会无法找到文档**。建议先在宿主机完成入库，再启动 Compose，让容器通过挂载复用该数据库。

## 当前限制

- `llm_models/all_model.py` 在导入模块时会立即向聊天模型发送一次测试请求，因此即使只启动服务，也会依赖有效凭据与可用网络，并产生 API 调用。
- `graph2/graph_2.py` 在模块顶层进入交互循环，不适合作为服务模块直接导入；`graph2/` 属于独立实验功能。
- `agent/rag_agent.py` 引用了 `langchain_classic`，但当前 `requirements.txt` 未显式列出它；该文件不属于主 API 启动流程。
- `documents/chroma_db.py` 保留了混用 Milvus/Chroma 的旧实验代码，不是当前写入脚本的推荐入口。
- 检查点为进程内 `MemorySaver`，服务重启后会话状态不会保留；本项目没有内置鉴权或生产部署配置。

不要在公开仓库、截图或日志中暴露 `.env`、API Key 或用户私有文档。相关接口可能产生第三方服务费用。
