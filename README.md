# Graph RAG 问答项目

基于 LangGraph、LangChain 和 ChromaDB 的检索增强问答（RAG）项目，提供文档检索、相关性评估、问题改写和回答生成能力，并通过 FastAPI 提供问答接口。

## 功能

- **文档问答**：从本地 ChromaDB 向量库检索资料，结合大模型生成回答。
- **工作流编排**：使用 LangGraph 串联检索、相关性判断、问题改写与答案生成。
- **HTTP 接口**：提供健康检查和问答接口，支持通过 `thread_id` 区分会话。
- **扩展流程**：`graph2/` 包含问题路由、Tavily 网页搜索和回答评估等实验功能。
- **文档入库**：解析 Markdown 文档，使用通义千问 Embedding 写入 ChromaDB。

## 项目结构

```text
agent/              工具调用 Agent
documents/          Markdown 解析与向量库写入
graph/              RAG 工作流与 FastAPI 接口
graph2/             问题路由及网页搜索工作流
llm_models/         聊天模型和 Embedding 配置
tools/              文档检索工具
utils/              日志等工具
Dockerfile          API 镜像
docker-compose.yml  Docker Compose 配置
requirements.txt    Python 依赖
```

## 快速开始

### 1. 安装依赖

建议使用 Python 3.12。在项目根目录执行：

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux：source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 配置环境变量

在项目根目录创建 `.env` 文件，填写所使用服务的密钥与地址：

```dotenv
# 聊天模型（OpenAI 兼容接口）
my=你的聊天模型API密钥
my_url=https://你的服务地址/v1

# 文档向量化（DashScope 通义千问）
QWEN_API_key=你的DashScope_API_Key

# 使用 graph2 网页搜索时填写
TAVILY_API_KEY=你的Tavily_API_Key

# 使用当前 Docker Compose 配置时填写
DEEPSEEK_API_KEY=你的DeepSeek_API_Key
BASE_URL=https://你的DeepSeek服务地址/v1
```

聊天模型配置对应 `llm_models/all_model.py`；本地 `.env` 已被 Git 忽略，不要将真实密钥提交到仓库。

### 3. 启动服务

```bash
python -m uvicorn graph.api:app --host 0.0.0.0 --port 8000
```

访问 `http://localhost:8000/docs` 查看交互式 API 文档，或使用以下命令：

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"文档主要讲什么？","thread_id":"demo-1"}'
```

也可以运行 `python -m graph.graph1` 使用命令行问答；输入 `q`、`exit` 或 `quit` 退出。

## 导入 Markdown 文档

在 `documents/write.py` 中将末尾的 `md_dir` 设置为 Markdown 文件所在目录，然后在项目根目录运行：

```bash
python -m documents.write
```

文档会写入 `./chroma_db` 下的 `pdf_docs` 集合；问答检索使用相同的集合及 `text-embedding-v3` Embedding 模型。`chroma_db/` 是本地数据目录，不会提交到 GitHub。

## Docker 部署

准备好 `.env` 和 `chroma_db` 目录后，在项目根目录执行：

```bash
docker compose up --build -d
```

服务默认监听 `http://localhost:8000`。Compose 会挂载本地 `chroma_db/` 和 `logs/` 目录，方便保存向量数据与日志。
