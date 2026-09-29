FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent ./agent
COPY documents ./documents
COPY graph ./graph
COPY graph2 ./graph2
COPY llm_models ./llm_models
COPY tools ./tools
COPY utils ./utils
COPY chroma_db ./chroma_db

EXPOSE 8000

CMD ["uvicorn", "graph.api:app", "--host", "0.0.0.0", "--port", "8000"]
