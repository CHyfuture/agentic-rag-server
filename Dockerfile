FROM agentic_base:latest

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UVICORN_WORKERS=2

WORKDIR /app
#
#RUN apt-get update && apt-get install -y --no-install-recommends \
#    build-essential \
#    && rm -rf /var/lib/apt/lists/*
#
#COPY requirements.txt pyproject.toml ./
#COPY BaseVector-Core ./BaseVector-Core
#
#RUN pip install --upgrade pip && \
#    pip install -r requirements.txt

COPY . .

ENV HOST=0.0.0.0 \
    PORT=8000 \
    MILVUS_HOST=192.168.31.51 \
    MILVUS_PORT=19530 \
    MILVUS_DB_NAME=default \
    COLLECTION_NAME=papers_chunks_collection     \
    EMBEDDING_MODEL=workspace/bge-base-zh-v1.5

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]