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

# 使用 .env.example 作为构建时的环境配置（应用通过 load_dotenv 加载 .env）
COPY .env.example .env

ARG REFRESH_DATE=1
RUN pip install --no-cache-dir -r requirement_customer.txt

EXPOSE 5010

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5010"]