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

# 1. 安装 git（如果基础镜像里没有的话）
RUN apt-get update && apt-get install -y git openssh-client && rm -rf /var/lib/apt/lists/*

# 2. 自动信任 GitHub 的公钥（解决 Host key verification failed）
RUN mkdir -p -m 0700 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

# 3. 使用 --mount=type=ssh 执行安装
ARG REFRESH_DATE=1
RUN --mount=type=ssh pip install --no-cache-dir -r requirement_customer.txt

RUN pip install python-multipart
RUN pip install peft

EXPOSE 5010

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5010"]
