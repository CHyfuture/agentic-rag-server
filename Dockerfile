FROM agentic-rag:base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UVICORN_WORKERS=2

WORKDIR /app


COPY . .


# 1. 安装 git（如果基础镜像里没有的话）
RUN apt-get update && apt-get install -y git openssh-client && rm -rf /var/lib/apt/lists/*

# 2. 强制 GitHub SSH 走 443（容器网络常见限制 22 端口）并预写 known_hosts
RUN mkdir -p -m 0700 ~/.ssh \
    && printf "Host github.com\n  HostName ssh.github.com\n  Port 443\n  User git\n" > ~/.ssh/config \
    && chmod 600 ~/.ssh/config \
    && printf "  StrictHostKeyChecking accept-new\n" >> ~/.ssh/config \
    && (ssh-keyscan -p 443 ssh.github.com >> ~/.ssh/known_hosts 2>/dev/null || true)

# 3. 使用 --mount=type=ssh 执行安装
ARG REFRESH_DATE=1
RUN --mount=type=ssh,required pip install --no-cache-dir -r requirement_customer.txt

RUN pip install peft

EXPOSE 5010

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5010"]
