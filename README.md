# Agentic RAG Server

RAG 能力的 API 服务，提供检索增强生成接口。

## 安装

### 从源码安装

```bash
pip install "git+https://github.com/CHyfuture/agentic-rag-server.git"
```

若使用 SSH：

```bash
pip install "git+ssh://git@github.com/CHyfuture/agentic-rag-server.git"
```

### 本地开发安装

```bash
pip install -e .
```

## 依赖

- [BaseVector-Core](https://github.com/CHyfuture/BaseVector-Core) - 通过 `git+ssh` 安装

## 运行

```bash
# 使用命令行入口
agentic-rag-server

# 或直接使用 uvicorn
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API

- `GET /health` - 健康检查
