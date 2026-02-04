"""FastAPI 应用入口。"""

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI

from app.api import api_router

app = FastAPI(
    title="Agentic RAG Server",
    description="RAG 能力的 API 服务，提供检索增强生成接口",
    version="0.1.0",
)

# 注册 API 路由 /api/v1/*
app.include_router(api_router, prefix="/api/v1")


def run():
    """命令行启动入口。"""
    import os
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("DEBUG", "false").lower() == "true"
    uvicorn.run("app.main:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    run()
