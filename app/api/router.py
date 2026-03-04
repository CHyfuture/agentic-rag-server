"""API 路由聚合配置。"""

from fastapi import APIRouter

from app.api.routes import (
    health_route,
    index_route,
    operator_route,
    questions_answers_route,
    retrieval_route,
)

api_router = APIRouter()

# 健康检查 /api/v1/health
api_router.include_router(health_route.router, tags=["health"])

# 检索接口 /api/v1/semantic, /api/v1/keyword, /api/v1/hybrid, /api/v1/fulltext, /api/v1/text_match, /api/v1/phrase_match
api_router.include_router(retrieval_route.router, tags=["retrieval"])
# 算子接口 /api/v1/parser/parse, /api/v1/chunker/chunk, /api/v1/storage/*
api_router.include_router(operator_route.router, tags=["operator"])
api_router.include_router(questions_answers_route.router, tags=["qa"])

# 索引构建接口 /api/v1/index/build
api_router.include_router(index_route.router, tags=["index"])
