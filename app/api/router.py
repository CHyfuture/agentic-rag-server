"""API 路由聚合配置。"""

from fastapi import APIRouter

from app.api.routes import health_route, retrieval_route

api_router = APIRouter()

# 健康检查 /api/v1/health
api_router.include_router(health_route.router, tags=["health"])

# 检索接口 /api/v1/semantic, /api/v1/keyword, /api/v1/hybrid, /api/v1/fulltext, /api/v1/text_match, /api/v1/phrase_match
api_router.include_router(retrieval_route.router, tags=["retrieval"])
