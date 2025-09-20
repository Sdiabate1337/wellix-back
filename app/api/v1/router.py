"""
Main API v1 router with token management system.
"""

from fastapi import APIRouter
from app.api.v1 import auth, health_profiles, chat, websocket, llm_config, tokens

api_router = APIRouter()

api_router.include_router(
    auth.router,
    prefix="/auth",
    tags=["authentication"]
)

api_router.include_router(
    tokens.router,
    prefix="/tokens",
    tags=["token-management"]
)

api_router.include_router(
    health_profiles.router,
    prefix="/health",
    tags=["health-profiles"]
)

api_router.include_router(
    analysis.router,
    prefix="/analysis",
    tags=["analysis-disabled"]
)

api_router.include_router(
    chat.router,
    prefix="/chat",
    tags=["ai-chat"]
)

api_router.include_router(
    websocket.router,
    prefix="/ws",
    tags=["websocket"]
)

api_router.include_router(
    llm_config.router,
    prefix="/llm",
    tags=["llm-configuration"]
)
