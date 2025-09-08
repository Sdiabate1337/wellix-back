"""
Main API v1 router that includes all endpoint routers.
"""

from fastapi import APIRouter
from app.api.v1 import auth, health_profiles, analysis, chat, websocket

api_router = APIRouter()

# Include all routers with their prefixes
api_router.include_router(
    auth.router,
    prefix="/auth",
    tags=["authentication"]
)

api_router.include_router(
    health_profiles.router,
    prefix="/health",
    tags=["health-profiles"]
)

api_router.include_router(
    analysis.router,
    prefix="/analysis",
    tags=["food-analysis"]
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
