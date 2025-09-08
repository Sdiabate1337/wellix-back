"""
FastAPI dependencies for authentication, database, and common utilities.
"""

from typing import Optional, Generator
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from app.core.security import security_manager
from app.db.database import get_async_session
from app.db.models.user import User
from app.cache.cache_manager import cache_manager
from sqlalchemy import select

logger = structlog.get_logger(__name__)

# Security scheme
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_async_session)
) -> User:
    """
    Get current authenticated user from JWT token.
    """
    try:
        # Extract token
        token = credentials.credentials
        
        # Verify token and extract user ID
        user_id = security_manager.extract_user_id(token)
        
        # Check cache first
        cached_user = await cache_manager.get_user_health_context(user_id)
        if cached_user:
            # Verify user still exists in database
            result = await db.execute(select(User).where(User.id == user_id))
            user = result.scalar_one_or_none()
            if user and user.is_active:
                return user
        
        # Get user from database
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Inactive user"
            )
        
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )


async def get_current_user_from_token(token: str) -> User:
    """Get current user from token string (for WebSocket authentication)."""
    try:
        payload = security_manager.verify_token(token)
        user_id: str = payload.get("sub")
        
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials"
            )
        
        async with get_async_session() as db:
            result = await db.execute(select(User).where(User.id == user_id))
            user = result.scalar_one_or_none()
            
            if user is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found"
                )
        
        return user
        
    except Exception as e:
        logger.error(f"Token validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current active user."""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


async def get_current_premium_user(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """
    Get current premium user (for premium features).
    """
    if not current_user.is_premium:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Premium subscription required"
        )
    return current_user


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_async_session)
) -> Optional[User]:
    """
    Get current user if authenticated, otherwise return None.
    Useful for endpoints that work for both authenticated and anonymous users.
    """
    if not credentials:
        return None
    
    try:
        token = credentials.credentials
        user_id = security_manager.extract_user_id(token)
        
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        
        if user and user.is_active:
            return user
        
    except Exception as e:
        logger.warning(f"Optional authentication failed: {e}")
    
    return None


class RateLimitDependency:
    """
    Rate limiting dependency factory.
    """
    
    def __init__(self, requests_per_hour: int = 100):
        self.requests_per_hour = requests_per_hour
    
    async def __call__(
        self,
        current_user: User = Depends(get_current_active_user)
    ):
        """
        Check rate limit for current user.
        """
        user_id = str(current_user.id)
        endpoint = "general"  # Can be customized per endpoint
        
        # Check rate limit
        if not await cache_manager.check_rate_limit(user_id, endpoint, self.requests_per_hour):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later."
            )
        
        # Increment counter
        await cache_manager.increment_rate_limit(user_id, endpoint)
        
        return current_user


# Common rate limit instances
rate_limit_standard = RateLimitDependency(requests_per_hour=100)
rate_limit_strict = RateLimitDependency(requests_per_hour=50)
rate_limit_premium = RateLimitDependency(requests_per_hour=500)
