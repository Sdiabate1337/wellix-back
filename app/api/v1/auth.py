"""
Authentication API endpoints for user registration, login, and token management.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import structlog
from datetime import timedelta

from app.core.dependencies import get_current_active_user
from app.core.security import security_manager, verify_password, get_password_hash
from app.db.database import get_async_session
from app.db.models.user import User
from app.cache.cache_manager import cache_manager
from pydantic import BaseModel, EmailStr, validator

logger = structlog.get_logger(__name__)

router = APIRouter()
security = HTTPBearer()


class UserRegistration(BaseModel):
    email: EmailStr
    password: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    
    @validator("password")
    def validate_password(cls, v):
        if not security_manager.validate_password_strength(v):
            raise ValueError("Password must be at least 8 characters with uppercase, lowercase, digit, and special character")
        return v


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: dict


class UserProfile(BaseModel):
    id: str
    email: str
    first_name: Optional[str]
    last_name: Optional[str]
    is_premium: bool
    created_at: str


@router.post("/register", response_model=TokenResponse)
async def register_user(
    user_data: UserRegistration,
    db: AsyncSession = Depends(get_async_session)
):
    """Register a new user account."""
    try:
        # Check if user already exists
        result = await db.execute(select(User).where(User.email == user_data.email))
        existing_user = result.scalar_one_or_none()
        
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create new user
        hashed_password = get_password_hash(user_data.password)
        
        new_user = User(
            email=user_data.email,
            hashed_password=hashed_password,
            first_name=user_data.first_name,
            last_name=user_data.last_name,
            is_active=True,
            is_verified=False  # Email verification would be implemented separately
        )
        
        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)
        
        # Generate tokens
        token_data = {"sub": str(new_user.id), "email": new_user.email}
        access_token = security_manager.create_access_token(token_data)
        refresh_token = security_manager.create_refresh_token(token_data)
        
        # Cache user session
        await cache_manager.set_user_health_context(
            str(new_user.id),
            {"user_id": str(new_user.id), "email": new_user.email}
        )
        
        logger.info(f"New user registered: {new_user.email}")
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=security_manager.access_token_expire_minutes * 60,
            user={
                "id": str(new_user.id),
                "email": new_user.email,
                "first_name": new_user.first_name,
                "last_name": new_user.last_name,
                "is_premium": new_user.is_premium
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/login", response_model=TokenResponse)
async def login_user(
    login_data: UserLogin,
    db: AsyncSession = Depends(get_async_session)
):
    """Authenticate user and return access tokens."""
    try:
        # Get user by email
        result = await db.execute(select(User).where(User.email == login_data.email))
        user = result.scalar_one_or_none()
        
        if not user or not verify_password(login_data.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Account is deactivated"
            )
        
        # Update last login
        from datetime import datetime
        user.last_login = datetime.utcnow()
        await db.commit()
        
        # Generate tokens
        token_data = {"sub": str(user.id), "email": user.email}
        access_token = security_manager.create_access_token(token_data)
        refresh_token = security_manager.create_refresh_token(token_data)
        
        logger.info(f"User logged in: {user.email}")
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=security_manager.access_token_expire_minutes * 60,
            user={
                "id": str(user.id),
                "email": user.email,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "is_premium": user.is_premium
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post("/refresh")
async def refresh_access_token(
    refresh_token: str,
    db: AsyncSession = Depends(get_async_session)
):
    """Refresh access token using refresh token."""
    try:
        # Verify refresh token
        payload = security_manager.verify_token(refresh_token, "refresh")
        user_id = payload.get("sub")
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        # Get user
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )
        
        # Generate new access token
        token_data = {"sub": str(user.id), "email": user.email}
        new_access_token = security_manager.create_access_token(token_data)
        
        return JSONResponse(content={
            "access_token": new_access_token,
            "token_type": "bearer",
            "expires_in": security_manager.access_token_expire_minutes * 60
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token refresh failed"
        )


@router.get("/profile", response_model=UserProfile)
async def get_user_profile(
    current_user: User = Depends(get_current_active_user)
):
    """Get current user's profile information."""
    return UserProfile(
        id=str(current_user.id),
        email=current_user.email,
        first_name=current_user.first_name,
        last_name=current_user.last_name,
        is_premium=current_user.is_premium,
        created_at=current_user.created_at.isoformat()
    )


@router.post("/logout")
async def logout_user(
    current_user: User = Depends(get_current_active_user)
):
    """Logout user and invalidate session."""
    try:
        # Invalidate user cache
        await cache_manager.invalidate_user_cache(str(current_user.id))
        
        logger.info(f"User logged out: {current_user.email}")
        
        return JSONResponse(content={"message": "Successfully logged out"})
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )
