"""
Token validation middleware for protecting API endpoints.
Implements token-based access control with rate limiting and quota enforcement.
"""

from typing import Optional, Callable, Any
from functools import wraps
from fastapi import HTTPException, Request, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import time
import uuid

from app.db.models.token_system import TransactionType, PlanType
from app.repositories.token_repository import TokenRepository, get_token_repository
from app.core.dependencies import get_db, get_current_user
from app.db.models.user import User


security = HTTPBearer(auto_error=False)


class TokenValidationError(Exception):
    """Custom exception for token validation errors."""
    pass


class InsufficientTokensError(TokenValidationError):
    """Raised when user doesn't have enough tokens."""
    pass


class PlanLimitationError(TokenValidationError):
    """Raised when user's plan doesn't allow the feature."""
    pass


class RateLimitError(TokenValidationError):
    """Raised when user exceeds rate limits."""
    pass


class TokenValidator:
    """Token validation and consumption logic."""
    
    def __init__(self, token_repo: TokenRepository):
        self.token_repo = token_repo
    
    async def validate_and_reserve(
        self,
        user_id: uuid.UUID,
        transaction_type: TransactionType,
        feature_name: str = None
    ) -> bool:
        """Validate tokens and reserve them for transaction."""
        
        # Check if user can afford the transaction
        can_afford = await self.token_repo.can_afford_transaction(user_id, transaction_type)
        if not can_afford:
            raise InsufficientTokensError(f"Insufficient tokens for {transaction_type}")
        
        # Check plan limitations
        balance = await self.token_repo.get_user_balance(user_id)
        if not balance:
            raise TokenValidationError("User token balance not found")
        
        await self._validate_plan_permissions(balance, transaction_type)
        await self._validate_rate_limits(balance, transaction_type)
        
        # Reserve tokens
        from app.db.models.token_system import TokenCosts
        cost = TokenCosts.get_cost(transaction_type)
        success = await self.token_repo.reserve_tokens(user_id, cost)
        
        if not success:
            raise InsufficientTokensError("Failed to reserve tokens")
        
        return True
    
    async def consume_tokens(
        self,
        user_id: uuid.UUID,
        transaction_type: TransactionType,
        feature_used: str = None,
        analysis_id: uuid.UUID = None,
        processing_time_ms: int = None,
        llm_model_used: str = None,
        metadata: dict = None
    ) -> bool:
        """Consume tokens after successful operation."""
        
        transaction = await self.token_repo.consume_tokens(
            user_id=user_id,
            transaction_type=transaction_type,
            feature_used=feature_used,
            analysis_id=analysis_id,
            processing_time_ms=processing_time_ms,
            llm_model_used=llm_model_used,
            metadata=metadata
        )
        
        return transaction is not None
    
    async def _validate_plan_permissions(self, balance, transaction_type: TransactionType):
        """Validate if user's plan allows the feature."""
        plan = await self.token_repo.get_plan_by_type(balance.plan_type)
        if not plan:
            raise TokenValidationError(f"Plan {balance.plan_type} not found")
        
        if transaction_type not in plan.allowed_features:
            raise PlanLimitationError(
                f"Feature {transaction_type} not available in {balance.plan_type} plan"
            )
    
    async def _validate_rate_limits(self, balance, transaction_type: TransactionType):
        """Validate daily and hourly rate limits."""
        plan = await self.token_repo.get_plan_by_type(balance.plan_type)
        
        # Check daily analysis limit
        if (plan.max_daily_analyses and 
            transaction_type in [TransactionType.BASIC_ANALYSIS, TransactionType.EXPERT_ANALYSIS]):
            
            today_analyses = await self._get_today_analysis_count(balance.user_id)
            if today_analyses >= plan.max_daily_analyses:
                raise RateLimitError(f"Daily analysis limit ({plan.max_daily_analyses}) exceeded")
        
        # Additional hourly limits for free users
        if balance.plan_type == PlanType.FREE:
            hourly_limit = 5
            recent_count = await self._get_recent_transaction_count(balance.user_id, hours=1)
            if recent_count >= hourly_limit:
                raise RateLimitError("Hourly rate limit exceeded for free plan")
    
    async def _get_today_analysis_count(self, user_id: uuid.UUID) -> int:
        """Get number of analyses performed today."""
        from datetime import datetime, timedelta
        from app.db.models.token_system import TokenTransaction
        from sqlalchemy import and_
        
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        
        count = self.token_repo.db.query(TokenTransaction).filter(
            and_(
                TokenTransaction.user_id == user_id,
                TokenTransaction.created_at >= today_start,
                TokenTransaction.transaction_type.in_([
                    TransactionType.BASIC_ANALYSIS,
                    TransactionType.EXPERT_ANALYSIS,
                    TransactionType.MULTI_CONDITION_ANALYSIS
                ])
            )
        ).count()
        
        return count
    
    async def _get_recent_transaction_count(self, user_id: uuid.UUID, hours: int = 1) -> int:
        """Get number of transactions in recent hours."""
        from datetime import datetime, timedelta
        from app.db.models.token_system import TokenTransaction
        from sqlalchemy import and_
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        count = self.token_repo.db.query(TokenTransaction).filter(
            and_(
                TokenTransaction.user_id == user_id,
                TokenTransaction.created_at >= cutoff_time,
                TokenTransaction.amount < 0  # Only consumption
            )
        ).count()
        
        return count


# Decorator factory for token-protected endpoints
def token_required(
    transaction_type: TransactionType,
    feature_name: str = None,
    auto_consume: bool = True
):
    """
    Decorator to protect endpoints with token validation.
    
    Args:
        transaction_type: Type of transaction that will consume tokens
        feature_name: Optional feature name for tracking
        auto_consume: Whether to automatically consume tokens on success
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract dependencies
            current_user: User = None
            token_repo: TokenRepository = None
            
            # Find dependencies in kwargs
            for key, value in kwargs.items():
                if isinstance(value, User):
                    current_user = value
                elif isinstance(value, TokenRepository):
                    token_repo = value
            
            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            if not token_repo:
                # Get from dependency injection if not provided
                token_repo = kwargs.get('token_repo') or get_token_repository()
            
            # Validate and reserve tokens
            validator = TokenValidator(token_repo)
            
            try:
                await validator.validate_and_reserve(
                    user_id=current_user.id,
                    transaction_type=transaction_type,
                    feature_name=feature_name
                )
            except InsufficientTokensError:
                raise HTTPException(
                    status_code=status.HTTP_402_PAYMENT_REQUIRED,
                    detail={
                        "error": "insufficient_tokens",
                        "message": f"Not enough tokens for {transaction_type}",
                        "required_tokens": f"{transaction_type}",
                        "upgrade_suggestion": "Consider upgrading your plan"
                    }
                )
            except PlanLimitationError as e:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail={
                        "error": "plan_limitation", 
                        "message": str(e),
                        "upgrade_required": True
                    }
                )
            except RateLimitError as e:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail={
                        "error": "rate_limit_exceeded",
                        "message": str(e),
                        "retry_after": 3600  # 1 hour
                    }
                )
            
            # Execute the actual function
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                # Auto-consume tokens on success
                if auto_consume:
                    processing_time_ms = int((time.time() - start_time) * 1000)
                    
                    await validator.consume_tokens(
                        user_id=current_user.id,
                        transaction_type=transaction_type,
                        feature_used=feature_name or func.__name__,
                        processing_time_ms=processing_time_ms,
                        metadata={"endpoint": func.__name__}
                    )
                
                return result
                
            except Exception as e:
                # Refund tokens on error
                # TODO: Implement refund logic for reserved tokens
                raise e
        
        return wrapper
    return decorator


# Alternative dependency-based approach
async def validate_tokens_dependency(
    transaction_type: TransactionType,
    current_user: User = Depends(get_current_user),
    token_repo: TokenRepository = Depends(get_token_repository)
):
    """Dependency for validating tokens in endpoints."""
    
    validator = TokenValidator(token_repo)
    
    try:
        await validator.validate_and_reserve(
            user_id=current_user.id,
            transaction_type=transaction_type
        )
        return {"validated": True, "user": current_user, "token_repo": token_repo}
        
    except InsufficientTokensError:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Insufficient tokens"
        )
    except (PlanLimitationError, RateLimitError) as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )


# Utility functions for manual token management
async def consume_tokens_manual(
    user_id: uuid.UUID,
    transaction_type: TransactionType,
    token_repo: TokenRepository,
    **kwargs
) -> bool:
    """Manually consume tokens (for complex operations)."""
    validator = TokenValidator(token_repo)
    return await validator.consume_tokens(
        user_id=user_id,
        transaction_type=transaction_type,
        **kwargs
    )


async def check_user_tokens(
    user_id: uuid.UUID,
    transaction_type: TransactionType,
    token_repo: TokenRepository
) -> dict:
    """Check user token status without consuming."""
    balance = await token_repo.get_user_balance(user_id)
    if not balance:
        return {"has_tokens": False, "balance": None}
    
    from app.db.models.token_system import TokenCosts
    cost = TokenCosts.get_cost(transaction_type)
    
    return {
        "has_tokens": balance.can_afford(cost),
        "tokens_remaining": balance.tokens_remaining,
        "monthly_quota": balance.monthly_token_quota,
        "plan_type": balance.plan_type,
        "cost_required": cost
    }