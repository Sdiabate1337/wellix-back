"""
Token repository for managing user tokens, quotas, and transactions.
Implements complete token business logic with atomic operations.
"""

from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, func, desc
from datetime import datetime, timedelta
import uuid

from app.db.models.token_system import (
    UserTokenBalance, 
    TokenTransaction, 
    SubscriptionPlan, 
    UsageAnalytics,
    PlanType, 
    TransactionType,
    TokenCosts
)
from app.db.models.user import User
from app.core.dependencies import get_db


class TokenRepository:
    """Repository for token management operations."""
    
    def __init__(self, db: Session):
        self.db = db
    
    async def get_user_balance(self, user_id: uuid.UUID) -> Optional[UserTokenBalance]:
        """Get current token balance for user."""
        balance = self.db.query(UserTokenBalance).filter(
            UserTokenBalance.user_id == user_id
        ).first()
        
        if balance and balance.needs_renewal:
            await self._reset_monthly_quota(balance)
            self.db.refresh(balance)
        
        return balance
    
    async def create_user_balance(
        self, 
        user_id: uuid.UUID, 
        plan_type: PlanType = PlanType.FREE
    ) -> UserTokenBalance:
        """Create initial token balance for new user."""
        plan = await self.get_plan_by_type(plan_type)
        if not plan:
            raise ValueError(f"Plan {plan_type} not found")
        
        # Calculate period dates
        now = datetime.utcnow()
        period_end = now + timedelta(days=30)
        
        balance = UserTokenBalance(
            user_id=user_id,
            plan_type=plan_type,
            plan_started_at=now,
            monthly_token_quota=plan.monthly_token_quota,
            tokens_used_this_month=0,
            bonus_tokens=plan.bonus_tokens_on_signup,
            current_period_start=now,
            current_period_end=period_end,
            last_reset_date=now
        )
        
        self.db.add(balance)
        
        # Create signup bonus transaction
        if plan.bonus_tokens_on_signup > 0:
            await self._create_transaction(
                user_id=user_id,
                amount=plan.bonus_tokens_on_signup,
                transaction_type=TransactionType.BONUS_CREDIT,
                feature_used="signup_bonus",
                plan_type_at_time=plan_type
            )
        
        self.db.commit()
        return balance
    
    async def can_afford_transaction(
        self, 
        user_id: uuid.UUID, 
        transaction_type: TransactionType
    ) -> bool:
        """Check if user can afford a specific transaction."""
        balance = await self.get_user_balance(user_id)
        if not balance:
            return False
        
        cost = TokenCosts.get_cost(transaction_type)
        return balance.can_afford(cost)
    
    async def reserve_tokens(
        self, 
        user_id: uuid.UUID, 
        amount: int
    ) -> bool:
        """Reserve tokens for upcoming transaction (prevents double-spending)."""
        balance = await self.get_user_balance(user_id)
        if not balance or not balance.can_afford(amount):
            return False
        
        # Create pending transaction (will be finalized later)
        reservation = TokenTransaction(
            user_id=user_id,
            amount=-amount,
            transaction_type="reserved",
            feature_used="reservation",
            plan_type_at_time=balance.plan_type,
            created_at=datetime.utcnow()
        )
        
        self.db.add(reservation)
        self.db.commit()
        return True
    
    async def consume_tokens(
        self,
        user_id: uuid.UUID,
        transaction_type: TransactionType,
        feature_used: str = None,
        analysis_id: uuid.UUID = None,
        chat_session_id: uuid.UUID = None,
        processing_time_ms: int = None,
        llm_model_used: str = None,
        metadata: Dict[str, Any] = None
    ) -> Optional[TokenTransaction]:
        """Consume tokens for a transaction atomically."""
        
        balance = await self.get_user_balance(user_id)
        if not balance:
            raise ValueError(f"No token balance found for user {user_id}")
        
        cost = TokenCosts.get_cost(transaction_type)
        
        if not balance.can_afford(cost):
            return None  # Insufficient tokens
        
        # Update balance
        if balance.bonus_tokens >= cost:
            balance.bonus_tokens -= cost
        else:
            remaining_cost = cost - balance.bonus_tokens
            balance.bonus_tokens = 0
            balance.tokens_used_this_month += remaining_cost
        
        balance.total_tokens_consumed += cost
        if transaction_type in [
            TransactionType.BASIC_ANALYSIS, 
            TransactionType.EXPERT_ANALYSIS,
            TransactionType.MULTI_CONDITION_ANALYSIS
        ]:
            balance.total_analyses_performed += 1
        
        # Create transaction record
        transaction = await self._create_transaction(
            user_id=user_id,
            amount=-cost,
            transaction_type=transaction_type,
            feature_used=feature_used,
            analysis_id=analysis_id,
            chat_session_id=chat_session_id,
            processing_time_ms=processing_time_ms,
            llm_model_used=llm_model_used,
            plan_type_at_time=balance.plan_type,
            metadata=metadata
        )
        
        self.db.commit()
        return transaction
    
    async def add_tokens(
        self,
        user_id: uuid.UUID,
        amount: int,
        transaction_type: TransactionType,
        reason: str = None
    ) -> TokenTransaction:
        """Add tokens to user balance (purchase, bonus, refund)."""
        
        balance = await self.get_user_balance(user_id)
        if not balance:
            raise ValueError(f"No token balance found for user {user_id}")
        
        # Add to bonus tokens (don't reset monthly)
        balance.bonus_tokens += amount
        
        transaction = await self._create_transaction(
            user_id=user_id,
            amount=amount,
            transaction_type=transaction_type,
            feature_used=reason,
            plan_type_at_time=balance.plan_type
        )
        
        self.db.commit()
        return transaction
    
    async def upgrade_plan(
        self,
        user_id: uuid.UUID,
        new_plan_type: PlanType
    ) -> UserTokenBalance:
        """Upgrade user to new plan."""
        
        balance = await self.get_user_balance(user_id)
        new_plan = await self.get_plan_by_type(new_plan_type)
        
        if not balance or not new_plan:
            raise ValueError("User balance or plan not found")
        
        # Update plan
        old_plan_type = balance.plan_type
        balance.plan_type = new_plan_type
        balance.monthly_token_quota = new_plan.monthly_token_quota
        balance.plan_started_at = datetime.utcnow()
        
        # Add upgrade bonus if applicable
        if new_plan.bonus_tokens_on_signup > 0:
            balance.bonus_tokens += new_plan.bonus_tokens_on_signup
            
            await self._create_transaction(
                user_id=user_id,
                amount=new_plan.bonus_tokens_on_signup,
                transaction_type=TransactionType.PLAN_PURCHASE,
                feature_used=f"upgrade_to_{new_plan_type}",
                plan_type_at_time=new_plan_type
            )
        
        self.db.commit()
        return balance
    
    async def get_transaction_history(
        self,
        user_id: uuid.UUID,
        limit: int = 50,
        offset: int = 0,
        transaction_type: TransactionType = None
    ) -> List[TokenTransaction]:
        """Get user transaction history."""
        
        query = self.db.query(TokenTransaction).filter(
            TokenTransaction.user_id == user_id
        )
        
        if transaction_type:
            query = query.filter(TokenTransaction.transaction_type == transaction_type)
        
        return query.order_by(desc(TokenTransaction.created_at)).offset(offset).limit(limit).all()
    
    async def get_usage_stats(
        self,
        user_id: uuid.UUID,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get user usage statistics."""
        
        start_date = datetime.utcnow() - timedelta(days=days)
        
        transactions = self.db.query(TokenTransaction).filter(
            and_(
                TokenTransaction.user_id == user_id,
                TokenTransaction.created_at >= start_date,
                TokenTransaction.amount < 0  # Only consumption
            )
        ).all()
        
        stats = {
            "total_tokens_consumed": sum(-t.amount for t in transactions),
            "total_transactions": len(transactions),
            "analyses_performed": len([t for t in transactions if "analysis" in t.transaction_type]),
            "chat_interactions": len([t for t in transactions if t.transaction_type == TransactionType.CHAT_INTERACTION]),
            "average_processing_time": 0,
            "feature_usage": {}
        }
        
        # Calculate average processing time
        processing_times = [t.processing_time_ms for t in transactions if t.processing_time_ms]
        if processing_times:
            stats["average_processing_time"] = sum(processing_times) / len(processing_times)
        
        # Feature usage breakdown
        for transaction in transactions:
            feature = transaction.transaction_type
            if feature not in stats["feature_usage"]:
                stats["feature_usage"][feature] = 0
            stats["feature_usage"][feature] += -transaction.amount
        
        return stats
    
    async def get_plan_by_type(self, plan_type: PlanType) -> Optional[SubscriptionPlan]:
        """Get subscription plan by type."""
        return self.db.query(SubscriptionPlan).filter(
            and_(
                SubscriptionPlan.plan_type == plan_type,
                SubscriptionPlan.is_active == True
            )
        ).first()
    
    async def get_all_active_plans(self) -> List[SubscriptionPlan]:
        """Get all active subscription plans."""
        return self.db.query(SubscriptionPlan).filter(
            and_(
                SubscriptionPlan.is_active == True,
                SubscriptionPlan.is_visible == True
            )
        ).order_by(SubscriptionPlan.price_eur).all()
    
    async def _reset_monthly_quota(self, balance: UserTokenBalance) -> None:
        """Reset monthly quota when period expires."""
        now = datetime.utcnow()
        
        # Reset usage
        balance.tokens_used_this_month = 0
        balance.current_period_start = now
        balance.current_period_end = now + timedelta(days=30)
        balance.last_reset_date = now
        
        # Create reset transaction for tracking
        await self._create_transaction(
            user_id=balance.user_id,
            amount=0,  # No tokens added, just tracking
            transaction_type="monthly_reset",
            feature_used="quota_reset",
            plan_type_at_time=balance.plan_type
        )
    
    async def _create_transaction(
        self,
        user_id: uuid.UUID,
        amount: int,
        transaction_type: str,
        feature_used: str = None,
        analysis_id: uuid.UUID = None,
        chat_session_id: uuid.UUID = None,
        processing_time_ms: int = None,
        llm_model_used: str = None,
        plan_type_at_time: str = None,
        metadata: Dict[str, Any] = None
    ) -> TokenTransaction:
        """Create a new token transaction."""
        
        transaction = TokenTransaction(
            user_id=user_id,
            amount=amount,
            transaction_type=transaction_type,
            feature_used=feature_used,
            analysis_id=analysis_id,
            chat_session_id=chat_session_id,
            processing_time_ms=processing_time_ms,
            llm_model_used=llm_model_used,
            plan_type_at_time=plan_type_at_time or PlanType.FREE,
            created_at=datetime.utcnow()
        )
        
        self.db.add(transaction)
        return transaction
    
    async def refund_transaction(
        self,
        transaction_id: uuid.UUID,
        reason: str = "error_refund"
    ) -> Optional[TokenTransaction]:
        """Refund a failed transaction."""
        
        original_transaction = self.db.query(TokenTransaction).filter(
            TokenTransaction.id == transaction_id
        ).first()
        
        if not original_transaction or original_transaction.refunded:
            return None
        
        # Mark original as refunded
        original_transaction.refunded = True
        
        # Create refund transaction
        refund_transaction = await self._create_transaction(
            user_id=original_transaction.user_id,
            amount=-original_transaction.amount,  # Opposite amount
            transaction_type=TransactionType.REFUND,
            feature_used=reason,
            plan_type_at_time=original_transaction.plan_type_at_time
        )
        
        # Update user balance
        balance = await self.get_user_balance(original_transaction.user_id)
        if balance:
            balance.bonus_tokens += -original_transaction.amount  # Add back tokens
        
        self.db.commit()
        return refund_transaction


# Dependency injection function
def get_token_repository(db: Session = get_db()) -> TokenRepository:
    """Get token repository instance."""
    return TokenRepository(db)