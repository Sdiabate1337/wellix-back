"""
Token system models for Wellix SaaS monetization.
Implements complete token-based billing with quotas, transactions, and plans.
"""

from typing import Optional, Dict, Any, List
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, Float, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import relationship, foreign
from sqlalchemy.sql import func
from datetime import datetime, timedelta
import uuid
from enum import Enum

from app.db.database import Base


class PlanType(str, Enum):
    """Subscription plan types."""
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class TransactionType(str, Enum):
    """Token transaction types."""
    # Consumption
    BASIC_ANALYSIS = "basic_analysis"
    EXPERT_ANALYSIS = "expert_analysis"
    MULTI_CONDITION_ANALYSIS = "multi_condition_analysis"
    ALTERNATIVES_GENERATION = "alternatives_generation"
    CHAT_INTERACTION = "chat_interaction"
    IMAGE_OCR_PROCESSING = "image_ocr_processing"
    DETAILED_RECOMMENDATIONS = "detailed_recommendations"
    
    # Premium features
    MEAL_PLANNING = "meal_planning"
    PROGRESS_TRACKING = "progress_tracking"
    
    # Credits
    PLAN_PURCHASE = "plan_purchase"
    BONUS_CREDIT = "bonus_credit"
    REFUND = "refund"


class TokenCosts:
    """Token costs configuration."""
    COSTS = {
        TransactionType.BASIC_ANALYSIS: 1,
        TransactionType.EXPERT_ANALYSIS: 5,
        TransactionType.MULTI_CONDITION_ANALYSIS: 7,
        TransactionType.ALTERNATIVES_GENERATION: 3,
        TransactionType.CHAT_INTERACTION: 1,
        TransactionType.IMAGE_OCR_PROCESSING: 2,
        TransactionType.DETAILED_RECOMMENDATIONS: 2,
        TransactionType.MEAL_PLANNING: 10,
        TransactionType.PROGRESS_TRACKING: 5,
    }
    
    @classmethod
    def get_cost(cls, transaction_type: TransactionType) -> int:
        """Get token cost for a transaction type."""
        return cls.COSTS.get(transaction_type, 1)


class SubscriptionPlan(Base):
    """Subscription plans configuration."""
    
    __tablename__ = "subscription_plans"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    plan_type = Column(String(20), unique=True, nullable=False, index=True)
    
    # Plan details
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    
    # Token allocation
    monthly_token_quota = Column(Integer, nullable=False)
    bonus_tokens_on_signup = Column(Integer, default=0)
    
    # Pricing
    price_eur = Column(Float, nullable=False)
    price_usd = Column(Float, nullable=False)
    
    # Features
    allowed_features = Column(ARRAY(String), default=list)
    max_daily_analyses = Column(Integer, nullable=True)
    priority_support = Column(Boolean, default=False)
    api_access = Column(Boolean, default=False)
    
    # Status
    is_active = Column(Boolean, default=True)
    is_visible = Column(Boolean, default=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<SubscriptionPlan(type={self.plan_type}, quota={self.monthly_token_quota})>"


class UserTokenBalance(Base):
    """User token balance and quota management."""
    
    __tablename__ = "user_token_balances"
    
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), primary_key=True, index=True)
    
    # Current plan
    plan_type = Column(String(20), nullable=False, default=PlanType.FREE)
    plan_started_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    
    # Token quotas and usage
    monthly_token_quota = Column(Integer, nullable=False, default=20)
    tokens_used_this_month = Column(Integer, default=0, nullable=False)
    bonus_tokens = Column(Integer, default=0, nullable=False)  # Tokens qui ne se reset pas
    
    # Current period tracking
    current_period_start = Column(DateTime(timezone=True), nullable=False, default=func.now())
    current_period_end = Column(DateTime(timezone=True), nullable=False)
    last_reset_date = Column(DateTime(timezone=True), nullable=False, default=func.now())
    
    # Usage analytics
    total_tokens_consumed = Column(Integer, default=0, nullable=False)
    total_analyses_performed = Column(Integer, default=0, nullable=False)
    
    # Billing
    auto_renewal = Column(Boolean, default=True)
    payment_method_id = Column(String(100), nullable=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User")
    transactions = relationship(
        "TokenTransaction", 
        back_populates="user_balance", 
        cascade="all, delete-orphan",
        primaryjoin="UserTokenBalance.user_id == foreign(TokenTransaction.user_id)"
    )
    
    @property
    def tokens_remaining(self) -> int:
        """Calculate remaining tokens for current period."""
        monthly_remaining = max(0, self.monthly_token_quota - self.tokens_used_this_month)
        return monthly_remaining + self.bonus_tokens
    
    @property
    def tokens_available(self) -> int:
        """Alias for tokens_remaining."""
        return self.tokens_remaining
    
    @property
    def usage_percentage(self) -> float:
        """Calculate usage percentage for current month."""
        if self.monthly_token_quota == 0:
            return 0.0
        return min(100.0, (self.tokens_used_this_month / self.monthly_token_quota) * 100)
    
    @property
    def needs_renewal(self) -> bool:
        """Check if plan needs renewal."""
        return datetime.utcnow() >= self.current_period_end
    
    def can_afford(self, cost: int) -> bool:
        """Check if user can afford a transaction."""
        return self.tokens_remaining >= cost
    
    def __repr__(self):
        return f"<UserTokenBalance(user_id={self.user_id}, plan={self.plan_type}, remaining={self.tokens_remaining})>"


class TokenTransaction(Base):
    """Individual token transactions for detailed tracking."""
    
    __tablename__ = "token_transactions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    
    # Transaction details
    amount = Column(Integer, nullable=False)  # Négatif = dépense, Positif = crédit
    transaction_type = Column(String(50), nullable=False, index=True)
    feature_used = Column(String(100), nullable=True)
    
    # Context and metadata
    analysis_id = Column(UUID(as_uuid=True), ForeignKey("food_analyses.id"), nullable=True, index=True)
    chat_session_id = Column(UUID(as_uuid=True), nullable=True)
    
    # Processing details
    processing_time_ms = Column(Integer, nullable=True)
    llm_model_used = Column(String(50), nullable=True)
    tokens_estimated = Column(Integer, nullable=True)
    tokens_actual = Column(Integer, nullable=True)
    
    # Quality metrics
    user_feedback = Column(Integer, nullable=True)  # 1-5 rating
    error_occurred = Column(Boolean, default=False)
    refunded = Column(Boolean, default=False)
    
    # Billing context
    plan_type_at_time = Column(String(20), nullable=False)
    rate_limited = Column(Boolean, default=False)
    
    # Additional metadata
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    # Relationships
    user_balance = relationship(
        "UserTokenBalance", 
        back_populates="transactions",
        primaryjoin="foreign(TokenTransaction.user_id) == UserTokenBalance.user_id"
    )
    analysis = relationship("FoodAnalysis", foreign_keys=[analysis_id])
    
    @property
    def is_consumption(self) -> bool:
        """Check if this is a token consumption transaction."""
        return self.amount < 0
    
    @property
    def is_credit(self) -> bool:
        """Check if this is a token credit transaction."""
        return self.amount > 0
    
    def __repr__(self):
        return f"<TokenTransaction(id={self.id}, amount={self.amount}, type={self.transaction_type})>"


class UsageAnalytics(Base):
    """Aggregated usage analytics for business insights."""
    
    __tablename__ = "usage_analytics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    
    # Time period
    date = Column(DateTime(timezone=True), nullable=False, index=True)
    period_type = Column(String(20), nullable=False)  # daily, weekly, monthly
    
    # User segmentation
    plan_type = Column(String(20), nullable=True, index=True)
    user_segment = Column(String(50), nullable=True)  # new, returning, churned
    
    # Usage metrics
    total_users = Column(Integer, default=0)
    active_users = Column(Integer, default=0)
    total_analyses = Column(Integer, default=0)
    total_tokens_consumed = Column(Integer, default=0)
    
    # Revenue metrics
    total_revenue = Column(Float, default=0.0)
    new_subscriptions = Column(Integer, default=0)
    cancelled_subscriptions = Column(Integer, default=0)
    
    # Feature adoption
    feature_usage = Column(JSONB, default=dict)  # feature_name -> usage_count
    
    # Quality metrics
    average_rating = Column(Float, nullable=True)
    error_rate = Column(Float, nullable=True)
    refund_rate = Column(Float, nullable=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<UsageAnalytics(date={self.date}, plan={self.plan_type}, users={self.total_users})>"


# Default subscription plans data
DEFAULT_PLANS = [
    {
        "plan_type": PlanType.FREE,
        "name": "Free Plan",
        "description": "Perfect for trying Wellix",
        "monthly_token_quota": 20,
        "bonus_tokens_on_signup": 10,
        "price_eur": 0.0,
        "price_usd": 0.0,
        "allowed_features": [
            TransactionType.BASIC_ANALYSIS,
            TransactionType.CHAT_INTERACTION,
            TransactionType.IMAGE_OCR_PROCESSING
        ],
        "max_daily_analyses": 5,
        "priority_support": False,
        "api_access": False
    },
    {
        "plan_type": PlanType.BASIC,
        "name": "Basic Plan", 
        "description": "Great for regular users",
        "monthly_token_quota": 100,
        "bonus_tokens_on_signup": 20,
        "price_eur": 9.99,
        "price_usd": 10.99,
        "allowed_features": [
            TransactionType.BASIC_ANALYSIS,
            TransactionType.EXPERT_ANALYSIS,
            TransactionType.ALTERNATIVES_GENERATION,
            TransactionType.CHAT_INTERACTION,
            TransactionType.IMAGE_OCR_PROCESSING,
            TransactionType.DETAILED_RECOMMENDATIONS
        ],
        "max_daily_analyses": 25,
        "priority_support": False,
        "api_access": False
    },
    {
        "plan_type": PlanType.PREMIUM,
        "name": "Premium Plan",
        "description": "Best for health-conscious users",
        "monthly_token_quota": 500,
        "bonus_tokens_on_signup": 50,
        "price_eur": 29.99,
        "price_usd": 32.99,
        "allowed_features": [
            TransactionType.BASIC_ANALYSIS,
            TransactionType.EXPERT_ANALYSIS,
            TransactionType.MULTI_CONDITION_ANALYSIS,
            TransactionType.ALTERNATIVES_GENERATION,
            TransactionType.CHAT_INTERACTION,
            TransactionType.IMAGE_OCR_PROCESSING,
            TransactionType.DETAILED_RECOMMENDATIONS,
            TransactionType.MEAL_PLANNING,
            TransactionType.PROGRESS_TRACKING
        ],
        "max_daily_analyses": 100,
        "priority_support": True,
        "api_access": True
    },
    {
        "plan_type": PlanType.ENTERPRISE,
        "name": "Enterprise Plan",
        "description": "For businesses and healthcare providers",
        "monthly_token_quota": 2000,
        "bonus_tokens_on_signup": 200,
        "price_eur": 99.99,
        "price_usd": 109.99,
        "allowed_features": [
            # All features available
            TransactionType.BASIC_ANALYSIS,
            TransactionType.EXPERT_ANALYSIS,
            TransactionType.MULTI_CONDITION_ANALYSIS,
            TransactionType.ALTERNATIVES_GENERATION,
            TransactionType.CHAT_INTERACTION,
            TransactionType.IMAGE_OCR_PROCESSING,
            TransactionType.DETAILED_RECOMMENDATIONS,
            TransactionType.MEAL_PLANNING,
            TransactionType.PROGRESS_TRACKING
        ],
        "max_daily_analyses": None,  # Unlimited
        "priority_support": True,
        "api_access": True
    }
]