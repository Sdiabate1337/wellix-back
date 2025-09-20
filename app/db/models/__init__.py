"""
Database models for Wellix SaaS.
"""

from .user import User
from .health_profile import HealthProfile, UserHealthContext
from .food_analysis import FoodAnalysis, ProductRecommendation, OCRResult
from .chat import ChatSession, ChatMessage
from .token_system import (
    UserTokenBalance,
    TokenTransaction, 
    SubscriptionPlan,
    UsageAnalytics,
    PlanType,
    TransactionType,
    TokenCosts,
    DEFAULT_PLANS
)

__all__ = [
    # User models
    "User",
    
    # Health models
    "HealthProfile",
    "UserHealthContext",
    
    # Food analysis models
    "FoodAnalysis",
    "ProductRecommendation", 
    "OCRResult",
    
    # Chat models
    "ChatSession",
    "ChatMessage",
    
    # Token system models
    "UserTokenBalance",
    "TokenTransaction",
    "SubscriptionPlan", 
    "UsageAnalytics",
    "PlanType",
    "TransactionType",
    "TokenCosts",
    "DEFAULT_PLANS"
]