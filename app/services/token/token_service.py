"""
Token management service for Wellix SaaS.
Handles all token-related business logic including billing, upgrades, and analytics.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import uuid

from app.repositories.token_repository import TokenRepository
from app.db.models.token_system import (
    UserTokenBalance, 
    TokenTransaction, 
    SubscriptionPlan,
    PlanType, 
    TransactionType,
    DEFAULT_PLANS
)
from app.db.models.user import User
from sqlalchemy.orm import Session


class TokenService:
    """Business logic for token management."""
    
    def __init__(self, token_repo: TokenRepository):
        self.token_repo = token_repo
    
    async def initialize_user_tokens(self, user: User) -> UserTokenBalance:
        """Initialize token balance for new user."""
        existing_balance = await self.token_repo.get_user_balance(user.id)
        if existing_balance:
            return existing_balance
        
        # Create with free plan
        balance = await self.token_repo.create_user_balance(user.id, PlanType.FREE)
        
        # Send welcome notification
        await self._send_welcome_notification(user, balance)
        
        return balance
    
    async def get_user_dashboard_data(self, user_id: uuid.UUID) -> Dict[str, Any]:
        """Get comprehensive dashboard data for user."""
        balance = await self.token_repo.get_user_balance(user_id)
        if not balance:
            return {"error": "Token balance not found"}
        
        # Usage stats for last 30 days
        usage_stats = await self.token_repo.get_usage_stats(user_id, days=30)
        
        # Recent transactions
        recent_transactions = await self.token_repo.get_transaction_history(
            user_id, limit=10
        )
        
        # Plan information
        current_plan = await self.token_repo.get_plan_by_type(balance.plan_type)
        available_plans = await self.token_repo.get_all_active_plans()
        
        # Calculate recommendations
        recommendations = await self._generate_plan_recommendations(balance, usage_stats)
        
        return {
            "balance": {
                "tokens_remaining": balance.tokens_remaining,
                "monthly_quota": balance.monthly_token_quota,
                "bonus_tokens": balance.bonus_tokens,
                "usage_percentage": balance.usage_percentage,
                "plan_type": balance.plan_type,
                "period_end": balance.current_period_end,
                "auto_renewal": balance.auto_renewal
            },
            "usage_stats": usage_stats,
            "recent_transactions": [
                {
                    "id": str(t.id),
                    "amount": t.amount,
                    "type": t.transaction_type,
                    "feature": t.feature_used,
                    "created_at": t.created_at,
                    "processing_time": t.processing_time_ms
                }
                for t in recent_transactions
            ],
            "current_plan": {
                "name": current_plan.name,
                "description": current_plan.description,
                "price": current_plan.price_eur,
                "features": current_plan.allowed_features
            } if current_plan else None,
            "available_plans": [
                {
                    "type": plan.plan_type,
                    "name": plan.name,
                    "description": plan.description,
                    "monthly_quota": plan.monthly_token_quota,
                    "price": plan.price_eur,
                    "features": plan.allowed_features,
                    "is_current": plan.plan_type == balance.plan_type
                }
                for plan in available_plans
            ],
            "recommendations": recommendations
        }
    
    async def upgrade_user_plan(
        self, 
        user_id: uuid.UUID, 
        new_plan_type: PlanType,
        payment_method_id: str = None
    ) -> Dict[str, Any]:
        """Upgrade user to new plan with payment processing."""
        
        balance = await self.token_repo.get_user_balance(user_id)
        current_plan = await self.token_repo.get_plan_by_type(balance.plan_type)
        new_plan = await self.token_repo.get_plan_by_type(new_plan_type)
        
        if not all([balance, current_plan, new_plan]):
            return {"success": False, "error": "Plan or balance not found"}
        
        # Validate upgrade (can't downgrade mid-cycle)
        if new_plan.price_eur <= current_plan.price_eur:
            return {"success": False, "error": "Cannot downgrade during current period"}
        
        # Process payment (mock for now)
        payment_result = await self._process_payment(
            user_id=user_id,
            amount=new_plan.price_eur,
            payment_method_id=payment_method_id,
            description=f"Upgrade to {new_plan.name}"
        )
        
        if not payment_result["success"]:
            return {"success": False, "error": "Payment failed", "details": payment_result}
        
        # Upgrade plan
        updated_balance = await self.token_repo.upgrade_plan(user_id, new_plan_type)
        
        # Send confirmation
        await self._send_upgrade_confirmation(user_id, current_plan, new_plan)
        
        return {
            "success": True,
            "new_balance": {
                "plan_type": updated_balance.plan_type,
                "tokens_remaining": updated_balance.tokens_remaining,
                "monthly_quota": updated_balance.monthly_token_quota
            },
            "payment_id": payment_result.get("payment_id")
        }
    
    async def purchase_token_pack(
        self,
        user_id: uuid.UUID,
        token_amount: int,
        payment_method_id: str = None
    ) -> Dict[str, Any]:
        """Purchase additional token pack."""
        
        # Calculate price (e.g., €0.10 per token for packs)
        price_per_token = 0.10
        total_price = token_amount * price_per_token
        
        # Process payment
        payment_result = await self._process_payment(
            user_id=user_id,
            amount=total_price,
            payment_method_id=payment_method_id,
            description=f"Token pack: {token_amount} tokens"
        )
        
        if not payment_result["success"]:
            return {"success": False, "error": "Payment failed"}
        
        # Add tokens
        transaction = await self.token_repo.add_tokens(
            user_id=user_id,
            amount=token_amount,
            transaction_type=TransactionType.PLAN_PURCHASE,
            reason=f"token_pack_{token_amount}"
        )
        
        return {
            "success": True,
            "tokens_added": token_amount,
            "transaction_id": str(transaction.id),
            "payment_id": payment_result.get("payment_id")
        }
    
    async def analyze_usage_patterns(self, user_id: uuid.UUID) -> Dict[str, Any]:
        """Analyze user usage patterns for optimization suggestions."""
        
        # Get extended usage stats
        usage_7d = await self.token_repo.get_usage_stats(user_id, days=7)
        usage_30d = await self.token_repo.get_usage_stats(user_id, days=30)
        
        balance = await self.token_repo.get_user_balance(user_id)
        
        # Calculate patterns
        analysis = {
            "usage_trend": "stable",  # stable, increasing, decreasing
            "primary_features": [],
            "efficiency_score": 0,  # 0-100
            "recommendations": [],
            "projected_monthly_usage": 0,
            "plan_optimization": None
        }
        
        # Usage trend analysis
        if usage_7d["total_tokens_consumed"] > 0:
            weekly_projection = usage_7d["total_tokens_consumed"] * 4.3  # ~weekly to monthly
            
            if weekly_projection > usage_30d["total_tokens_consumed"] * 1.2:
                analysis["usage_trend"] = "increasing"
            elif weekly_projection < usage_30d["total_tokens_consumed"] * 0.8:
                analysis["usage_trend"] = "decreasing"
        
        # Primary features analysis
        feature_usage = usage_30d.get("feature_usage", {})
        sorted_features = sorted(feature_usage.items(), key=lambda x: x[1], reverse=True)
        analysis["primary_features"] = [feature for feature, _ in sorted_features[:3]]
        
        # Efficiency score (how well user utilizes their quota)
        if balance.monthly_token_quota > 0:
            utilization_rate = min(100, (usage_30d["total_tokens_consumed"] / balance.monthly_token_quota) * 100)
            analysis["efficiency_score"] = utilization_rate
        
        # Generate recommendations
        analysis["recommendations"] = await self._generate_usage_recommendations(
            balance, usage_30d, analysis
        )
        
        # Plan optimization
        analysis["plan_optimization"] = await self._suggest_plan_optimization(
            balance, usage_30d, analysis
        )
        
        return analysis
    
    async def handle_payment_webhook(self, webhook_data: Dict[str, Any]) -> bool:
        """Handle payment provider webhooks."""
        # Implementation depends on payment provider (Stripe, PayPal, etc.)
        
        event_type = webhook_data.get("type")
        
        if event_type == "payment.succeeded":
            return await self._handle_successful_payment(webhook_data)
        elif event_type == "payment.failed":
            return await self._handle_failed_payment(webhook_data)
        elif event_type == "subscription.cancelled":
            return await self._handle_subscription_cancellation(webhook_data)
        
        return False
    
    async def get_billing_history(self, user_id: uuid.UUID) -> List[Dict[str, Any]]:
        """Get user billing history."""
        
        # Get payment transactions
        payment_transactions = await self.token_repo.get_transaction_history(
            user_id=user_id,
            limit=100,
            transaction_type=TransactionType.PLAN_PURCHASE
        )
        
        billing_history = []
        for transaction in payment_transactions:
            billing_history.append({
                "id": str(transaction.id),
                "date": transaction.created_at,
                "description": transaction.feature_used,
                "amount": f"€{abs(transaction.amount) * 0.02:.2f}",  # Convert tokens to euros
                "status": "completed",
                "tokens_received": abs(transaction.amount)
            })
        
        return billing_history
    
    async def cancel_subscription(self, user_id: uuid.UUID) -> Dict[str, Any]:
        """Cancel user subscription (downgrade to free at end of period)."""
        
        balance = await self.token_repo.get_user_balance(user_id)
        if not balance or balance.plan_type == PlanType.FREE:
            return {"success": False, "error": "No active subscription to cancel"}
        
        # Set auto-renewal to false
        balance.auto_renewal = False
        
        # Schedule downgrade at period end
        # In real implementation, use a job scheduler
        await self._schedule_plan_downgrade(user_id, balance.current_period_end)
        
        return {
            "success": True,
            "message": "Subscription will be cancelled at the end of current period",
            "period_end": balance.current_period_end,
            "downgrade_to": PlanType.FREE
        }
    
    # Private helper methods
    
    async def _generate_plan_recommendations(
        self, 
        balance: UserTokenBalance, 
        usage_stats: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate personalized plan recommendations."""
        
        recommendations = []
        monthly_usage = usage_stats.get("total_tokens_consumed", 0)
        
        # If user is frequently running out of tokens
        if balance.usage_percentage > 80:
            recommendations.append({
                "type": "upgrade_needed",
                "message": "You're using most of your monthly quota. Consider upgrading.",
                "suggested_plan": PlanType.BASIC if balance.plan_type == PlanType.FREE else PlanType.PREMIUM,
                "priority": "high"
            })
        
        # If user has too many unused tokens
        elif balance.usage_percentage < 20:
            recommendations.append({
                "type": "optimization",
                "message": "You have many unused tokens. You might save with a lower plan.",
                "suggested_plan": PlanType.FREE if balance.plan_type == PlanType.BASIC else balance.plan_type,
                "priority": "low"
            })
        
        return recommendations
    
    async def _generate_usage_recommendations(
        self,
        balance: UserTokenBalance,
        usage_stats: Dict[str, Any],
        analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate usage optimization recommendations."""
        
        recommendations = []
        
        # Feature-specific recommendations
        feature_usage = usage_stats.get("feature_usage", {})
        
        if TransactionType.CHAT_INTERACTION in feature_usage:
            chat_usage = feature_usage[TransactionType.CHAT_INTERACTION]
            if chat_usage > 20:  # High chat usage
                recommendations.append(
                    "Consider batching your questions to reduce chat token consumption"
                )
        
        if TransactionType.IMAGE_OCR_PROCESSING in feature_usage:
            ocr_usage = feature_usage[TransactionType.IMAGE_OCR_PROCESSING]
            if ocr_usage > 10:
                recommendations.append(
                    "Try manual input for simple products to save OCR tokens"
                )
        
        # Plan-specific recommendations
        if balance.plan_type == PlanType.FREE and usage_stats["total_tokens_consumed"] > 15:
            recommendations.append(
                "Upgrade to Basic plan for better value and more features"
            )
        
        return recommendations
    
    async def _suggest_plan_optimization(
        self,
        balance: UserTokenBalance,
        usage_stats: Dict[str, Any],
        analysis: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Suggest optimal plan based on usage patterns."""
        
        monthly_usage = usage_stats.get("total_tokens_consumed", 0)
        projected_usage = monthly_usage * 1.2  # Add 20% buffer
        
        all_plans = await self.token_repo.get_all_active_plans()
        
        # Find most cost-effective plan
        optimal_plan = None
        for plan in all_plans:
            if plan.monthly_token_quota >= projected_usage:
                if not optimal_plan or plan.price_eur < optimal_plan.price_eur:
                    optimal_plan = plan
        
        if optimal_plan and optimal_plan.plan_type != balance.plan_type:
            return {
                "suggested_plan": optimal_plan.plan_type,
                "reason": f"Based on your usage of {monthly_usage} tokens/month",
                "savings": f"€{abs(optimal_plan.price_eur - balance.monthly_token_quota * 0.02):.2f}/month"
            }
        
        return None
    
    async def _process_payment(
        self,
        user_id: uuid.UUID,
        amount: float,
        payment_method_id: str = None,
        description: str = ""
    ) -> Dict[str, Any]:
        """Process payment through payment provider."""
        
        # Mock implementation - integrate with Stripe, PayPal, etc.
        import random
        
        # Simulate payment processing
        success = random.choice([True, True, True, False])  # 75% success rate
        
        if success:
            return {
                "success": True,
                "payment_id": f"pay_{uuid.uuid4().hex[:16]}",
                "amount": amount,
                "currency": "EUR"
            }
        else:
            return {
                "success": False,
                "error": "payment_declined",
                "message": "Payment was declined by your bank"
            }
    
    async def _send_welcome_notification(self, user: User, balance: UserTokenBalance):
        """Send welcome notification to new user."""
        # Implement email/push notification
        pass
    
    async def _send_upgrade_confirmation(self, user_id: uuid.UUID, old_plan, new_plan):
        """Send upgrade confirmation notification."""
        # Implement notification
        pass
    
    async def _schedule_plan_downgrade(self, user_id: uuid.UUID, downgrade_date: datetime):
        """Schedule automatic plan downgrade."""
        # Implement with job scheduler (Celery, etc.)
        pass
    
    async def _handle_successful_payment(self, webhook_data: Dict[str, Any]) -> bool:
        """Handle successful payment webhook."""
        # Extract user_id, amount, etc. from webhook
        # Add tokens to user account
        return True
    
    async def _handle_failed_payment(self, webhook_data: Dict[str, Any]) -> bool:
        """Handle failed payment webhook."""
        # Notify user, potentially suspend account
        return True
    
    async def _handle_subscription_cancellation(self, webhook_data: Dict[str, Any]) -> bool:
        """Handle subscription cancellation webhook."""
        # Downgrade user to free plan
        return True


# Factory function for dependency injection
def create_token_service(token_repo: TokenRepository) -> TokenService:
    """Create token service instance."""
    return TokenService(token_repo)