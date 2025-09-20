"""
Token management API endpoints.
Provides REST API for token operations, billing, and plan management.
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from pydantic import BaseModel
import uuid

from app.core.dependencies import get_db, get_current_user
from app.db.models.user import User
from app.db.models.token_system import PlanType, TransactionType
from app.repositories.token_repository import TokenRepository, get_token_repository
from app.services.token.token_service import TokenService, create_token_service
from app.middleware.token_validation import check_user_tokens


router = APIRouter(prefix="/tokens", tags=["tokens"])


# Pydantic models for requests/responses

class TokenBalanceResponse(BaseModel):
    tokens_remaining: int
    monthly_quota: int
    bonus_tokens: int
    usage_percentage: float
    plan_type: str
    period_end: str
    auto_renewal: bool


class PlanUpgradeRequest(BaseModel):
    new_plan_type: PlanType
    payment_method_id: Optional[str] = None


class TokenPackPurchaseRequest(BaseModel):
    token_amount: int
    payment_method_id: Optional[str] = None


class UsageStatsResponse(BaseModel):
    total_tokens_consumed: int
    total_transactions: int
    analyses_performed: int
    chat_interactions: int
    average_processing_time: float
    feature_usage: Dict[str, int]


class PlanRecommendationResponse(BaseModel):
    type: str
    message: str
    suggested_plan: Optional[str]
    priority: str


class DashboardResponse(BaseModel):
    balance: TokenBalanceResponse
    usage_stats: UsageStatsResponse
    recent_transactions: List[Dict[str, Any]]
    current_plan: Optional[Dict[str, Any]]
    available_plans: List[Dict[str, Any]]
    recommendations: List[PlanRecommendationResponse]


# API Endpoints

@router.get("/balance", response_model=TokenBalanceResponse)
async def get_token_balance(
    current_user: User = Depends(get_current_user),
    token_repo: TokenRepository = Depends(get_token_repository)
):
    """Get current user token balance and quota information."""
    
    balance = await token_repo.get_user_balance(current_user.id)
    if not balance:
        # Initialize balance for new user
        token_service = create_token_service(token_repo)
        balance = await token_service.initialize_user_tokens(current_user)
    
    return TokenBalanceResponse(
        tokens_remaining=balance.tokens_remaining,
        monthly_quota=balance.monthly_token_quota,
        bonus_tokens=balance.bonus_tokens,
        usage_percentage=balance.usage_percentage,
        plan_type=balance.plan_type,
        period_end=balance.current_period_end.isoformat(),
        auto_renewal=balance.auto_renewal
    )


@router.get("/dashboard", response_model=DashboardResponse)
async def get_dashboard_data(
    current_user: User = Depends(get_current_user),
    token_repo: TokenRepository = Depends(get_token_repository)
):
    """Get comprehensive dashboard data for user."""
    
    token_service = create_token_service(token_repo)
    dashboard_data = await token_service.get_user_dashboard_data(current_user.id)
    
    if "error" in dashboard_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=dashboard_data["error"]
        )
    
    return DashboardResponse(**dashboard_data)


@router.get("/plans")
async def get_available_plans(
    token_repo: TokenRepository = Depends(get_token_repository)
):
    """Get all available subscription plans."""
    
    plans = await token_repo.get_all_active_plans()
    
    return {
        "plans": [
            {
                "type": plan.plan_type,
                "name": plan.name,
                "description": plan.description,
                "monthly_quota": plan.monthly_token_quota,
                "price_eur": plan.price_eur,
                "price_usd": plan.price_usd,
                "features": plan.allowed_features,
                "max_daily_analyses": plan.max_daily_analyses,
                "priority_support": plan.priority_support,
                "api_access": plan.api_access
            }
            for plan in plans
        ]
    }


@router.post("/upgrade")
async def upgrade_plan(
    upgrade_request: PlanUpgradeRequest,
    current_user: User = Depends(get_current_user),
    token_repo: TokenRepository = Depends(get_token_repository)
):
    """Upgrade user to a new subscription plan."""
    
    token_service = create_token_service(token_repo)
    
    result = await token_service.upgrade_user_plan(
        user_id=current_user.id,
        new_plan_type=upgrade_request.new_plan_type,
        payment_method_id=upgrade_request.payment_method_id
    )
    
    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("error", "Upgrade failed")
        )
    
    return {
        "message": "Plan upgraded successfully",
        "new_balance": result["new_balance"],
        "payment_id": result.get("payment_id")
    }


@router.post("/purchase-pack")
async def purchase_token_pack(
    purchase_request: TokenPackPurchaseRequest,
    current_user: User = Depends(get_current_user),
    token_repo: TokenRepository = Depends(get_token_repository)
):
    """Purchase additional token pack."""
    
    if purchase_request.token_amount < 10 or purchase_request.token_amount > 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Token amount must be between 10 and 1000"
        )
    
    token_service = create_token_service(token_repo)
    
    result = await token_service.purchase_token_pack(
        user_id=current_user.id,
        token_amount=purchase_request.token_amount,
        payment_method_id=purchase_request.payment_method_id
    )
    
    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("error", "Purchase failed")
        )
    
    return {
        "message": "Token pack purchased successfully",
        "tokens_added": result["tokens_added"],
        "transaction_id": result["transaction_id"]
    }


@router.get("/history")
async def get_transaction_history(
    limit: int = 50,
    offset: int = 0,
    transaction_type: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    token_repo: TokenRepository = Depends(get_token_repository)
):
    """Get user transaction history."""
    
    if limit > 100:
        limit = 100
    
    filter_type = None
    if transaction_type:
        try:
            filter_type = TransactionType(transaction_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid transaction type: {transaction_type}"
            )
    
    transactions = await token_repo.get_transaction_history(
        user_id=current_user.id,
        limit=limit,
        offset=offset,
        transaction_type=filter_type
    )
    
    return {
        "transactions": [
            {
                "id": str(t.id),
                "amount": t.amount,
                "type": t.transaction_type,
                "feature": t.feature_used,
                "created_at": t.created_at.isoformat(),
                "processing_time_ms": t.processing_time_ms,
                "analysis_id": str(t.analysis_id) if t.analysis_id else None,
                "refunded": t.refunded,
                "error_occurred": t.error_occurred
            }
            for t in transactions
        ],
        "total": len(transactions),
        "limit": limit,
        "offset": offset
    }


@router.get("/usage-stats")
async def get_usage_stats(
    days: int = 30,
    current_user: User = Depends(get_current_user),
    token_repo: TokenRepository = Depends(get_token_repository)
):
    """Get detailed usage statistics."""
    
    if days > 365:
        days = 365
    
    stats = await token_repo.get_usage_stats(current_user.id, days=days)
    
    return {
        "period_days": days,
        "stats": stats
    }


@router.get("/usage-analysis")
async def get_usage_analysis(
    current_user: User = Depends(get_current_user),
    token_repo: TokenRepository = Depends(get_token_repository)
):
    """Get AI-powered usage pattern analysis and recommendations."""
    
    token_service = create_token_service(token_repo)
    analysis = await token_service.analyze_usage_patterns(current_user.id)
    
    return {
        "analysis": analysis,
        "generated_at": str(uuid.uuid4())  # Analysis ID for tracking
    }


@router.post("/cancel-subscription")
async def cancel_subscription(
    current_user: User = Depends(get_current_user),
    token_repo: TokenRepository = Depends(get_token_repository)
):
    """Cancel current subscription (downgrade at period end)."""
    
    token_service = create_token_service(token_repo)
    result = await token_service.cancel_subscription(current_user.id)
    
    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("error", "Cancellation failed")
        )
    
    return {
        "message": result["message"],
        "period_end": result["period_end"].isoformat(),
        "downgrade_to": result["downgrade_to"]
    }


@router.get("/billing-history")
async def get_billing_history(
    current_user: User = Depends(get_current_user),
    token_repo: TokenRepository = Depends(get_token_repository)
):
    """Get user billing and payment history."""
    
    token_service = create_token_service(token_repo)
    billing_history = await token_service.get_billing_history(current_user.id)
    
    return {
        "billing_history": billing_history
    }


@router.get("/check-affordability/{transaction_type}")
async def check_transaction_affordability(
    transaction_type: str,
    current_user: User = Depends(get_current_user),
    token_repo: TokenRepository = Depends(get_token_repository)
):
    """Check if user can afford a specific transaction type."""
    
    try:
        trans_type = TransactionType(transaction_type)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid transaction type: {transaction_type}"
        )
    
    token_status = await check_user_tokens(
        user_id=current_user.id,
        transaction_type=trans_type,
        token_repo=token_repo
    )
    
    return {
        "can_afford": token_status["has_tokens"],
        "tokens_remaining": token_status["tokens_remaining"],
        "cost_required": token_status["cost_required"],
        "plan_type": token_status["plan_type"]
    }


# Webhook endpoint for payment providers
@router.post("/webhooks/payment")
async def handle_payment_webhook(
    request: Request,
    token_repo: TokenRepository = Depends(get_token_repository)
):
    """Handle payment provider webhooks (Stripe, PayPal, etc.)."""
    
    # Get raw payload for signature verification
    payload = await request.body()
    
    # Verify webhook signature (implement based on payment provider)
    # webhook_signature = request.headers.get("stripe-signature")
    # if not verify_webhook_signature(payload, webhook_signature):
    #     raise HTTPException(status_code=400, detail="Invalid signature")
    
    webhook_data = await request.json()
    
    token_service = create_token_service(token_repo)
    success = await token_service.handle_payment_webhook(webhook_data)
    
    if success:
        return {"status": "processed"}
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Webhook processing failed"
        )


# Admin endpoints (add proper admin authentication)
@router.post("/admin/initialize-plans")
async def initialize_default_plans(
    # admin_user: User = Depends(get_admin_user),  # Implement admin auth
    token_repo: TokenRepository = Depends(get_token_repository)
):
    """Initialize default subscription plans (admin only)."""
    
    from app.db.models.token_system import DEFAULT_PLANS, SubscriptionPlan
    
    created_plans = []
    
    for plan_data in DEFAULT_PLANS:
        # Check if plan already exists
        existing_plan = await token_repo.get_plan_by_type(plan_data["plan_type"])
        if existing_plan:
            continue
        
        # Create new plan
        plan = SubscriptionPlan(**plan_data)
        token_repo.db.add(plan)
        created_plans.append(plan_data["plan_type"])
    
    token_repo.db.commit()
    
    return {
        "message": f"Initialized {len(created_plans)} plans",
        "created_plans": created_plans
    }