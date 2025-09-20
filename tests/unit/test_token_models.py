"""
Unit tests for token system models.
Tests all business logic, computed properties, and edge cases.
"""

import pytest
from datetime import datetime, timedelta
import uuid

from app.db.models.token_system import (
    UserTokenBalance,
    TokenTransaction,
    SubscriptionPlan,
    PlanType,
    TransactionType,
    TokenCosts,
    DEFAULT_PLANS
)
from app.db.models.user import User


class TestUserTokenBalance:
    """Test UserTokenBalance model and business logic."""
    
    def test_tokens_remaining_calculation(self, db_session):
        """Test tokens_remaining property calculation."""
        user_id = uuid.uuid4()
        now = datetime.utcnow()
        
        balance = UserTokenBalance(
            user_id=user_id,
            plan_type=PlanType.BASIC,
            monthly_token_quota=100,
            tokens_used_this_month=30,
            bonus_tokens=15,
            current_period_start=now,
            current_period_end=now + timedelta(days=30),
            last_reset_date=now
        )
        
        # Monthly remaining (100 - 30) + bonus (15) = 85
        assert balance.tokens_remaining == 85
        assert balance.tokens_available == 85  # Alias
    
    def test_tokens_remaining_with_overuse(self, db_session):
        """Test tokens_remaining when monthly quota is exceeded."""
        user_id = uuid.uuid4()
        now = datetime.utcnow()
        
        balance = UserTokenBalance(
            user_id=user_id,
            monthly_token_quota=50,
            tokens_used_this_month=60,  # Over quota
            bonus_tokens=20,
            current_period_start=now,
            current_period_end=now + timedelta(days=30),
            last_reset_date=now
        )
        
        # Monthly remaining = max(0, 50-60) + bonus (20) = 20
        assert balance.tokens_remaining == 20
    
    def test_usage_percentage_calculation(self, db_session):
        """Test usage_percentage property."""
        user_id = uuid.uuid4()
        now = datetime.utcnow()
        
        balance = UserTokenBalance(
            user_id=user_id,
            monthly_token_quota=200,
            tokens_used_this_month=50,
            current_period_start=now,
            current_period_end=now + timedelta(days=30),
            last_reset_date=now
        )
        
        assert balance.usage_percentage == 25.0  # 50/200 * 100
    
    def test_usage_percentage_over_quota(self, db_session):
        """Test usage_percentage when over quota."""
        user_id = uuid.uuid4()
        now = datetime.utcnow()
        
        balance = UserTokenBalance(
            user_id=user_id,
            monthly_token_quota=100,
            tokens_used_this_month=150,  # 150% usage
            current_period_start=now,
            current_period_end=now + timedelta(days=30),
            last_reset_date=now
        )
        
        # Should be capped at 100%
        assert balance.usage_percentage == 100.0
    
    def test_usage_percentage_zero_quota(self, db_session):
        """Test usage_percentage with zero quota (edge case)."""
        user_id = uuid.uuid4()
        now = datetime.utcnow()
        
        balance = UserTokenBalance(
            user_id=user_id,
            monthly_token_quota=0,
            tokens_used_this_month=10,
            current_period_start=now,
            current_period_end=now + timedelta(days=30),
            last_reset_date=now
        )
        
        assert balance.usage_percentage == 0.0
    
    def test_needs_renewal_property(self, db_session):
        """Test needs_renewal property."""
        user_id = uuid.uuid4()
        now = datetime.utcnow()
        
        # Not expired
        balance_current = UserTokenBalance(
            user_id=user_id,
            current_period_end=now + timedelta(days=10),
            current_period_start=now,
            last_reset_date=now
        )
        assert not balance_current.needs_renewal
        
        # Expired
        balance_expired = UserTokenBalance(
            user_id=user_id,
            current_period_end=now - timedelta(days=1),
            current_period_start=now - timedelta(days=31),
            last_reset_date=now - timedelta(days=31)
        )
        assert balance_expired.needs_renewal
    
    def test_can_afford_method(self, db_session):
        """Test can_afford method with various scenarios."""
        user_id = uuid.uuid4()
        now = datetime.utcnow()
        
        balance = UserTokenBalance(
            user_id=user_id,
            monthly_token_quota=100,
            tokens_used_this_month=80,
            bonus_tokens=10,
            current_period_start=now,
            current_period_end=now + timedelta(days=30),
            last_reset_date=now
        )
        
        # Total available: (100-80) + 10 = 30 tokens
        assert balance.can_afford(1)
        assert balance.can_afford(30)
        assert not balance.can_afford(31)
        assert not balance.can_afford(100)
    
    def test_can_afford_edge_cases(self, db_session):
        """Test can_afford edge cases."""
        user_id = uuid.uuid4()
        now = datetime.utcnow()
        
        # Zero tokens available
        balance_empty = UserTokenBalance(
            user_id=user_id,
            monthly_token_quota=50,
            tokens_used_this_month=50,
            bonus_tokens=0,
            current_period_start=now,
            current_period_end=now + timedelta(days=30),
            last_reset_date=now
        )
        assert not balance_empty.can_afford(1)
        assert balance_empty.can_afford(0)
        
        # Negative cost (should work)
        balance_normal = UserTokenBalance(
            user_id=user_id,
            monthly_token_quota=100,
            tokens_used_this_month=0,
            bonus_tokens=0,
            current_period_start=now,
            current_period_end=now + timedelta(days=30),
            last_reset_date=now
        )
        assert balance_normal.can_afford(-5)  # Refund scenario


class TestTokenTransaction:
    """Test TokenTransaction model and properties."""
    
    def test_is_consumption_property(self, db_session):
        """Test is_consumption property."""
        user_id = uuid.uuid4()
        
        consumption_tx = TokenTransaction(
            user_id=user_id,
            amount=-5,
            transaction_type=TransactionType.EXPERT_ANALYSIS,
            plan_type_at_time=PlanType.FREE
        )
        assert consumption_tx.is_consumption
        assert not consumption_tx.is_credit
        
        credit_tx = TokenTransaction(
            user_id=user_id,
            amount=10,
            transaction_type=TransactionType.BONUS_CREDIT,
            plan_type_at_time=PlanType.FREE
        )
        assert not credit_tx.is_consumption
        assert credit_tx.is_credit
    
    def test_zero_amount_transaction(self, db_session):
        """Test transaction with zero amount (tracking only)."""
        user_id = uuid.uuid4()
        
        tracking_tx = TokenTransaction(
            user_id=user_id,
            amount=0,
            transaction_type="monthly_reset",
            plan_type_at_time=PlanType.BASIC
        )
        assert not tracking_tx.is_consumption
        assert not tracking_tx.is_credit
    
    def test_transaction_validation(self, db_session):
        """Test transaction data validation."""
        from tests.conftest import assert_transaction_valid
        
        user_id = uuid.uuid4()
        now = datetime.utcnow()
        
        valid_tx = TokenTransaction(
            user_id=user_id,
            amount=-3,
            transaction_type=TransactionType.CHAT_INTERACTION,
            feature_used="chat_message",
            plan_type_at_time=PlanType.PREMIUM,
            created_at=now
        )
        
        # Should not raise any assertion errors
        assert_transaction_valid(valid_tx)


class TestTokenCosts:
    """Test TokenCosts utility class."""
    
    def test_get_cost_for_all_transaction_types(self):
        """Test that all transaction types have defined costs."""
        for transaction_type in TransactionType:
            if transaction_type.value.endswith(('_credit', '_refund', 'reset')):
                continue  # Skip non-billable types
                
            cost = TokenCosts.get_cost(transaction_type)
            assert isinstance(cost, int)
            assert cost >= 0
    
    def test_specific_cost_values(self):
        """Test specific cost values match expectations."""
        assert TokenCosts.get_cost(TransactionType.BASIC_ANALYSIS) == 1
        assert TokenCosts.get_cost(TransactionType.EXPERT_ANALYSIS) == 5
        assert TokenCosts.get_cost(TransactionType.MULTI_CONDITION_ANALYSIS) == 7
        assert TokenCosts.get_cost(TransactionType.ALTERNATIVES_GENERATION) == 3
        assert TokenCosts.get_cost(TransactionType.CHAT_INTERACTION) == 1
        assert TokenCosts.get_cost(TransactionType.IMAGE_OCR_PROCESSING) == 2
    
    def test_unknown_transaction_type_fallback(self):
        """Test fallback cost for unknown transaction types."""
        # Create a mock transaction type
        unknown_type = "unknown_feature"
        cost = TokenCosts.get_cost(unknown_type)
        assert cost == 1  # Default fallback


class TestSubscriptionPlan:
    """Test SubscriptionPlan model."""
    
    def test_default_plans_validity(self, db_session):
        """Test that all DEFAULT_PLANS are valid."""
        for plan_data in DEFAULT_PLANS:
            plan = SubscriptionPlan(**plan_data)
            
            # Basic validations
            assert plan.plan_type in [p.value for p in PlanType]
            assert plan.monthly_token_quota >= 0
            assert plan.price_eur >= 0
            assert plan.price_usd >= 0
            assert isinstance(plan.allowed_features, list)
            assert plan.name is not None
            assert len(plan.name) > 0
    
    def test_plan_feature_validation(self, db_session):
        """Test that plan features are valid transaction types."""
        for plan_data in DEFAULT_PLANS:
            plan = SubscriptionPlan(**plan_data)
            
            for feature in plan.allowed_features:
                # Should be a valid TransactionType or special permission
                try:
                    TransactionType(feature)
                except ValueError:
                    # Allow special features that aren't transaction types
                    assert feature in ["all_features", "unlimited_analysis"]
    
    def test_plan_pricing_logic(self, db_session):
        """Test that plan pricing follows logical progression."""
        plans = [SubscriptionPlan(**data) for data in DEFAULT_PLANS]
        
        # Sort by price
        plans.sort(key=lambda p: p.price_eur)
        
        # Prices should be increasing
        for i in range(1, len(plans)):
            assert plans[i].price_eur >= plans[i-1].price_eur
            assert plans[i].monthly_token_quota >= plans[i-1].monthly_token_quota


class TestPlanType:
    """Test PlanType enum."""
    
    def test_all_plan_types_defined(self):
        """Test that all expected plan types exist."""
        expected_plans = ["free", "basic", "premium", "enterprise"]
        
        for plan_name in expected_plans:
            assert hasattr(PlanType, plan_name.upper())
            assert PlanType(plan_name).value == plan_name
    
    def test_plan_type_string_conversion(self):
        """Test PlanType string conversion."""
        assert PlanType.FREE.value == "free"
        assert PlanType.BASIC.value == "basic"
        assert PlanType.PREMIUM.value == "premium"
        assert PlanType.ENTERPRISE.value == "enterprise"


class TestTransactionType:
    """Test TransactionType enum."""
    
    def test_all_transaction_types_defined(self):
        """Test that all expected transaction types exist."""
        expected_types = [
            "basic_analysis", "expert_analysis", "multi_condition_analysis",
            "alternatives_generation", "chat_interaction", "image_ocr_processing",
            "detailed_recommendations", "meal_planning", "progress_tracking",
            "plan_purchase", "bonus_credit", "refund"
        ]
        
        for tx_type in expected_types:
            # Should not raise ValueError
            TransactionType(tx_type)
    
    def test_transaction_type_categories(self):
        """Test transaction type categorization."""
        consumption_types = [
            TransactionType.BASIC_ANALYSIS,
            TransactionType.EXPERT_ANALYSIS,
            TransactionType.CHAT_INTERACTION,
            TransactionType.IMAGE_OCR_PROCESSING
        ]
        
        credit_types = [
            TransactionType.PLAN_PURCHASE,
            TransactionType.BONUS_CREDIT,
            TransactionType.REFUND
        ]
        
        # All consumption types should have positive costs
        for tx_type in consumption_types:
            assert TokenCosts.get_cost(tx_type) > 0
        
        # Credit types should typically not have defined costs
        for tx_type in credit_types:
            cost = TokenCosts.get_cost(tx_type)
            assert cost >= 0  # Can be 0 or positive


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_token_balance_with_extreme_values(self, db_session):
        """Test token balance with extreme values."""
        user_id = uuid.uuid4()
        now = datetime.utcnow()
        
        # Very large numbers
        large_balance = UserTokenBalance(
            user_id=user_id,
            monthly_token_quota=999999999,
            tokens_used_this_month=999999998,
            bonus_tokens=1000000,
            current_period_start=now,
            current_period_end=now + timedelta(days=30),
            last_reset_date=now
        )
        
        assert large_balance.tokens_remaining == 1000001  # (999999999-999999998) + 1000000
        assert large_balance.usage_percentage == 100.0  # Should be capped
    
    def test_token_balance_with_far_future_dates(self, db_session):
        """Test token balance with far future dates."""
        user_id = uuid.uuid4()
        now = datetime.utcnow()
        far_future = now + timedelta(days=365 * 10)  # 10 years
        
        balance = UserTokenBalance(
            user_id=user_id,
            current_period_start=now,
            current_period_end=far_future,
            last_reset_date=now,
            monthly_token_quota=100,
            tokens_used_this_month=0,
            bonus_tokens=0
        )
        
        assert not balance.needs_renewal
        assert balance.tokens_remaining == 100
    
    def test_transaction_with_extreme_processing_time(self, db_session):
        """Test transaction with extreme processing times."""
        user_id = uuid.uuid4()
        
        # Very slow transaction (1 hour = 3,600,000 ms)
        slow_tx = TokenTransaction(
            user_id=user_id,
            amount=-1,
            transaction_type=TransactionType.EXPERT_ANALYSIS,
            processing_time_ms=3600000,
            plan_type_at_time=PlanType.FREE
        )
        
        assert slow_tx.processing_time_ms == 3600000
        assert slow_tx.is_consumption
    
    def test_unicode_and_special_characters(self, db_session):
        """Test handling of unicode and special characters."""
        user_id = uuid.uuid4()
        
        # Transaction with unicode in feature name
        unicode_tx = TokenTransaction(
            user_id=user_id,
            amount=-2,
            transaction_type=TransactionType.CHAT_INTERACTION,
            feature_used="analyse_nutritionnelle_français_🥗",
            plan_type_at_time=PlanType.BASIC
        )
        
        assert unicode_tx.feature_used == "analyse_nutritionnelle_français_🥗"