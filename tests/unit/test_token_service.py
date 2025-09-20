"""
Unit tests for TokenService business logic.
Tests payment integration, analytics, recommendations, and business rules.
"""

import pytest
from datetime import datetime, timedelta
import uuid
from unittest.mock import Mock, patch, MagicMock
from decimal import Decimal

from app.services.token.token_service import TokenService
from app.repositories.token_repository import TokenRepository
from app.db.models.token_system import (
    UserTokenBalance,
    TokenTransaction,
    SubscriptionPlan,
    PlanType,
    TransactionType,
    DEFAULT_PLANS
)


class TestTokenServiceInitialization:
    """Test TokenService initialization and user onboarding."""
    
    def test_initialize_user_tokens_new_user(self, db_session, sample_user):
        """Test initializing tokens for new user."""
        mock_repo = Mock(spec=TokenRepository)
        mock_repo.get_user_balance.return_value = None  # No existing balance
        mock_repo.create_user_balance.return_value = Mock(
            user_id=sample_user.id,
            plan_type=PlanType.FREE,
            monthly_token_quota=20,
            tokens_used_this_month=0,
            bonus_tokens=0
        )
        
        service = TokenService(mock_repo)
        
        result = service.initialize_user_tokens(sample_user.id)
        
        assert result['success'] is True
        assert result['balance'] is not None
        assert result['message'] == "User tokens initialized successfully"
        mock_repo.create_user_balance.assert_called_once_with(
            user_id=sample_user.id,
            plan_type=PlanType.FREE,
            monthly_quota=20
        )
    
    def test_initialize_user_tokens_existing_user(self, db_session, sample_user, sample_token_balance):
        """Test initializing tokens for existing user."""
        mock_repo = Mock(spec=TokenRepository)
        mock_repo.get_user_balance.return_value = sample_token_balance
        
        service = TokenService(mock_repo)
        
        result = service.initialize_user_tokens(sample_user.id)
        
        assert result['success'] is True
        assert result['balance'] == sample_token_balance
        assert result['message'] == "User tokens already initialized"
        mock_repo.create_user_balance.assert_not_called()
    
    def test_initialize_user_tokens_error_handling(self, db_session, sample_user):
        """Test error handling during initialization."""
        mock_repo = Mock(spec=TokenRepository)
        mock_repo.get_user_balance.return_value = None
        mock_repo.create_user_balance.side_effect = Exception("Database error")
        
        service = TokenService(mock_repo)
        
        result = service.initialize_user_tokens(sample_user.id)
        
        assert result['success'] is False
        assert 'error' in result
        assert "Database error" in result['error']


class TestPlanUpgrade:
    """Test subscription plan upgrade functionality."""
    
    def test_upgrade_user_plan_success(self, db_session, sample_user):
        """Test successful plan upgrade."""
        mock_repo = Mock(spec=TokenRepository)
        mock_repo.upgrade_user_plan.return_value = (True, Mock(
            plan_type=PlanType.PREMIUM,
            monthly_token_quota=200
        ))
        
        service = TokenService(mock_repo)
        
        with patch.object(service, '_process_payment', return_value={'success': True, 'transaction_id': 'pay_123'}):
            result = service.upgrade_user_plan(
                user_id=sample_user.id,
                new_plan=PlanType.PREMIUM,
                payment_method_id="pm_test_123"
            )
        
        assert result['success'] is True
        assert result['new_plan'] == PlanType.PREMIUM
        assert 'payment_confirmation' in result
        mock_repo.upgrade_user_plan.assert_called_once()
    
    def test_upgrade_user_plan_payment_failure(self, db_session, sample_user):
        """Test plan upgrade with payment failure."""
        mock_repo = Mock(spec=TokenRepository)
        service = TokenService(mock_repo)
        
        with patch.object(service, '_process_payment', return_value={'success': False, 'error': 'Card declined'}):
            result = service.upgrade_user_plan(
                user_id=sample_user.id,
                new_plan=PlanType.PREMIUM,
                payment_method_id="pm_invalid"
            )
        
        assert result['success'] is False
        assert 'Card declined' in result['error']
        mock_repo.upgrade_user_plan.assert_not_called()
    
    def test_upgrade_to_same_plan(self, db_session, sample_user, sample_token_balance):
        """Test upgrading to the same plan."""
        mock_repo = Mock(spec=TokenRepository)
        mock_repo.get_user_balance.return_value = sample_token_balance
        
        service = TokenService(mock_repo)
        
        result = service.upgrade_user_plan(
            user_id=sample_user.id,
            new_plan=sample_token_balance.plan_type,
            payment_method_id="pm_test_123"
        )
        
        assert result['success'] is False
        assert 'already on this plan' in result['error'].lower()
    
    def test_upgrade_to_lower_plan(self, db_session, sample_user):
        """Test downgrading to a lower plan."""
        # Create a user with Premium plan
        premium_balance = Mock(
            plan_type=PlanType.PREMIUM,
            monthly_token_quota=200
        )
        
        mock_repo = Mock(spec=TokenRepository)
        mock_repo.get_user_balance.return_value = premium_balance
        mock_repo.upgrade_user_plan.return_value = (True, Mock(
            plan_type=PlanType.BASIC,
            monthly_token_quota=100
        ))
        
        service = TokenService(mock_repo)
        
        result = service.upgrade_user_plan(
            user_id=sample_user.id,
            new_plan=PlanType.BASIC,
            payment_method_id=None  # No payment for downgrade
        )
        
        assert result['success'] is True
        assert result['new_plan'] == PlanType.BASIC
        assert 'downgrade' in result['message'].lower()


class TestTokenPurchase:
    """Test token pack purchase functionality."""
    
    def test_purchase_token_pack_success(self, db_session, sample_user, sample_token_balance):
        """Test successful token pack purchase."""
        mock_repo = Mock(spec=TokenRepository)
        mock_repo.get_user_balance.return_value = sample_token_balance
        mock_repo.add_bonus_tokens.return_value = (True, Mock(
            amount=100,
            transaction_type=TransactionType.PLAN_PURCHASE
        ))
        
        service = TokenService(mock_repo)
        
        with patch.object(service, '_process_payment', return_value={'success': True, 'transaction_id': 'pay_pack_123'}):
            result = service.purchase_token_pack(
                user_id=sample_user.id,
                pack_size="medium",
                payment_method_id="pm_test_123"
            )
        
        assert result['success'] is True
        assert result['tokens_added'] == 100
        assert 'payment_confirmation' in result
        mock_repo.add_bonus_tokens.assert_called_once_with(
            user_id=sample_user.id,
            amount=100,
            reason="Token pack purchase: medium"
        )
    
    def test_purchase_token_pack_invalid_size(self, db_session, sample_user):
        """Test purchasing invalid token pack size."""
        mock_repo = Mock(spec=TokenRepository)
        service = TokenService(mock_repo)
        
        result = service.purchase_token_pack(
            user_id=sample_user.id,
            pack_size="invalid_size",
            payment_method_id="pm_test_123"
        )
        
        assert result['success'] is False
        assert 'invalid pack size' in result['error'].lower()
    
    def test_purchase_token_pack_payment_failure(self, db_session, sample_user, sample_token_balance):
        """Test token pack purchase with payment failure."""
        mock_repo = Mock(spec=TokenRepository)
        mock_repo.get_user_balance.return_value = sample_token_balance
        
        service = TokenService(mock_repo)
        
        with patch.object(service, '_process_payment', return_value={'success': False, 'error': 'Insufficient funds'}):
            result = service.purchase_token_pack(
                user_id=sample_user.id,
                pack_size="large",
                payment_method_id="pm_test_123"
            )
        
        assert result['success'] is False
        assert 'Insufficient funds' in result['error']
        mock_repo.add_bonus_tokens.assert_not_called()


class TestUserDashboard:
    """Test user dashboard data compilation."""
    
    def test_get_user_dashboard_complete(self, db_session, sample_user, sample_token_balance, sample_transactions):
        """Test getting complete user dashboard data."""
        mock_repo = Mock(spec=TokenRepository)
        mock_repo.get_user_balance.return_value = sample_token_balance
        mock_repo.get_user_transactions.return_value = sample_transactions
        mock_repo.get_usage_analytics.return_value = {
            'total_tokens_consumed': 150,
            'total_transactions': 25,
            'most_used_feature': TransactionType.BASIC_ANALYSIS,
            'daily_usage': [5, 8, 12, 3, 7, 9, 6]
        }
        
        service = TokenService(mock_repo)
        
        dashboard = service.get_user_dashboard(sample_user.id)
        
        assert dashboard['success'] is True
        assert 'balance' in dashboard
        assert 'recent_transactions' in dashboard
        assert 'usage_analytics' in dashboard
        assert 'recommendations' in dashboard
        assert 'available_plans' in dashboard
        
        # Check analytics
        analytics = dashboard['usage_analytics']
        assert analytics['total_tokens_consumed'] == 150
        assert analytics['most_used_feature'] == TransactionType.BASIC_ANALYSIS
    
    def test_get_user_dashboard_new_user(self, db_session, sample_user):
        """Test dashboard for new user with no transactions."""
        mock_repo = Mock(spec=TokenRepository)
        mock_repo.get_user_balance.return_value = None
        
        service = TokenService(mock_repo)
        
        dashboard = service.get_user_dashboard(sample_user.id)
        
        assert dashboard['success'] is False
        assert 'not initialized' in dashboard['error'].lower()
    
    def test_get_user_dashboard_with_recommendations(self, db_session, sample_user, sample_token_balance):
        """Test dashboard includes personalized recommendations."""
        # High usage user
        high_usage_balance = Mock(
            user_id=sample_user.id,
            plan_type=PlanType.FREE,
            monthly_token_quota=20,
            tokens_used_this_month=18,
            tokens_remaining=2,
            usage_percentage=90.0
        )
        
        mock_repo = Mock(spec=TokenRepository)
        mock_repo.get_user_balance.return_value = high_usage_balance
        mock_repo.get_user_transactions.return_value = []
        mock_repo.get_usage_analytics.return_value = {
            'total_tokens_consumed': 18,
            'total_transactions': 18,
            'most_used_feature': TransactionType.EXPERT_ANALYSIS,
            'daily_usage': [3, 3, 3, 3, 3, 3, 0]
        }
        
        service = TokenService(mock_repo)
        
        dashboard = service.get_user_dashboard(sample_user.id)
        
        recommendations = dashboard['recommendations']
        assert len(recommendations) > 0
        
        # Should recommend upgrade for high usage
        upgrade_rec = next((r for r in recommendations if r['type'] == 'upgrade'), None)
        assert upgrade_rec is not None
        assert upgrade_rec['priority'] == 'high'


class TestUsageAnalytics:
    """Test usage analytics and insights."""
    
    def test_analyze_usage_patterns_heavy_user(self, db_session, sample_user):
        """Test usage pattern analysis for heavy user."""
        mock_repo = Mock(spec=TokenRepository)
        mock_repo.get_usage_analytics.return_value = {
            'total_tokens_consumed': 180,
            'total_transactions': 45,
            'most_used_feature': TransactionType.EXPERT_ANALYSIS,
            'daily_usage': [25, 30, 35, 20, 25, 30, 15]
        }
        
        service = TokenService(mock_repo)
        
        patterns = service.analyze_usage_patterns(sample_user.id, days=7)
        
        assert patterns['usage_level'] == 'heavy'
        assert patterns['primary_feature'] == TransactionType.EXPERT_ANALYSIS
        assert patterns['daily_average'] > 20
        assert 'peak_usage_day' in patterns
    
    def test_analyze_usage_patterns_light_user(self, db_session, sample_user):
        """Test usage pattern analysis for light user."""
        mock_repo = Mock(spec=TokenRepository)
        mock_repo.get_usage_analytics.return_value = {
            'total_tokens_consumed': 5,
            'total_transactions': 5,
            'most_used_feature': TransactionType.BASIC_ANALYSIS,
            'daily_usage': [1, 0, 2, 1, 0, 1, 0]
        }
        
        service = TokenService(mock_repo)
        
        patterns = service.analyze_usage_patterns(sample_user.id, days=7)
        
        assert patterns['usage_level'] == 'light'
        assert patterns['daily_average'] < 5
    
    def test_analyze_usage_patterns_no_data(self, db_session, sample_user):
        """Test usage pattern analysis with no data."""
        mock_repo = Mock(spec=TokenRepository)
        mock_repo.get_usage_analytics.return_value = {
            'total_tokens_consumed': 0,
            'total_transactions': 0,
            'most_used_feature': None,
            'daily_usage': [0, 0, 0, 0, 0, 0, 0]
        }
        
        service = TokenService(mock_repo)
        
        patterns = service.analyze_usage_patterns(sample_user.id, days=7)
        
        assert patterns['usage_level'] == 'inactive'
        assert patterns['daily_average'] == 0


class TestRecommendationEngine:
    """Test recommendation engine logic."""
    
    def test_generate_recommendations_high_usage_free_user(self, db_session, sample_user):
        """Test recommendations for high usage free user."""
        high_usage_balance = Mock(
            plan_type=PlanType.FREE,
            usage_percentage=95.0,
            tokens_remaining=1
        )
        
        analytics = {
            'total_tokens_consumed': 19,
            'most_used_feature': TransactionType.EXPERT_ANALYSIS
        }
        
        service = TokenService(Mock())
        
        recommendations = service._generate_recommendations(high_usage_balance, analytics)
        
        # Should recommend upgrade
        upgrade_rec = next((r for r in recommendations if r['type'] == 'upgrade'), None)
        assert upgrade_rec is not None
        assert upgrade_rec['priority'] == 'high'
        assert PlanType.BASIC in upgrade_rec['suggested_plan']
    
    def test_generate_recommendations_moderate_usage(self, db_session, sample_user):
        """Test recommendations for moderate usage user."""
        moderate_balance = Mock(
            plan_type=PlanType.BASIC,
            usage_percentage=60.0,
            tokens_remaining=40
        )
        
        analytics = {
            'total_tokens_consumed': 60,
            'most_used_feature': TransactionType.BASIC_ANALYSIS
        }
        
        service = TokenService(Mock())
        
        recommendations = service._generate_recommendations(moderate_balance, analytics)
        
        # Should suggest optimizations
        optimization_rec = next((r for r in recommendations if r['type'] == 'optimization'), None)
        assert optimization_rec is not None
    
    def test_generate_recommendations_low_usage_premium(self, db_session, sample_user):
        """Test recommendations for low usage premium user."""
        low_usage_balance = Mock(
            plan_type=PlanType.PREMIUM,
            usage_percentage=15.0,
            tokens_remaining=170
        )
        
        analytics = {
            'total_tokens_consumed': 30,
            'most_used_feature': TransactionType.BASIC_ANALYSIS
        }
        
        service = TokenService(Mock())
        
        recommendations = service._generate_recommendations(low_usage_balance, analytics)
        
        # Should suggest downgrade
        downgrade_rec = next((r for r in recommendations if r['type'] == 'downgrade'), None)
        assert downgrade_rec is not None
        assert downgrade_rec['suggested_plan'] == PlanType.BASIC


class TestPaymentIntegration:
    """Test payment processing integration."""
    
    def test_process_payment_success(self, db_session):
        """Test successful payment processing."""
        service = TokenService(Mock())
        
        with patch('app.services.token.token_service.stripe') as mock_stripe:
            mock_stripe.PaymentIntent.create.return_value = Mock(
                id='pi_test_123',
                status='succeeded',
                amount=999,
                currency='eur'
            )
            
            result = service._process_payment(
                amount_eur=9.99,
                payment_method_id="pm_test_123",
                description="Basic Plan Upgrade"
            )
        
        assert result['success'] is True
        assert result['transaction_id'] == 'pi_test_123'
        assert result['amount'] == 999
    
    def test_process_payment_failure(self, db_session):
        """Test payment processing failure."""
        service = TokenService(Mock())
        
        with patch('app.services.token.token_service.stripe') as mock_stripe:
            mock_stripe.PaymentIntent.create.side_effect = Exception("Card declined")
            
            result = service._process_payment(
                amount_eur=9.99,
                payment_method_id="pm_invalid",
                description="Failed Payment"
            )
        
        assert result['success'] is False
        assert 'Card declined' in result['error']
    
    def test_process_payment_zero_amount(self, db_session):
        """Test processing zero amount payment (downgrade)."""
        service = TokenService(Mock())
        
        result = service._process_payment(
            amount_eur=0.0,
            payment_method_id=None,
            description="Downgrade to Free"
        )
        
        assert result['success'] is True
        assert result['transaction_id'] == 'no_payment_required'


class TestMonthlyReset:
    """Test monthly quota reset functionality."""
    
    def test_process_monthly_resets_due_users(self, db_session):
        """Test processing monthly resets for users who need them."""
        mock_repo = Mock(spec=TokenRepository)
        
        # Mock users with expired periods
        expired_users = [
            Mock(id=uuid.uuid4(), needs_renewal=True),
            Mock(id=uuid.uuid4(), needs_renewal=True),
        ]
        
        mock_repo.get_users_needing_reset.return_value = expired_users
        mock_repo.reset_monthly_quota.return_value = (True, Mock())
        
        service = TokenService(mock_repo)
        
        result = service.process_monthly_resets()
        
        assert result['success'] is True
        assert result['users_reset'] == 2
        assert mock_repo.reset_monthly_quota.call_count == 2
    
    def test_process_monthly_resets_no_users(self, db_session):
        """Test monthly reset when no users need reset."""
        mock_repo = Mock(spec=TokenRepository)
        mock_repo.get_users_needing_reset.return_value = []
        
        service = TokenService(mock_repo)
        
        result = service.process_monthly_resets()
        
        assert result['success'] is True
        assert result['users_reset'] == 0
        mock_repo.reset_monthly_quota.assert_not_called()


class TestRefundAndSupport:
    """Test refund and customer support functionality."""
    
    def test_process_refund_success(self, db_session, sample_user, sample_transactions):
        """Test successful refund processing."""
        original_transaction = sample_transactions[0]
        original_transaction.amount = -5  # Consumption
        
        mock_repo = Mock(spec=TokenRepository)
        mock_repo.get_transaction_by_id.return_value = original_transaction
        mock_repo.refund_transaction.return_value = (True, Mock(
            amount=5,
            transaction_type=TransactionType.REFUND
        ))
        
        service = TokenService(mock_repo)
        
        with patch.object(service, '_process_refund_payment', return_value={'success': True}):
            result = service.process_refund(
                user_id=sample_user.id,
                transaction_id=original_transaction.id,
                refund_amount=5,
                reason="Service failure"
            )
        
        assert result['success'] is True
        assert result['refund_amount'] == 5
        mock_repo.refund_transaction.assert_called_once()
    
    def test_process_refund_invalid_transaction(self, db_session, sample_user):
        """Test refund for invalid transaction."""
        mock_repo = Mock(spec=TokenRepository)
        mock_repo.get_transaction_by_id.return_value = None
        
        service = TokenService(mock_repo)
        
        result = service.process_refund(
            user_id=sample_user.id,
            transaction_id=uuid.uuid4(),
            refund_amount=5,
            reason="Invalid transaction"
        )
        
        assert result['success'] is False
        assert 'transaction not found' in result['error'].lower()
    
    def test_process_partial_refund(self, db_session, sample_user, sample_transactions):
        """Test partial refund processing."""
        original_transaction = sample_transactions[0]
        original_transaction.amount = -10  # Consumed 10 tokens
        
        mock_repo = Mock(spec=TokenRepository)
        mock_repo.get_transaction_by_id.return_value = original_transaction
        mock_repo.refund_transaction.return_value = (True, Mock(
            amount=3,  # Partial refund
            transaction_type=TransactionType.REFUND
        ))
        
        service = TokenService(mock_repo)
        
        with patch.object(service, '_process_refund_payment', return_value={'success': True}):
            result = service.process_refund(
                user_id=sample_user.id,
                transaction_id=original_transaction.id,
                refund_amount=3,
                reason="Partial service failure"
            )
        
        assert result['success'] is True
        assert result['refund_amount'] == 3


class TestBusinessRules:
    """Test business rules and validation."""
    
    def test_validate_plan_upgrade_rules(self, db_session):
        """Test plan upgrade validation rules."""
        service = TokenService(Mock())
        
        # Valid upgrades
        assert service._validate_plan_upgrade(PlanType.FREE, PlanType.BASIC) is True
        assert service._validate_plan_upgrade(PlanType.BASIC, PlanType.PREMIUM) is True
        assert service._validate_plan_upgrade(PlanType.PREMIUM, PlanType.ENTERPRISE) is True
        
        # Same plan
        assert service._validate_plan_upgrade(PlanType.BASIC, PlanType.BASIC) is False
        
        # Valid downgrades
        assert service._validate_plan_upgrade(PlanType.PREMIUM, PlanType.BASIC) is True
        assert service._validate_plan_upgrade(PlanType.ENTERPRISE, PlanType.FREE) is True
    
    def test_calculate_plan_price(self, db_session):
        """Test plan price calculation."""
        service = TokenService(Mock())
        
        for plan_data in DEFAULT_PLANS:
            if plan_data['plan_type'] != PlanType.FREE:
                price = service._calculate_plan_price(PlanType(plan_data['plan_type']))
                assert price > 0
                assert isinstance(price, (int, float, Decimal))
        
        # Free plan should be 0
        free_price = service._calculate_plan_price(PlanType.FREE)
        assert free_price == 0
    
    def test_token_pack_pricing(self, db_session):
        """Test token pack pricing calculation."""
        service = TokenService(Mock())
        
        pack_configs = service._get_token_pack_configs()
        
        assert 'small' in pack_configs
        assert 'medium' in pack_configs
        assert 'large' in pack_configs
        
        for pack_name, config in pack_configs.items():
            assert config['tokens'] > 0
            assert config['price_eur'] > 0
            assert config['tokens'] / config['price_eur'] > 0  # Token per euro ratio


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""
    
    def test_service_with_database_errors(self, db_session, sample_user):
        """Test service behavior with database errors."""
        mock_repo = Mock(spec=TokenRepository)
        mock_repo.get_user_balance.side_effect = Exception("Database connection lost")
        
        service = TokenService(mock_repo)
        
        result = service.get_user_dashboard(sample_user.id)
        
        assert result['success'] is False
        assert 'error' in result
    
    def test_service_with_none_values(self, db_session):
        """Test service with None values."""
        mock_repo = Mock(spec=TokenRepository)
        service = TokenService(mock_repo)
        
        # None user_id should be handled gracefully
        result = service.get_user_dashboard(None)
        assert result['success'] is False
        
        result = service.initialize_user_tokens(None)
        assert result['success'] is False
    
    def test_concurrent_operations_handling(self, db_session, sample_user):
        """Test handling of concurrent operations."""
        mock_repo = Mock(spec=TokenRepository)
        
        # Simulate race condition in balance updates
        mock_repo.get_user_balance.side_effect = [
            Mock(tokens_remaining=10),  # First check
            Mock(tokens_remaining=5),   # Second check (changed by concurrent operation)
        ]
        
        service = TokenService(mock_repo)
        
        # Service should handle this gracefully
        with patch.object(service, '_process_payment', return_value={'success': True}):
            result = service.upgrade_user_plan(
                user_id=sample_user.id,
                new_plan=PlanType.BASIC,
                payment_method_id="pm_test"
            )
        
        # Should either succeed or fail gracefully, not crash
        assert isinstance(result['success'], bool)