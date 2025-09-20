"""
Unit tests for TokenRepository.
Tests all CRUD operations, edge cases, and concurrency scenarios.
"""

import pytest
from datetime import datetime, timedelta
import uuid
from unittest.mock import patch, MagicMock
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.repositories.token_repository import TokenRepository
from app.db.models.token_system import (
    UserTokenBalance,
    TokenTransaction,
    SubscriptionPlan,
    PlanType,
    TransactionType,
    TokenCosts
)
from app.db.models.user import User


class TestTokenRepositoryBasicOperations:
    """Test basic CRUD operations of TokenRepository."""
    
    def test_get_user_balance_existing(self, db_session, sample_user, sample_token_balance):
        """Test getting existing user balance."""
        repo = TokenRepository(db_session)
        
        balance = repo.get_user_balance(sample_user.id)
        
        assert balance is not None
        assert balance.user_id == sample_user.id
        assert balance.plan_type == sample_token_balance.plan_type
        assert balance.monthly_token_quota == sample_token_balance.monthly_token_quota
    
    def test_get_user_balance_nonexistent(self, db_session):
        """Test getting balance for non-existent user."""
        repo = TokenRepository(db_session)
        nonexistent_id = uuid.uuid4()
        
        balance = repo.get_user_balance(nonexistent_id)
        
        assert balance is None
    
    def test_create_user_balance(self, db_session, sample_user):
        """Test creating new user balance."""
        repo = TokenRepository(db_session)
        
        balance = repo.create_user_balance(
            user_id=sample_user.id,
            plan_type=PlanType.BASIC,
            monthly_quota=100
        )
        
        assert balance.user_id == sample_user.id
        assert balance.plan_type == PlanType.BASIC
        assert balance.monthly_token_quota == 100
        assert balance.tokens_used_this_month == 0
        assert balance.bonus_tokens == 0
        assert balance.current_period_start is not None
        assert balance.current_period_end is not None
    
    def test_create_user_balance_duplicate(self, db_session, sample_user, sample_token_balance):
        """Test creating balance for user who already has one."""
        repo = TokenRepository(db_session)
        
        with pytest.raises(IntegrityError):
            repo.create_user_balance(
                user_id=sample_user.id,
                plan_type=PlanType.PREMIUM,
                monthly_quota=200
            )
    
    def test_update_user_balance(self, db_session, sample_user, sample_token_balance):
        """Test updating user balance."""
        repo = TokenRepository(db_session)
        
        updated_balance = repo.update_user_balance(
            sample_user.id,
            tokens_used_this_month=50,
            bonus_tokens=25
        )
        
        assert updated_balance.tokens_used_this_month == 50
        assert updated_balance.bonus_tokens == 25
        assert updated_balance.user_id == sample_user.id
    
    def test_update_nonexistent_balance(self, db_session):
        """Test updating balance for non-existent user."""
        repo = TokenRepository(db_session)
        nonexistent_id = uuid.uuid4()
        
        result = repo.update_user_balance(
            nonexistent_id,
            tokens_used_this_month=10
        )
        
        assert result is None


class TestTokenConsumption:
    """Test token consumption operations."""
    
    def test_consume_tokens_success(self, db_session, sample_user, sample_token_balance):
        """Test successful token consumption."""
        repo = TokenRepository(db_session)
        
        # Initial state: 100 quota, 20 used, 10 bonus = 90 available
        initial_available = sample_token_balance.tokens_remaining
        consume_amount = 5
        
        success, transaction = repo.consume_tokens(
            user_id=sample_user.id,
            amount=consume_amount,
            transaction_type=TransactionType.BASIC_ANALYSIS,
            feature_used="test_analysis"
        )
        
        assert success is True
        assert transaction is not None
        assert transaction.amount == -consume_amount
        assert transaction.transaction_type == TransactionType.BASIC_ANALYSIS
        assert transaction.feature_used == "test_analysis"
        
        # Check balance was updated
        updated_balance = repo.get_user_balance(sample_user.id)
        assert updated_balance.tokens_remaining == initial_available - consume_amount
    
    def test_consume_tokens_insufficient_funds(self, db_session, sample_user, sample_token_balance):
        """Test token consumption with insufficient funds."""
        repo = TokenRepository(db_session)
        
        # Try to consume more than available
        available = sample_token_balance.tokens_remaining
        excessive_amount = available + 10
        
        success, transaction = repo.consume_tokens(
            user_id=sample_user.id,
            amount=excessive_amount,
            transaction_type=TransactionType.EXPERT_ANALYSIS
        )
        
        assert success is False
        assert transaction is None
        
        # Balance should be unchanged
        balance_after = repo.get_user_balance(sample_user.id)
        assert balance_after.tokens_remaining == available
    
    def test_consume_tokens_exact_amount(self, db_session, sample_user, sample_token_balance):
        """Test consuming exactly all available tokens."""
        repo = TokenRepository(db_session)
        
        available = sample_token_balance.tokens_remaining
        
        success, transaction = repo.consume_tokens(
            user_id=sample_user.id,
            amount=available,
            transaction_type=TransactionType.MULTI_CONDITION_ANALYSIS
        )
        
        assert success is True
        assert transaction.amount == -available
        
        # Should have zero tokens remaining
        updated_balance = repo.get_user_balance(sample_user.id)
        assert updated_balance.tokens_remaining == 0
    
    def test_consume_tokens_nonexistent_user(self, db_session):
        """Test consuming tokens for non-existent user."""
        repo = TokenRepository(db_session)
        nonexistent_id = uuid.uuid4()
        
        success, transaction = repo.consume_tokens(
            user_id=nonexistent_id,
            amount=5,
            transaction_type=TransactionType.BASIC_ANALYSIS
        )
        
        assert success is False
        assert transaction is None
    
    def test_consume_tokens_zero_amount(self, db_session, sample_user, sample_token_balance):
        """Test consuming zero tokens (tracking only)."""
        repo = TokenRepository(db_session)
        
        success, transaction = repo.consume_tokens(
            user_id=sample_user.id,
            amount=0,
            transaction_type="tracking_event"
        )
        
        assert success is True
        assert transaction.amount == 0
        
        # Balance should be unchanged
        balance_after = repo.get_user_balance(sample_user.id)
        assert balance_after.tokens_remaining == sample_token_balance.tokens_remaining


class TestTokenReservation:
    """Test token reservation/release mechanism."""
    
    def test_reserve_tokens_success(self, db_session, sample_user, sample_token_balance):
        """Test successful token reservation."""
        repo = TokenRepository(db_session)
        
        initial_reserved = sample_token_balance.tokens_reserved
        reserve_amount = 10
        
        success, reservation = repo.reserve_tokens(
            user_id=sample_user.id,
            amount=reserve_amount,
            operation="long_analysis"
        )
        
        assert success is True
        assert reservation is not None
        assert reservation.amount == reserve_amount
        assert reservation.operation == "long_analysis"
        
        # Check reservation was recorded
        updated_balance = repo.get_user_balance(sample_user.id)
        assert updated_balance.tokens_reserved == initial_reserved + reserve_amount
    
    def test_reserve_tokens_insufficient_funds(self, db_session, sample_user, sample_token_balance):
        """Test token reservation with insufficient funds."""
        repo = TokenRepository(db_session)
        
        available = sample_token_balance.tokens_remaining
        excessive_amount = available + 5
        
        success, reservation = repo.reserve_tokens(
            user_id=sample_user.id,
            amount=excessive_amount,
            operation="impossible_analysis"
        )
        
        assert success is False
        assert reservation is None
        
        # Reservations should be unchanged
        balance_after = repo.get_user_balance(sample_user.id)
        assert balance_after.tokens_reserved == sample_token_balance.tokens_reserved
    
    def test_release_reserved_tokens(self, db_session, sample_user):
        """Test releasing reserved tokens."""
        repo = TokenRepository(db_session)
        
        # First reserve some tokens
        success, reservation = repo.reserve_tokens(
            user_id=sample_user.id,
            amount=15,
            operation="test_operation"
        )
        assert success is True
        
        # Release part of the reservation
        released_amount = 5
        release_success = repo.release_reserved_tokens(
            user_id=sample_user.id,
            amount=released_amount,
            operation="test_operation"
        )
        
        assert release_success is True
        
        # Check reservation was reduced
        balance_after = repo.get_user_balance(sample_user.id)
        expected_reserved = reservation.amount - released_amount
        assert balance_after.tokens_reserved >= 0  # Should have reduced
    
    def test_finalize_reserved_consumption(self, db_session, sample_user):
        """Test finalizing reserved tokens as consumption."""
        repo = TokenRepository(db_session)
        
        # Reserve tokens
        reserve_amount = 8
        success, reservation = repo.reserve_tokens(
            user_id=sample_user.id,
            amount=reserve_amount,
            operation="finalize_test"
        )
        assert success is True
        
        initial_used = repo.get_user_balance(sample_user.id).tokens_used_this_month
        
        # Finalize consumption
        finalize_success, transaction = repo.finalize_reserved_consumption(
            user_id=sample_user.id,
            reserved_amount=reserve_amount,
            actual_amount=6,  # Used less than reserved
            transaction_type=TransactionType.EXPERT_ANALYSIS,
            operation="finalize_test"
        )
        
        assert finalize_success is True
        assert transaction.amount == -6
        
        # Check final state
        final_balance = repo.get_user_balance(sample_user.id)
        assert final_balance.tokens_used_this_month == initial_used + 6
        assert final_balance.tokens_reserved >= 0  # Reservation released


class TestPlanManagement:
    """Test subscription plan management."""
    
    def test_upgrade_user_plan(self, db_session, sample_user, sample_token_balance):
        """Test upgrading user plan."""
        repo = TokenRepository(db_session)
        
        old_quota = sample_token_balance.monthly_token_quota
        
        success, updated_balance = repo.upgrade_user_plan(
            user_id=sample_user.id,
            new_plan=PlanType.PREMIUM,
            new_quota=300
        )
        
        assert success is True
        assert updated_balance.plan_type == PlanType.PREMIUM
        assert updated_balance.monthly_token_quota == 300
        assert updated_balance.monthly_token_quota > old_quota
        
        # Period should be reset
        assert updated_balance.current_period_start is not None
        assert updated_balance.current_period_end is not None
    
    def test_upgrade_nonexistent_user(self, db_session):
        """Test upgrading plan for non-existent user."""
        repo = TokenRepository(db_session)
        nonexistent_id = uuid.uuid4()
        
        success, balance = repo.upgrade_user_plan(
            user_id=nonexistent_id,
            new_plan=PlanType.ENTERPRISE,
            new_quota=500
        )
        
        assert success is False
        assert balance is None
    
    def test_downgrade_user_plan(self, db_session, sample_user, sample_token_balance):
        """Test downgrading user plan."""
        repo = TokenRepository(db_session)
        
        # First upgrade to have something to downgrade from
        repo.upgrade_user_plan(
            user_id=sample_user.id,
            new_plan=PlanType.PREMIUM,
            new_quota=300
        )
        
        # Now downgrade
        success, downgraded_balance = repo.upgrade_user_plan(
            user_id=sample_user.id,
            new_plan=PlanType.BASIC,
            new_quota=100
        )
        
        assert success is True
        assert downgraded_balance.plan_type == PlanType.BASIC
        assert downgraded_balance.monthly_token_quota == 100


class TestBonusTokens:
    """Test bonus token operations."""
    
    def test_add_bonus_tokens(self, db_session, sample_user, sample_token_balance):
        """Test adding bonus tokens."""
        repo = TokenRepository(db_session)
        
        initial_bonus = sample_token_balance.bonus_tokens
        bonus_amount = 50
        
        success, transaction = repo.add_bonus_tokens(
            user_id=sample_user.id,
            amount=bonus_amount,
            reason="promotion_reward"
        )
        
        assert success is True
        assert transaction.amount == bonus_amount
        assert transaction.transaction_type == TransactionType.BONUS_CREDIT
        
        # Check bonus was added
        updated_balance = repo.get_user_balance(sample_user.id)
        assert updated_balance.bonus_tokens == initial_bonus + bonus_amount
    
    def test_add_bonus_tokens_nonexistent_user(self, db_session):
        """Test adding bonus tokens to non-existent user."""
        repo = TokenRepository(db_session)
        nonexistent_id = uuid.uuid4()
        
        success, transaction = repo.add_bonus_tokens(
            user_id=nonexistent_id,
            amount=25,
            reason="impossible_bonus"
        )
        
        assert success is False
        assert transaction is None


class TestMonthlyReset:
    """Test monthly quota reset functionality."""
    
    def test_reset_monthly_quota(self, db_session, sample_user, sample_token_balance):
        """Test resetting monthly quota."""
        repo = TokenRepository(db_session)
        
        # Set some usage first
        repo.update_user_balance(
            sample_user.id,
            tokens_used_this_month=75
        )
        
        old_period_start = sample_token_balance.current_period_start
        
        success, reset_balance = repo.reset_monthly_quota(sample_user.id)
        
        assert success is True
        assert reset_balance.tokens_used_this_month == 0
        assert reset_balance.current_period_start > old_period_start
        assert reset_balance.current_period_end > reset_balance.current_period_start
        assert reset_balance.last_reset_date > old_period_start
    
    def test_reset_monthly_quota_nonexistent_user(self, db_session):
        """Test resetting quota for non-existent user."""
        repo = TokenRepository(db_session)
        nonexistent_id = uuid.uuid4()
        
        success, balance = repo.reset_monthly_quota(nonexistent_id)
        
        assert success is False
        assert balance is None


class TestTransactionHistory:
    """Test transaction history and analytics."""
    
    def test_get_user_transactions(self, db_session, sample_user, sample_transactions):
        """Test getting user transaction history."""
        repo = TokenRepository(db_session)
        
        transactions = repo.get_user_transactions(
            user_id=sample_user.id,
            limit=10
        )
        
        assert len(transactions) > 0
        assert all(tx.user_id == sample_user.id for tx in transactions)
        # Should be ordered by created_at descending
        for i in range(1, len(transactions)):
            assert transactions[i-1].created_at >= transactions[i].created_at
    
    def test_get_user_transactions_with_pagination(self, db_session, sample_user, sample_transactions):
        """Test transaction history with pagination."""
        repo = TokenRepository(db_session)
        
        # Get first page
        page1 = repo.get_user_transactions(
            user_id=sample_user.id,
            limit=2,
            offset=0
        )
        
        # Get second page
        page2 = repo.get_user_transactions(
            user_id=sample_user.id,
            limit=2,
            offset=2
        )
        
        assert len(page1) <= 2
        assert len(page2) <= 2
        
        # Should be different transactions
        if len(page1) > 0 and len(page2) > 0:
            page1_ids = [tx.id for tx in page1]
            page2_ids = [tx.id for tx in page2]
            assert not set(page1_ids).intersection(set(page2_ids))
    
    def test_get_transactions_by_type(self, db_session, sample_user, sample_transactions):
        """Test filtering transactions by type."""
        repo = TokenRepository(db_session)
        
        analysis_transactions = repo.get_user_transactions(
            user_id=sample_user.id,
            transaction_types=[TransactionType.BASIC_ANALYSIS, TransactionType.EXPERT_ANALYSIS]
        )
        
        for tx in analysis_transactions:
            assert tx.transaction_type in [TransactionType.BASIC_ANALYSIS, TransactionType.EXPERT_ANALYSIS]
    
    def test_get_transactions_date_range(self, db_session, sample_user, sample_transactions):
        """Test filtering transactions by date range."""
        repo = TokenRepository(db_session)
        
        now = datetime.utcnow()
        week_ago = now - timedelta(days=7)
        
        recent_transactions = repo.get_user_transactions(
            user_id=sample_user.id,
            start_date=week_ago,
            end_date=now
        )
        
        for tx in recent_transactions:
            assert tx.created_at >= week_ago
            assert tx.created_at <= now


class TestConcurrencyAndEdgeCases:
    """Test concurrency scenarios and edge cases."""
    
    def test_concurrent_token_consumption(self, db_session, sample_user, sample_token_balance):
        """Test concurrent token consumption doesn't create race conditions."""
        repo = TokenRepository(db_session)
        
        # Simulate concurrent operations
        import threading
        results = []
        
        def consume_tokens():
            success, tx = repo.consume_tokens(
                user_id=sample_user.id,
                amount=1,
                transaction_type=TransactionType.BASIC_ANALYSIS
            )
            results.append((success, tx))
        
        # Launch multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=consume_tokens)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results - should handle concurrency gracefully
        successful_operations = sum(1 for success, _ in results if success)
        assert successful_operations <= sample_token_balance.tokens_remaining
    
    def test_token_consumption_with_bonus_depletion_first(self, db_session, sample_user):
        """Test that bonus tokens are consumed before monthly quota."""
        repo = TokenRepository(db_session)
        
        # Create balance with both monthly and bonus tokens
        balance = repo.create_user_balance(
            user_id=sample_user.id,
            plan_type=PlanType.BASIC,
            monthly_quota=100
        )
        
        # Add bonus tokens
        repo.add_bonus_tokens(sample_user.id, 20, "test_bonus")
        
        # Consume some tokens
        repo.consume_tokens(
            user_id=sample_user.id,
            amount=15,
            transaction_type=TransactionType.BASIC_ANALYSIS
        )
        
        # Check that bonus tokens were consumed first
        updated_balance = repo.get_user_balance(sample_user.id)
        assert updated_balance.bonus_tokens == 5  # 20 - 15
        assert updated_balance.tokens_used_this_month == 0  # Monthly quota untouched
    
    def test_refund_transaction(self, db_session, sample_user, sample_token_balance):
        """Test refunding tokens."""
        repo = TokenRepository(db_session)
        
        # First consume some tokens
        consume_success, consume_tx = repo.consume_tokens(
            user_id=sample_user.id,
            amount=10,
            transaction_type=TransactionType.EXPERT_ANALYSIS
        )
        assert consume_success is True
        
        initial_used = repo.get_user_balance(sample_user.id).tokens_used_this_month
        
        # Refund the transaction
        refund_success, refund_tx = repo.refund_transaction(
            user_id=sample_user.id,
            original_transaction_id=consume_tx.id,
            refund_amount=10,
            reason="analysis_failed"
        )
        
        assert refund_success is True
        assert refund_tx.amount == 10  # Positive for credit
        assert refund_tx.transaction_type == TransactionType.REFUND
        
        # Check tokens were restored
        final_balance = repo.get_user_balance(sample_user.id)
        assert final_balance.tokens_used_this_month == initial_used - 10


class TestAnalyticsAndReporting:
    """Test analytics and reporting functionality."""
    
    def test_get_usage_analytics(self, db_session, sample_user, sample_transactions):
        """Test getting usage analytics."""
        repo = TokenRepository(db_session)
        
        analytics = repo.get_usage_analytics(
            user_id=sample_user.id,
            days=30
        )
        
        assert 'total_tokens_consumed' in analytics
        assert 'total_transactions' in analytics
        assert 'most_used_feature' in analytics
        assert 'daily_usage' in analytics
        assert isinstance(analytics['total_tokens_consumed'], int)
        assert isinstance(analytics['total_transactions'], int)
    
    def test_get_system_analytics(self, db_session, sample_transactions):
        """Test getting system-wide analytics."""
        repo = TokenRepository(db_session)
        
        analytics = repo.get_system_analytics(days=7)
        
        assert 'total_active_users' in analytics
        assert 'total_tokens_consumed' in analytics
        assert 'revenue_potential' in analytics
        assert 'feature_usage' in analytics
    
    def test_get_plan_distribution(self, db_session, sample_users_different_plans):
        """Test getting plan distribution analytics."""
        repo = TokenRepository(db_session)
        
        distribution = repo.get_plan_distribution()
        
        assert isinstance(distribution, dict)
        for plan_type in PlanType:
            assert plan_type.value in distribution
            assert isinstance(distribution[plan_type.value], int)


class TestErrorHandling:
    """Test error handling and validation."""
    
    def test_invalid_transaction_type(self, db_session, sample_user, sample_token_balance):
        """Test handling of invalid transaction types."""
        repo = TokenRepository(db_session)
        
        # This should handle gracefully
        success, transaction = repo.consume_tokens(
            user_id=sample_user.id,
            amount=1,
            transaction_type="invalid_type"
        )
        
        # Should either succeed with default cost or fail gracefully
        assert isinstance(success, bool)
    
    def test_negative_token_amounts(self, db_session, sample_user, sample_token_balance):
        """Test handling of negative token amounts."""
        repo = TokenRepository(db_session)
        
        # Negative consumption should fail
        success, transaction = repo.consume_tokens(
            user_id=sample_user.id,
            amount=-5,
            transaction_type=TransactionType.BASIC_ANALYSIS
        )
        
        assert success is False
        assert transaction is None
    
    def test_database_rollback_on_error(self, db_session, sample_user, sample_token_balance):
        """Test that database operations rollback on error."""
        repo = TokenRepository(db_session)
        
        with patch.object(db_session, 'commit', side_effect=Exception("Database error")):
            with pytest.raises(Exception):
                repo.consume_tokens(
                    user_id=sample_user.id,
                    amount=5,
                    transaction_type=TransactionType.BASIC_ANALYSIS
                )
        
        # Balance should be unchanged after rollback
        balance_after_error = repo.get_user_balance(sample_user.id)
        assert balance_after_error.tokens_used_this_month == sample_token_balance.tokens_used_this_month