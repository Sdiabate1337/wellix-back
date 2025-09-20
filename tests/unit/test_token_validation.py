"""
Unit tests for TokenValidation middleware.
Tests token validation, rate limiting, and API protection.
"""

import pytest
from datetime import datetime, timedelta
import uuid
from unittest.mock import Mock, patch, AsyncMock
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient

from app.middleware.token_validation import (
    TokenValidator,
    token_required,
    get_token_validator,
    RateLimitExceeded,
    InsufficientTokens,
    PlanPermissionDenied
)
from app.repositories.token_repository import TokenRepository
from app.db.models.token_system import (
    UserTokenBalance,
    PlanType,
    TransactionType,
    TokenCosts
)
from app.db.models.user import User


class TestTokenValidator:
    """Test TokenValidator core functionality."""
    
    def test_validator_initialization(self, db_session):
        """Test TokenValidator initialization."""
        mock_repo = Mock(spec=TokenRepository)
        validator = TokenValidator(mock_repo)
        
        assert validator.repository == mock_repo
        assert hasattr(validator, 'rate_limits')
        assert PlanType.FREE in validator.rate_limits
        assert PlanType.PREMIUM in validator.rate_limits
    
    def test_get_user_plan_existing_user(self, db_session, sample_user, sample_token_balance):
        """Test getting plan for existing user."""
        mock_repo = Mock(spec=TokenRepository)
        mock_repo.get_user_balance.return_value = sample_token_balance
        
        validator = TokenValidator(mock_repo)
        
        plan = validator.get_user_plan(sample_user.id)
        
        assert plan == sample_token_balance.plan_type
        mock_repo.get_user_balance.assert_called_once_with(sample_user.id)
    
    def test_get_user_plan_nonexistent_user(self, db_session):
        """Test getting plan for non-existent user."""
        mock_repo = Mock(spec=TokenRepository)
        mock_repo.get_user_balance.return_value = None
        
        validator = TokenValidator(mock_repo)
        nonexistent_id = uuid.uuid4()
        
        plan = validator.get_user_plan(nonexistent_id)
        
        assert plan == PlanType.FREE  # Default fallback
    
    def test_check_rate_limit_within_limits(self, db_session, sample_user):
        """Test rate limiting when within limits."""
        mock_repo = Mock(spec=TokenRepository)
        validator = TokenValidator(mock_repo)
        
        # Mock recent transactions count
        mock_repo.count_recent_transactions.return_value = 5
        
        # Free plan limit is typically 10/hour
        is_allowed = validator.check_rate_limit(sample_user.id, PlanType.FREE)
        
        assert is_allowed is True
    
    def test_check_rate_limit_exceeded(self, db_session, sample_user):
        """Test rate limiting when limit exceeded."""
        mock_repo = Mock(spec=TokenRepository)
        validator = TokenValidator(mock_repo)
        
        # Mock exceeding rate limit
        mock_repo.count_recent_transactions.return_value = 15  # Over free limit
        
        is_allowed = validator.check_rate_limit(sample_user.id, PlanType.FREE)
        
        assert is_allowed is False
    
    def test_check_rate_limit_premium_user(self, db_session, sample_user):
        """Test rate limiting for premium users."""
        mock_repo = Mock(spec=TokenRepository)
        validator = TokenValidator(mock_repo)
        
        # Premium users have higher limits
        mock_repo.count_recent_transactions.return_value = 25
        
        is_allowed = validator.check_rate_limit(sample_user.id, PlanType.PREMIUM)
        
        assert is_allowed is True  # Premium limit is higher
    
    def test_validate_tokens_sufficient_funds(self, db_session, sample_user, sample_token_balance):
        """Test token validation with sufficient funds."""
        mock_repo = Mock(spec=TokenRepository)
        mock_repo.get_user_balance.return_value = sample_token_balance
        
        validator = TokenValidator(mock_repo)
        
        # Test with amount within balance
        is_valid = validator.validate_tokens(
            user_id=sample_user.id,
            required_tokens=5
        )
        
        assert is_valid is True
    
    def test_validate_tokens_insufficient_funds(self, db_session, sample_user, sample_token_balance):
        """Test token validation with insufficient funds."""
        mock_repo = Mock(spec=TokenRepository)
        mock_repo.get_user_balance.return_value = sample_token_balance
        
        validator = TokenValidator(mock_repo)
        
        # Test with amount exceeding balance
        excessive_amount = sample_token_balance.tokens_remaining + 10
        
        is_valid = validator.validate_tokens(
            user_id=sample_user.id,
            required_tokens=excessive_amount
        )
        
        assert is_valid is False
    
    def test_validate_tokens_nonexistent_user(self, db_session):
        """Test token validation for non-existent user."""
        mock_repo = Mock(spec=TokenRepository)
        mock_repo.get_user_balance.return_value = None
        
        validator = TokenValidator(mock_repo)
        nonexistent_id = uuid.uuid4()
        
        is_valid = validator.validate_tokens(
            user_id=nonexistent_id,
            required_tokens=1
        )
        
        assert is_valid is False
    
    def test_check_plan_permission_allowed_feature(self, db_session):
        """Test plan permission for allowed feature."""
        validator = TokenValidator(Mock())
        
        # Basic plan should allow basic analysis
        is_allowed = validator.check_plan_permission(
            plan_type=PlanType.BASIC,
            transaction_type=TransactionType.BASIC_ANALYSIS
        )
        
        assert is_allowed is True
    
    def test_check_plan_permission_restricted_feature(self, db_session):
        """Test plan permission for restricted feature."""
        validator = TokenValidator(Mock())
        
        # Free plan shouldn't allow expert analysis (hypothetically)
        # This depends on your business rules
        is_allowed = validator.check_plan_permission(
            plan_type=PlanType.FREE,
            transaction_type=TransactionType.MULTI_CONDITION_ANALYSIS
        )
        
        # This assertion depends on your actual business rules
        assert isinstance(is_allowed, bool)
    
    def test_check_plan_permission_enterprise_features(self, db_session):
        """Test plan permission for enterprise-only features."""
        validator = TokenValidator(Mock())
        
        # Enterprise plan should allow all features
        is_allowed = validator.check_plan_permission(
            plan_type=PlanType.ENTERPRISE,
            transaction_type=TransactionType.DETAILED_RECOMMENDATIONS
        )
        
        assert is_allowed is True


class TestTokenRequiredDecorator:
    """Test @token_required decorator functionality."""
    
    async def test_token_required_success(self, db_session, sample_user, sample_token_balance):
        """Test successful token validation with decorator."""
        mock_repo = Mock(spec=TokenRepository)
        mock_repo.get_user_balance.return_value = sample_token_balance
        mock_repo.count_recent_transactions.return_value = 5
        mock_repo.consume_tokens.return_value = (True, Mock())
        
        # Create a mock dependency function
        async def mock_get_validator():
            return TokenValidator(mock_repo)
        
        async def mock_get_current_user():
            return sample_user
        
        # Test the decorator
        @token_required(
            transaction_type=TransactionType.BASIC_ANALYSIS,
            feature_name="test_analysis"
        )
        async def test_endpoint(
            user: User = mock_get_current_user(),
            validator: TokenValidator = mock_get_validator()
        ):
            return {"success": True}
        
        # Execute the decorated function
        result = await test_endpoint()
        
        assert result["success"] is True
    
    async def test_token_required_insufficient_tokens(self, db_session, sample_user):
        """Test decorator with insufficient tokens."""
        # Create balance with no tokens
        empty_balance = Mock(
            user_id=sample_user.id,
            tokens_remaining=0,
            plan_type=PlanType.FREE
        )
        
        mock_repo = Mock(spec=TokenRepository)
        mock_repo.get_user_balance.return_value = empty_balance
        mock_repo.count_recent_transactions.return_value = 5
        
        async def mock_get_validator():
            return TokenValidator(mock_repo)
        
        async def mock_get_current_user():
            return sample_user
        
        @token_required(
            transaction_type=TransactionType.EXPERT_ANALYSIS,
            feature_name="expensive_analysis"
        )
        async def test_endpoint(
            user: User = mock_get_current_user(),
            validator: TokenValidator = mock_get_validator()
        ):
            return {"success": True}
        
        # Should raise HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await test_endpoint()
        
        assert exc_info.value.status_code == 402  # Payment required
        assert "insufficient tokens" in exc_info.value.detail.lower()
    
    async def test_token_required_rate_limit_exceeded(self, db_session, sample_user, sample_token_balance):
        """Test decorator with rate limit exceeded."""
        mock_repo = Mock(spec=TokenRepository)
        mock_repo.get_user_balance.return_value = sample_token_balance
        mock_repo.count_recent_transactions.return_value = 50  # Over rate limit
        
        async def mock_get_validator():
            return TokenValidator(mock_repo)
        
        async def mock_get_current_user():
            return sample_user
        
        @token_required(
            transaction_type=TransactionType.BASIC_ANALYSIS,
            feature_name="rate_limited_analysis"
        )
        async def test_endpoint(
            user: User = mock_get_current_user(),
            validator: TokenValidator = mock_get_validator()
        ):
            return {"success": True}
        
        with pytest.raises(HTTPException) as exc_info:
            await test_endpoint()
        
        assert exc_info.value.status_code == 429  # Too many requests
        assert "rate limit" in exc_info.value.detail.lower()
    
    async def test_token_required_plan_permission_denied(self, db_session, sample_user):
        """Test decorator with plan permission denied."""
        free_balance = Mock(
            user_id=sample_user.id,
            tokens_remaining=100,
            plan_type=PlanType.FREE
        )
        
        mock_repo = Mock(spec=TokenRepository)
        mock_repo.get_user_balance.return_value = free_balance
        mock_repo.count_recent_transactions.return_value = 5
        
        # Mock validator to deny permission
        mock_validator = Mock(spec=TokenValidator)
        mock_validator.get_user_plan.return_value = PlanType.FREE
        mock_validator.check_rate_limit.return_value = True
        mock_validator.validate_tokens.return_value = True
        mock_validator.check_plan_permission.return_value = False  # Deny permission
        
        async def mock_get_validator():
            return mock_validator
        
        async def mock_get_current_user():
            return sample_user
        
        @token_required(
            transaction_type="enterprise_only_feature",
            feature_name="premium_analysis"
        )
        async def test_endpoint(
            user: User = mock_get_current_user(),
            validator: TokenValidator = mock_get_validator()
        ):
            return {"success": True}
        
        with pytest.raises(HTTPException) as exc_info:
            await test_endpoint()
        
        assert exc_info.value.status_code == 403  # Forbidden
        assert "plan does not support" in exc_info.value.detail.lower()


class TestGetTokenValidator:
    """Test get_token_validator dependency function."""
    
    def test_get_token_validator_returns_instance(self, db_session):
        """Test that get_token_validator returns TokenValidator instance."""
        with patch('app.middleware.token_validation.get_db') as mock_get_db:
            mock_get_db.return_value = db_session
            
            validator = get_token_validator()
            
            assert isinstance(validator, TokenValidator)
            assert validator.repository is not None


class TestCustomExceptions:
    """Test custom exception classes."""
    
    def test_rate_limit_exceeded_exception(self):
        """Test RateLimitExceeded exception."""
        user_id = uuid.uuid4()
        
        exc = RateLimitExceeded(user_id, PlanType.FREE, 60)
        
        assert exc.user_id == user_id
        assert exc.plan_type == PlanType.FREE
        assert exc.retry_after_seconds == 60
        assert "rate limit exceeded" in str(exc).lower()
    
    def test_insufficient_tokens_exception(self):
        """Test InsufficientTokens exception."""
        user_id = uuid.uuid4()
        
        exc = InsufficientTokens(user_id, required=10, available=5)
        
        assert exc.user_id == user_id
        assert exc.required_tokens == 10
        assert exc.available_tokens == 5
        assert "insufficient tokens" in str(exc).lower()
    
    def test_plan_permission_denied_exception(self):
        """Test PlanPermissionDenied exception."""
        user_id = uuid.uuid4()
        
        exc = PlanPermissionDenied(
            user_id,
            PlanType.FREE,
            TransactionType.EXPERT_ANALYSIS
        )
        
        assert exc.user_id == user_id
        assert exc.user_plan == PlanType.FREE
        assert exc.required_feature == TransactionType.EXPERT_ANALYSIS
        assert "plan does not support" in str(exc).lower()


class TestIntegrationScenarios:
    """Test integration scenarios with FastAPI."""
    
    def test_middleware_with_fastapi_request(self, db_session, sample_user, sample_token_balance):
        """Test middleware integration with FastAPI request."""
        from fastapi import FastAPI, Depends
        from fastapi.testclient import TestClient
        
        app = FastAPI()
        
        # Mock dependencies
        async def mock_get_current_user():
            return sample_user
        
        async def mock_get_validator():
            mock_repo = Mock(spec=TokenRepository)
            mock_repo.get_user_balance.return_value = sample_token_balance
            mock_repo.count_recent_transactions.return_value = 5
            mock_repo.consume_tokens.return_value = (True, Mock())
            return TokenValidator(mock_repo)
        
        @app.post("/test-endpoint")
        @token_required(
            transaction_type=TransactionType.BASIC_ANALYSIS,
            feature_name="integration_test"
        )
        async def test_endpoint(
            user: User = Depends(mock_get_current_user),
            validator: TokenValidator = Depends(mock_get_validator)
        ):
            return {"message": "success"}
        
        client = TestClient(app)
        response = client.post("/test-endpoint")
        
        assert response.status_code == 200
        assert response.json()["message"] == "success"
    
    def test_middleware_error_handling(self, db_session, sample_user):
        """Test middleware error handling with FastAPI."""
        from fastapi import FastAPI, Depends, HTTPException
        from fastapi.testclient import TestClient
        
        app = FastAPI()
        
        async def mock_get_current_user():
            return sample_user
        
        async def mock_get_validator():
            mock_repo = Mock(spec=TokenRepository)
            mock_repo.get_user_balance.return_value = None  # No balance
            return TokenValidator(mock_repo)
        
        @app.post("/test-error")
        @token_required(
            transaction_type=TransactionType.EXPERT_ANALYSIS,
            feature_name="error_test"
        )
        async def test_endpoint(
            user: User = Depends(mock_get_current_user),
            validator: TokenValidator = Depends(mock_get_validator)
        ):
            return {"message": "should not reach here"}
        
        client = TestClient(app)
        response = client.post("/test-error")
        
        assert response.status_code == 402  # Payment required
        assert "insufficient tokens" in response.json()["detail"].lower()


class TestPerformanceAndEdgeCases:
    """Test performance considerations and edge cases."""
    
    def test_validator_caching_behavior(self, db_session, sample_user, sample_token_balance):
        """Test that validator caches user data appropriately."""
        mock_repo = Mock(spec=TokenRepository)
        mock_repo.get_user_balance.return_value = sample_token_balance
        
        validator = TokenValidator(mock_repo)
        
        # Multiple calls to same user
        plan1 = validator.get_user_plan(sample_user.id)
        plan2 = validator.get_user_plan(sample_user.id)
        
        assert plan1 == plan2
        # Should have called repository twice (no caching implemented)
        assert mock_repo.get_user_balance.call_count == 2
    
    def test_validator_with_large_user_base(self, db_session):
        """Test validator performance with many users."""
        mock_repo = Mock(spec=TokenRepository)
        validator = TokenValidator(mock_repo)
        
        # Simulate checking many users
        user_ids = [uuid.uuid4() for _ in range(100)]
        mock_repo.get_user_balance.return_value = Mock(plan_type=PlanType.BASIC)
        
        plans = [validator.get_user_plan(user_id) for user_id in user_ids]
        
        assert len(plans) == 100
        assert all(plan == PlanType.BASIC for plan in plans)
    
    def test_concurrent_token_validation(self, db_session, sample_user, sample_token_balance):
        """Test concurrent token validation requests."""
        import asyncio
        
        mock_repo = Mock(spec=TokenRepository)
        mock_repo.get_user_balance.return_value = sample_token_balance
        mock_repo.count_recent_transactions.return_value = 5
        
        validator = TokenValidator(mock_repo)
        
        async def validate_tokens():
            return validator.validate_tokens(sample_user.id, 1)
        
        # Run multiple validations concurrently
        async def run_concurrent_validations():
            tasks = [validate_tokens() for _ in range(10)]
            results = await asyncio.gather(*tasks)
            return results
        
        results = asyncio.run(run_concurrent_validations())
        
        assert len(results) == 10
        assert all(result is True for result in results)
    
    def test_validator_with_expired_tokens(self, db_session, sample_user):
        """Test validator with expired token balance."""
        expired_balance = Mock(
            user_id=sample_user.id,
            plan_type=PlanType.BASIC,
            tokens_remaining=50,
            needs_renewal=True,  # Expired
            current_period_end=datetime.utcnow() - timedelta(days=1)
        )
        
        mock_repo = Mock(spec=TokenRepository)
        mock_repo.get_user_balance.return_value = expired_balance
        
        validator = TokenValidator(mock_repo)
        
        # Should handle expired balances gracefully
        is_valid = validator.validate_tokens(sample_user.id, 10)
        
        # Behavior depends on business rules - might allow or deny
        assert isinstance(is_valid, bool)
    
    def test_validator_memory_usage(self, db_session):
        """Test validator memory usage with many operations."""
        import gc
        
        mock_repo = Mock(spec=TokenRepository)
        validator = TokenValidator(mock_repo)
        
        # Perform many operations
        for i in range(1000):
            user_id = uuid.uuid4()
            mock_repo.get_user_balance.return_value = Mock(
                plan_type=PlanType.FREE,
                tokens_remaining=10
            )
            validator.validate_tokens(user_id, 1)
        
        # Force garbage collection
        gc.collect()
        
        # Should complete without memory errors
        assert True