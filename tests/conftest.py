"""
Test configuration and fixtures for comprehensive testing.
Uses PostgreSQL for database compatibility.
"""

import pytest
import asyncio
import os
import uuid
from datetime import datetime, timedelta
from typing import Generator
from unittest.mock import Mock

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from app.db.database import Base
from app.db.models.user import User
from app.db.models.token_system import (
    UserTokenBalance,
    TokenTransaction,
    SubscriptionPlan,
    PlanType,
    TransactionType
)
from app.repositories.token_repository import TokenRepository
from app.services.token.token_service import TokenService
from app.core.dependencies import get_db

# Test database configuration
TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL", 
    "postgresql://wellix_test_user:wellix_test_password@localhost:5433/wellix_test"
)

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Generator, AsyncGenerator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import uuid

from app.db.database import Base
from app.db.models.user import User
from app.db.models.token_system import (
    UserTokenBalance, 
    TokenTransaction, 
    SubscriptionPlan,
    PlanType, 
    TransactionType,
    DEFAULT_PLANS
)
from app.repositories.token_repository import TokenRepository
from app.services.token.token_service import TokenService


# Test database setup
SQLALCHEMY_TEST_DATABASE_URL = "sqlite:///:memory:"

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
def db_session():
    """Create a fresh database session for each test with PostgreSQL."""
    engine = create_engine(
        TEST_DATABASE_URL,
        echo=False,
        pool_pre_ping=True,
    )
    
    # Create all tables
    try:
        Base.metadata.create_all(bind=engine)
    except SQLAlchemyError as e:
        pytest.skip(f"PostgreSQL test database not available: {e}")
    
    # Create session
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    
    try:
        yield session
    finally:
        session.close()
        # Clean up tables after test
        try:
            Base.metadata.drop_all(bind=engine)
        except SQLAlchemyError:
            pass
        engine.dispose()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
async def token_repository(db_session):
    """Create TokenRepository instance with test database."""
    return TokenRepository(db_session)


@pytest.fixture  
async def token_service(token_repository):
    """Create TokenService instance with test repository."""
    return TokenService(token_repository)


@pytest.fixture
async def sample_user(db_session) -> User:
    """Create a sample user for testing."""
    user = User(
        id=uuid.uuid4(),
        email="test@wellix.com",
        username="testuser",
        hashed_password="hashed_password_123",
        first_name="Test",
        last_name="User",
        is_active=True,
        is_verified=True,
        is_premium=False
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
async def premium_user(db_session) -> User:
    """Create a premium user for testing."""
    user = User(
        id=uuid.uuid4(),
        email="premium@wellix.com", 
        username="premiumuser",
        hashed_password="hashed_password_123",
        first_name="Premium",
        last_name="User",
        is_active=True,
        is_verified=True,
        is_premium=True
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
async def subscription_plans(db_session) -> list[SubscriptionPlan]:
    """Create all default subscription plans."""
    plans = []
    for plan_data in DEFAULT_PLANS:
        plan = SubscriptionPlan(**plan_data)
        db_session.add(plan)
        plans.append(plan)
    
    db_session.commit()
    for plan in plans:
        db_session.refresh(plan)
    
    return plans


@pytest.fixture
async def user_with_balance(db_session, sample_user, subscription_plans) -> tuple[User, UserTokenBalance]:
    """Create user with initialized token balance."""
    now = datetime.utcnow()
    balance = UserTokenBalance(
        user_id=sample_user.id,
        plan_type=PlanType.FREE,
        plan_started_at=now,
        monthly_token_quota=20,
        tokens_used_this_month=5,
        bonus_tokens=10,
        current_period_start=now,
        current_period_end=now + timedelta(days=30),
        last_reset_date=now
    )
    db_session.add(balance)
    db_session.commit()
    db_session.refresh(balance)
    
    return sample_user, balance


@pytest.fixture
async def user_with_premium_balance(db_session, premium_user, subscription_plans) -> tuple[User, UserTokenBalance]:
    """Create premium user with premium balance."""
    now = datetime.utcnow()
    balance = UserTokenBalance(
        user_id=premium_user.id,
        plan_type=PlanType.PREMIUM,
        plan_started_at=now,
        monthly_token_quota=500,
        tokens_used_this_month=100,
        bonus_tokens=50,
        current_period_start=now,
        current_period_end=now + timedelta(days=30),
        last_reset_date=now
    )
    db_session.add(balance)
    db_session.commit()
    db_session.refresh(balance)
    
    return premium_user, balance


@pytest.fixture
async def user_with_expired_period(db_session, sample_user, subscription_plans) -> tuple[User, UserTokenBalance]:
    """Create user with expired quota period (needs reset)."""
    now = datetime.utcnow()
    expired_date = now - timedelta(days=35)  # 35 days ago
    
    balance = UserTokenBalance(
        user_id=sample_user.id,
        plan_type=PlanType.BASIC,
        plan_started_at=expired_date,
        monthly_token_quota=100,
        tokens_used_this_month=95,  # Almost used up
        bonus_tokens=0,
        current_period_start=expired_date,
        current_period_end=expired_date + timedelta(days=30),  # Expired
        last_reset_date=expired_date
    )
    db_session.add(balance)
    db_session.commit()
    db_session.refresh(balance)
    
    return sample_user, balance


@pytest.fixture
async def sample_transactions(db_session, user_with_balance) -> list[TokenTransaction]:
    """Create sample token transactions for testing."""
    user, balance = user_with_balance
    
    transactions = [
        # Consumption transactions
        TokenTransaction(
            user_id=user.id,
            amount=-5,
            transaction_type=TransactionType.EXPERT_ANALYSIS,
            feature_used="nutrition_analysis",
            plan_type_at_time=PlanType.FREE,
            processing_time_ms=1500,
            llm_model_used="gpt-4o-mini",
            created_at=datetime.utcnow() - timedelta(hours=2)
        ),
        TokenTransaction(
            user_id=user.id,
            amount=-2,
            transaction_type=TransactionType.IMAGE_OCR_PROCESSING,
            feature_used="ocr_scan",
            plan_type_at_time=PlanType.FREE,
            processing_time_ms=800,
            created_at=datetime.utcnow() - timedelta(hours=1)
        ),
        
        # Credit transaction
        TokenTransaction(
            user_id=user.id,
            amount=10,
            transaction_type=TransactionType.BONUS_CREDIT,
            feature_used="signup_bonus",
            plan_type_at_time=PlanType.FREE,
            created_at=datetime.utcnow() - timedelta(days=1)
        )
    ]
    
    for transaction in transactions:
        db_session.add(transaction)
    
    db_session.commit()
    for transaction in transactions:
        db_session.refresh(transaction)
    
    return transactions


# Test data factories

class TokenTestData:
    """Factory for creating test data."""
    
    @staticmethod
    def create_user_data(**overrides) -> dict:
        """Create user data for testing."""
        default_data = {
            "email": f"test_{uuid.uuid4().hex[:8]}@wellix.com",
            "username": f"user_{uuid.uuid4().hex[:8]}",
            "hashed_password": "hashed_password_123",
            "first_name": "Test",
            "last_name": "User",
            "is_active": True,
            "is_verified": True,
            "is_premium": False
        }
        default_data.update(overrides)
        return default_data
    
    @staticmethod
    def create_balance_data(user_id: uuid.UUID, **overrides) -> dict:
        """Create token balance data for testing."""
        now = datetime.utcnow()
        default_data = {
            "user_id": user_id,
            "plan_type": PlanType.FREE,
            "plan_started_at": now,
            "monthly_token_quota": 20,
            "tokens_used_this_month": 0,
            "bonus_tokens": 0,
            "current_period_start": now,
            "current_period_end": now + timedelta(days=30),
            "last_reset_date": now
        }
        default_data.update(overrides)
        return default_data
    
    @staticmethod
    def create_transaction_data(user_id: uuid.UUID, **overrides) -> dict:
        """Create transaction data for testing."""
        default_data = {
            "user_id": user_id,
            "amount": -1,
            "transaction_type": TransactionType.BASIC_ANALYSIS,
            "feature_used": "test_feature",
            "plan_type_at_time": PlanType.FREE,
            "created_at": datetime.utcnow()
        }
        default_data.update(overrides)
        return default_data


# Mock payment service for testing

class MockPaymentService:
    """Mock payment service for testing payment flows."""
    
    def __init__(self, success_rate: float = 1.0):
        self.success_rate = success_rate
        self.payments = []
    
    async def process_payment(self, amount: float, payment_method_id: str = None) -> dict:
        """Mock payment processing."""
        payment_id = f"pay_test_{uuid.uuid4().hex[:16]}"
        
        success = True  # For testing, default to success
        
        payment_result = {
            "payment_id": payment_id,
            "amount": amount,
            "success": success,
            "currency": "EUR"
        }
        
        if not success:
            payment_result.update({
                "error": "payment_declined",
                "message": "Mock payment decline for testing"
            })
        
        self.payments.append(payment_result)
        return payment_result
    
    def get_payment_history(self) -> list:
        """Get all mock payments."""
        return self.payments
    
    def set_success_rate(self, rate: float):
        """Set payment success rate for testing failures."""
        self.success_rate = rate


# Utility functions for tests

def assert_balance_consistent(balance: UserTokenBalance):
    """Assert that balance calculations are consistent."""
    calculated_remaining = max(0, balance.monthly_token_quota - balance.tokens_used_this_month) + balance.bonus_tokens
    assert balance.tokens_remaining == calculated_remaining, "Balance calculation inconsistent"


def assert_transaction_valid(transaction: TokenTransaction):
    """Assert that transaction data is valid."""
    assert transaction.user_id is not None, "Transaction must have user_id"
    assert transaction.amount != 0, "Transaction amount cannot be zero"
    assert transaction.transaction_type is not None, "Transaction must have type"
    assert transaction.plan_type_at_time is not None, "Transaction must record plan type"
    assert transaction.created_at is not None, "Transaction must have timestamp"


async def wait_for_async_task(coroutine, timeout: float = 5.0):
    """Wait for async task with timeout."""
    try:
        return await asyncio.wait_for(coroutine, timeout=timeout)
    except asyncio.TimeoutError:
        pytest.fail(f"Async task timed out after {timeout}s")


# Performance testing utilities

class PerformanceTimer:
    """Context manager for measuring execution time."""
    
    def __init__(self, max_duration_ms: int = 1000):
        self.max_duration_ms = max_duration_ms
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = datetime.utcnow()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.utcnow()
        duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        
        if duration_ms > self.max_duration_ms:
            pytest.fail(f"Operation took {duration_ms:.2f}ms, expected < {self.max_duration_ms}ms")
    
    @property
    def duration_ms(self) -> float:
        """Get duration in milliseconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return 0


# Concurrency testing utilities

async def run_concurrent_operations(operations: list, max_concurrent: int = 10):
    """Run multiple operations concurrently for testing race conditions."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def wrapped_operation(operation):
        async with semaphore:
            return await operation()
    
    results = await asyncio.gather(
        *[wrapped_operation(op) for op in operations],
        return_exceptions=True
    )
    
    return results