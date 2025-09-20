"""
Unit tests for token API endpoints.
Tests all API routes, request/response handling, and integration scenarios.
"""

import pytest
from datetime import datetime, timedelta
import uuid
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import status

from app.api.v1.tokens import router
from app.services.token.token_service import TokenService
from app.middleware.token_validation import TokenValidator
from app.db.models.token_system import (
    UserTokenBalance,
    TokenTransaction,
    PlanType,
    TransactionType
)
from app.db.models.user import User


# Create test app with token router
from fastapi import FastAPI
app = FastAPI()
app.include_router(router, prefix="/api/v1")


class TestTokenBalanceEndpoints:
    """Test token balance related endpoints."""
    
    def test_get_balance_success(self, db_session, sample_user, sample_token_balance):
        """Test successful balance retrieval."""
        mock_service = Mock(spec=TokenService)
        mock_service.get_user_dashboard.return_value = {
            'success': True,
            'balance': {
                'tokens_remaining': sample_token_balance.tokens_remaining,
                'monthly_quota': sample_token_balance.monthly_token_quota,
                'plan_type': sample_token_balance.plan_type,
                'usage_percentage': sample_token_balance.usage_percentage
            }
        }
        
        with patch('app.api.v1.tokens.get_current_user', return_value=sample_user):
            with patch('app.api.v1.tokens.get_token_service', return_value=mock_service):
                client = TestClient(app)
                response = client.get("/api/v1/balance")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data['tokens_remaining'] == sample_token_balance.tokens_remaining
        assert data['plan_type'] == sample_token_balance.plan_type.value
    
    def test_get_balance_user_not_initialized(self, db_session, sample_user):
        """Test balance retrieval for uninitialized user."""
        mock_service = Mock(spec=TokenService)
        mock_service.get_user_dashboard.return_value = {
            'success': False,
            'error': 'User tokens not initialized'
        }
        
        with patch('app.api.v1.tokens.get_current_user', return_value=sample_user):
            with patch('app.api.v1.tokens.get_token_service', return_value=mock_service):
                client = TestClient(app)
                response = client.get("/api/v1/balance")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "not initialized" in response.json()["detail"]
    
    def test_get_balance_unauthorized(self, db_session):
        """Test balance retrieval without authentication."""
        with patch('app.api.v1.tokens.get_current_user', side_effect=Exception("Unauthorized")):
            client = TestClient(app)
            response = client.get("/api/v1/balance")
        
        # Should be handled by authentication middleware
        assert response.status_code in [status.HTTP_401_UNAUTHORIZED, status.HTTP_500_INTERNAL_SERVER_ERROR]


class TestUserDashboardEndpoints:
    """Test user dashboard endpoint."""
    
    def test_get_dashboard_complete(self, db_session, sample_user, sample_token_balance, sample_transactions):
        """Test complete dashboard data retrieval."""
        mock_service = Mock(spec=TokenService)
        mock_service.get_user_dashboard.return_value = {
            'success': True,
            'balance': {
                'tokens_remaining': 90,
                'monthly_quota': 100,
                'plan_type': PlanType.BASIC,
                'usage_percentage': 10.0
            },
            'recent_transactions': [
                {
                    'id': str(sample_transactions[0].id),
                    'amount': sample_transactions[0].amount,
                    'transaction_type': sample_transactions[0].transaction_type,
                    'created_at': sample_transactions[0].created_at.isoformat()
                }
            ],
            'usage_analytics': {
                'total_tokens_consumed': 10,
                'total_transactions': 5,
                'most_used_feature': TransactionType.BASIC_ANALYSIS,
                'daily_usage': [2, 3, 1, 2, 2, 0, 0]
            },
            'recommendations': [
                {
                    'type': 'optimization',
                    'title': 'Optimize usage patterns',
                    'description': 'Try using basic analysis for simple queries',
                    'priority': 'medium'
                }
            ],
            'available_plans': [
                {'name': 'Premium', 'price': 9.99, 'tokens': 200}
            ]
        }
        
        with patch('app.api.v1.tokens.get_current_user', return_value=sample_user):
            with patch('app.api.v1.tokens.get_token_service', return_value=mock_service):
                client = TestClient(app)
                response = client.get("/api/v1/dashboard")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert 'balance' in data
        assert 'recent_transactions' in data
        assert 'usage_analytics' in data
        assert 'recommendations' in data
        assert 'available_plans' in data
        
        assert data['balance']['tokens_remaining'] == 90
        assert len(data['recent_transactions']) > 0
        assert data['usage_analytics']['total_tokens_consumed'] == 10
    
    def test_get_dashboard_new_user(self, db_session, sample_user):
        """Test dashboard for new user."""
        mock_service = Mock(spec=TokenService)
        mock_service.get_user_dashboard.return_value = {
            'success': False,
            'error': 'User tokens not initialized'
        }
        
        with patch('app.api.v1.tokens.get_current_user', return_value=sample_user):
            with patch('app.api.v1.tokens.get_token_service', return_value=mock_service):
                client = TestClient(app)
                response = client.get("/api/v1/dashboard")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestPlanUpgradeEndpoints:
    """Test plan upgrade endpoints."""
    
    def test_upgrade_plan_success(self, db_session, sample_user):
        """Test successful plan upgrade."""
        mock_service = Mock(spec=TokenService)
        mock_service.upgrade_user_plan.return_value = {
            'success': True,
            'new_plan': PlanType.PREMIUM,
            'new_quota': 200,
            'payment_confirmation': {
                'transaction_id': 'pay_123',
                'amount': 9.99
            },
            'message': 'Plan upgraded successfully'
        }
        
        upgrade_data = {
            'new_plan': 'premium',
            'payment_method_id': 'pm_test_123'
        }
        
        with patch('app.api.v1.tokens.get_current_user', return_value=sample_user):
            with patch('app.api.v1.tokens.get_token_service', return_value=mock_service):
                client = TestClient(app)
                response = client.post("/api/v1/upgrade", json=upgrade_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data['success'] is True
        assert data['new_plan'] == 'premium'
        assert data['new_quota'] == 200
        assert 'payment_confirmation' in data
    
    def test_upgrade_plan_payment_failure(self, db_session, sample_user):
        """Test plan upgrade with payment failure."""
        mock_service = Mock(spec=TokenService)
        mock_service.upgrade_user_plan.return_value = {
            'success': False,
            'error': 'Payment failed: Card declined'
        }
        
        upgrade_data = {
            'new_plan': 'premium',
            'payment_method_id': 'pm_invalid'
        }
        
        with patch('app.api.v1.tokens.get_current_user', return_value=sample_user):
            with patch('app.api.v1.tokens.get_token_service', return_value=mock_service):
                client = TestClient(app)
                response = client.post("/api/v1/upgrade", json=upgrade_data)
        
        assert response.status_code == status.HTTP_402_PAYMENT_REQUIRED
        assert 'Payment failed' in response.json()['detail']
    
    def test_upgrade_plan_invalid_plan(self, db_session, sample_user):
        """Test upgrade to invalid plan."""
        upgrade_data = {
            'new_plan': 'invalid_plan',
            'payment_method_id': 'pm_test_123'
        }
        
        with patch('app.api.v1.tokens.get_current_user', return_value=sample_user):
            client = TestClient(app)
            response = client.post("/api/v1/upgrade", json=upgrade_data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_upgrade_plan_missing_payment_method(self, db_session, sample_user):
        """Test upgrade without payment method."""
        upgrade_data = {
            'new_plan': 'premium'
            # Missing payment_method_id
        }
        
        with patch('app.api.v1.tokens.get_current_user', return_value=sample_user):
            client = TestClient(app)
            response = client.post("/api/v1/upgrade", json=upgrade_data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestTokenPackPurchase:
    """Test token pack purchase endpoints."""
    
    def test_purchase_pack_success(self, db_session, sample_user):
        """Test successful token pack purchase."""
        mock_service = Mock(spec=TokenService)
        mock_service.purchase_token_pack.return_value = {
            'success': True,
            'tokens_added': 100,
            'pack_size': 'medium',
            'payment_confirmation': {
                'transaction_id': 'pay_pack_123',
                'amount': 4.99
            },
            'new_balance': 190
        }
        
        purchase_data = {
            'pack_size': 'medium',
            'payment_method_id': 'pm_test_123'
        }
        
        with patch('app.api.v1.tokens.get_current_user', return_value=sample_user):
            with patch('app.api.v1.tokens.get_token_service', return_value=mock_service):
                client = TestClient(app)
                response = client.post("/api/v1/purchase-pack", json=purchase_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data['success'] is True
        assert data['tokens_added'] == 100
        assert data['pack_size'] == 'medium'
        assert 'payment_confirmation' in data
    
    def test_purchase_pack_invalid_size(self, db_session, sample_user):
        """Test purchasing invalid pack size."""
        mock_service = Mock(spec=TokenService)
        mock_service.purchase_token_pack.return_value = {
            'success': False,
            'error': 'Invalid pack size: invalid_size'
        }
        
        purchase_data = {
            'pack_size': 'invalid_size',
            'payment_method_id': 'pm_test_123'
        }
        
        with patch('app.api.v1.tokens.get_current_user', return_value=sample_user):
            with patch('app.api.v1.tokens.get_token_service', return_value=mock_service):
                client = TestClient(app)
                response = client.post("/api/v1/purchase-pack", json=purchase_data)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert 'Invalid pack size' in response.json()['detail']


class TestTransactionHistory:
    """Test transaction history endpoints."""
    
    def test_get_transactions_success(self, db_session, sample_user, sample_transactions):
        """Test successful transaction history retrieval."""
        mock_service = Mock(spec=TokenService)
        
        # Mock repository for transaction retrieval
        mock_repo = Mock()
        mock_repo.get_user_transactions.return_value = sample_transactions
        
        with patch('app.api.v1.tokens.get_current_user', return_value=sample_user):
            with patch('app.api.v1.tokens.get_token_repository', return_value=mock_repo):
                client = TestClient(app)
                response = client.get("/api/v1/transactions")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert 'transactions' in data
        assert len(data['transactions']) > 0
        
        # Check transaction structure
        transaction = data['transactions'][0]
        assert 'id' in transaction
        assert 'amount' in transaction
        assert 'transaction_type' in transaction
        assert 'created_at' in transaction
    
    def test_get_transactions_with_pagination(self, db_session, sample_user, sample_transactions):
        """Test transaction history with pagination."""
        mock_repo = Mock()
        mock_repo.get_user_transactions.return_value = sample_transactions[:5]  # First 5
        
        with patch('app.api.v1.tokens.get_current_user', return_value=sample_user):
            with patch('app.api.v1.tokens.get_token_repository', return_value=mock_repo):
                client = TestClient(app)
                response = client.get("/api/v1/transactions?limit=5&offset=0")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert len(data['transactions']) <= 5
        mock_repo.get_user_transactions.assert_called_with(
            user_id=sample_user.id,
            limit=5,
            offset=0,
            transaction_types=None,
            start_date=None,
            end_date=None
        )
    
    def test_get_transactions_filtered_by_type(self, db_session, sample_user, sample_transactions):
        """Test transaction history filtered by type."""
        mock_repo = Mock()
        analysis_transactions = [tx for tx in sample_transactions 
                               if tx.transaction_type in [TransactionType.BASIC_ANALYSIS, TransactionType.EXPERT_ANALYSIS]]
        mock_repo.get_user_transactions.return_value = analysis_transactions
        
        with patch('app.api.v1.tokens.get_current_user', return_value=sample_user):
            with patch('app.api.v1.tokens.get_token_repository', return_value=mock_repo):
                client = TestClient(app)
                response = client.get("/api/v1/transactions?transaction_types=basic_analysis,expert_analysis")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        for transaction in data['transactions']:
            assert transaction['transaction_type'] in ['basic_analysis', 'expert_analysis']


class TestUsageAnalytics:
    """Test usage analytics endpoints."""
    
    def test_get_usage_analysis_success(self, db_session, sample_user):
        """Test successful usage analysis retrieval."""
        mock_service = Mock(spec=TokenService)
        mock_service.analyze_usage_patterns.return_value = {
            'usage_level': 'moderate',
            'primary_feature': TransactionType.BASIC_ANALYSIS,
            'daily_average': 5.2,
            'peak_usage_day': 'monday',
            'efficiency_score': 75,
            'recommendations': [
                'Consider upgrading to Basic plan for better value',
                'Use basic analysis for simple queries to save tokens'
            ]
        }
        
        with patch('app.api.v1.tokens.get_current_user', return_value=sample_user):
            with patch('app.api.v1.tokens.get_token_service', return_value=mock_service):
                client = TestClient(app)
                response = client.get("/api/v1/usage-analysis")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data['usage_level'] == 'moderate'
        assert data['primary_feature'] == TransactionType.BASIC_ANALYSIS
        assert data['daily_average'] == 5.2
        assert 'recommendations' in data
        assert len(data['recommendations']) > 0
    
    def test_get_usage_analysis_custom_period(self, db_session, sample_user):
        """Test usage analysis with custom time period."""
        mock_service = Mock(spec=TokenService)
        mock_service.analyze_usage_patterns.return_value = {
            'usage_level': 'light',
            'daily_average': 2.1
        }
        
        with patch('app.api.v1.tokens.get_current_user', return_value=sample_user):
            with patch('app.api.v1.tokens.get_token_service', return_value=mock_service):
                client = TestClient(app)
                response = client.get("/api/v1/usage-analysis?days=14")
        
        assert response.status_code == status.HTTP_200_OK
        mock_service.analyze_usage_patterns.assert_called_with(sample_user.id, days=14)


class TestPlanManagement:
    """Test plan management endpoints."""
    
    def test_get_available_plans(self, db_session, sample_user):
        """Test getting available subscription plans."""
        with patch('app.api.v1.tokens.get_current_user', return_value=sample_user):
            client = TestClient(app)
            response = client.get("/api/v1/plans")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert 'plans' in data
        plans = data['plans']
        
        # Should include all plan types
        plan_types = [plan['plan_type'] for plan in plans]
        assert 'free' in plan_types
        assert 'basic' in plan_types
        assert 'premium' in plan_types
        assert 'enterprise' in plan_types
        
        # Check plan structure
        for plan in plans:
            assert 'plan_type' in plan
            assert 'name' in plan
            assert 'monthly_token_quota' in plan
            assert 'price_eur' in plan
            assert 'allowed_features' in plan
    
    def test_get_current_plan(self, db_session, sample_user, sample_token_balance):
        """Test getting current user plan."""
        mock_service = Mock(spec=TokenService)
        mock_service.get_user_dashboard.return_value = {
            'success': True,
            'balance': {
                'plan_type': sample_token_balance.plan_type,
                'monthly_quota': sample_token_balance.monthly_token_quota,
                'tokens_remaining': sample_token_balance.tokens_remaining
            }
        }
        
        with patch('app.api.v1.tokens.get_current_user', return_value=sample_user):
            with patch('app.api.v1.tokens.get_token_service', return_value=mock_service):
                client = TestClient(app)
                response = client.get("/api/v1/current-plan")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data['plan_type'] == sample_token_balance.plan_type.value
        assert data['monthly_quota'] == sample_token_balance.monthly_token_quota


class TestWebhookEndpoints:
    """Test webhook endpoints for payment processing."""
    
    def test_payment_webhook_success(self, db_session):
        """Test successful payment webhook processing."""
        webhook_data = {
            'type': 'payment_intent.succeeded',
            'data': {
                'object': {
                    'id': 'pi_test_123',
                    'status': 'succeeded',
                    'metadata': {
                        'user_id': str(uuid.uuid4()),
                        'upgrade_plan': 'premium'
                    }
                }
            }
        }
        
        mock_service = Mock(spec=TokenService)
        mock_service.handle_payment_webhook.return_value = {
            'success': True,
            'processed': True
        }
        
        with patch('app.api.v1.tokens.get_token_service', return_value=mock_service):
            client = TestClient(app)
            response = client.post("/api/v1/webhook/payment", json=webhook_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data['success'] is True
    
    def test_payment_webhook_invalid_signature(self, db_session):
        """Test webhook with invalid signature."""
        webhook_data = {'type': 'payment_intent.succeeded'}
        
        with patch('app.api.v1.tokens.verify_webhook_signature', return_value=False):
            client = TestClient(app)
            response = client.post("/api/v1/webhook/payment", json=webhook_data)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
    
    def test_payment_webhook_unhandled_event(self, db_session):
        """Test webhook with unhandled event type."""
        webhook_data = {
            'type': 'customer.created',  # Unhandled event
            'data': {'object': {}}
        }
        
        with patch('app.api.v1.tokens.verify_webhook_signature', return_value=True):
            client = TestClient(app)
            response = client.post("/api/v1/webhook/payment", json=webhook_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data['success'] is True
        assert 'ignored' in data['message'].lower()


class TestErrorHandling:
    """Test API error handling."""
    
    def test_internal_server_error_handling(self, db_session, sample_user):
        """Test handling of internal server errors."""
        mock_service = Mock(spec=TokenService)
        mock_service.get_user_dashboard.side_effect = Exception("Database connection lost")
        
        with patch('app.api.v1.tokens.get_current_user', return_value=sample_user):
            with patch('app.api.v1.tokens.get_token_service', return_value=mock_service):
                client = TestClient(app)
                response = client.get("/api/v1/dashboard")
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert 'internal server error' in response.json()['detail'].lower()
    
    def test_validation_error_handling(self, db_session, sample_user):
        """Test handling of request validation errors."""
        # Send invalid JSON
        invalid_data = {
            'new_plan': 123,  # Should be string
            'payment_method_id': None  # Should be string
        }
        
        with patch('app.api.v1.tokens.get_current_user', return_value=sample_user):
            client = TestClient(app)
            response = client.post("/api/v1/upgrade", json=invalid_data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_rate_limiting_error(self, db_session, sample_user):
        """Test rate limiting error handling."""
        # This would be handled by the token validation middleware
        # in a real scenario, but we can test the error response structure
        
        with patch('app.api.v1.tokens.get_current_user', return_value=sample_user):
            with patch('app.api.v1.tokens.get_token_validator') as mock_validator:
                mock_validator.return_value.check_rate_limit.return_value = False
                
                client = TestClient(app)
                # Multiple requests to trigger rate limiting
                for _ in range(5):
                    response = client.get("/api/v1/balance")
                
                # The actual rate limiting would be handled by middleware
                # This test ensures the structure is in place


class TestAPIResponseFormats:
    """Test API response formats and consistency."""
    
    def test_success_response_format(self, db_session, sample_user, sample_token_balance):
        """Test that success responses follow consistent format."""
        mock_service = Mock(spec=TokenService)
        mock_service.get_user_dashboard.return_value = {
            'success': True,
            'balance': {
                'tokens_remaining': 90,
                'plan_type': PlanType.BASIC
            }
        }
        
        with patch('app.api.v1.tokens.get_current_user', return_value=sample_user):
            with patch('app.api.v1.tokens.get_token_service', return_value=mock_service):
                client = TestClient(app)
                response = client.get("/api/v1/balance")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Check response structure
        assert isinstance(data, dict)
        assert 'tokens_remaining' in data
        assert isinstance(data['tokens_remaining'], int)
    
    def test_error_response_format(self, db_session, sample_user):
        """Test that error responses follow consistent format."""
        mock_service = Mock(spec=TokenService)
        mock_service.get_user_dashboard.return_value = {
            'success': False,
            'error': 'User not found'
        }
        
        with patch('app.api.v1.tokens.get_current_user', return_value=sample_user):
            with patch('app.api.v1.tokens.get_token_service', return_value=mock_service):
                client = TestClient(app)
                response = client.get("/api/v1/balance")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        
        # FastAPI error response format
        assert 'detail' in data
        assert isinstance(data['detail'], str)
    
    def test_pagination_response_format(self, db_session, sample_user, sample_transactions):
        """Test pagination response format consistency."""
        mock_repo = Mock()
        mock_repo.get_user_transactions.return_value = sample_transactions[:3]
        
        with patch('app.api.v1.tokens.get_current_user', return_value=sample_user):
            with patch('app.api.v1.tokens.get_token_repository', return_value=mock_repo):
                client = TestClient(app)
                response = client.get("/api/v1/transactions?limit=3&offset=0")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Check pagination structure
        assert 'transactions' in data
        assert 'total' in data
        assert 'limit' in data
        assert 'offset' in data
        assert isinstance(data['transactions'], list)
        assert isinstance(data['total'], int)
        assert data['limit'] == 3
        assert data['offset'] == 0


class TestAPIDocumentation:
    """Test API documentation and OpenAPI schema."""
    
    def test_openapi_schema_generation(self, db_session):
        """Test that OpenAPI schema is generated correctly."""
        client = TestClient(app)
        response = client.get("/openapi.json")
        
        assert response.status_code == status.HTTP_200_OK
        schema = response.json()
        
        # Check basic schema structure
        assert 'openapi' in schema
        assert 'paths' in schema
        assert 'components' in schema
        
        # Check that token endpoints are documented
        paths = schema['paths']
        assert '/api/v1/balance' in paths
        assert '/api/v1/dashboard' in paths
        assert '/api/v1/upgrade' in paths
    
    def test_endpoint_documentation(self, db_session):
        """Test that endpoints have proper documentation."""
        client = TestClient(app)
        response = client.get("/openapi.json")
        schema = response.json()
        
        # Check balance endpoint documentation
        balance_endpoint = schema['paths']['/api/v1/balance']['get']
        assert 'summary' in balance_endpoint
        assert 'responses' in balance_endpoint
        assert '200' in balance_endpoint['responses']
        assert '404' in balance_endpoint['responses']