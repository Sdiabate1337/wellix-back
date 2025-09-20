"""
Tests spécifiques pour le système de tokens avec PostgreSQL.
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4

# Import des modèles de base de données
from app.db.models.user import User
from app.db.models.token_system import UserTokenBalance, TokenTransaction, PlanType


@pytest.mark.asyncio
async def test_postgresql_connection(db_session):
    """Test de base pour vérifier la connexion PostgreSQL."""
    from sqlalchemy import text
    
    # Test simple de connexion
    result = db_session.execute(text("SELECT 1 as test"))
    value = result.scalar()
    assert value == 1
    print("✅ Connexion PostgreSQL établie avec succès")


@pytest.mark.asyncio
async def test_token_models_creation(db_session):
    """Test de création des modèles de tokens avec PostgreSQL."""
    # Créer un utilisateur de test
    test_user = User(
        id=uuid4(),
        email="test@wellix.ai",
        username="testuser",
        hashed_password="hashed_password_example",
        is_active=True,
        is_verified=True
    )
    db_session.add(test_user)
    db_session.flush()  # Pour obtenir l'ID
    
    # Créer un balance de tokens
    token_balance = UserTokenBalance(
        user_id=test_user.id,
        plan_type=PlanType.FREE,
        monthly_token_quota=1000,
        tokens_used_this_month=250,
        bonus_tokens=100,
        current_period_start=datetime.utcnow(),
        current_period_end=datetime.utcnow() + timedelta(days=30)
    )
    db_session.add(token_balance)
    db_session.flush()
    
    # Créer une transaction de tokens
    transaction = TokenTransaction(
        user_id=test_user.id,
        amount=-50,  # Consommation
        transaction_type="food_analysis",
        feature_used="nutrition_analysis",
        llm_model_used="gpt-4",
        tokens_estimated=45,
        tokens_actual=50,
        plan_type_at_time=PlanType.FREE.value
    )
    db_session.add(transaction)
    
    db_session.commit()
    
    # Vérifier que tout est créé correctement
    assert test_user.id is not None
    assert token_balance.user_id == test_user.id
    assert transaction.user_id == test_user.id
    assert transaction.amount == -50
    
    print("✅ Modèles de tokens créés avec succès dans PostgreSQL")


@pytest.mark.asyncio
async def test_token_relationships(db_session):
    """Test des relations entre les modèles de tokens."""
    # Créer un utilisateur de test
    test_user = User(
        id=uuid4(),
        email="relationships@wellix.ai",
        username="reltest",
        hashed_password="hashed_password_example",
        is_active=True,
        is_verified=True
    )
    db_session.add(test_user)
    db_session.flush()
    
    # Créer un balance de tokens
    token_balance = UserTokenBalance(
        user_id=test_user.id,
        plan_type=PlanType.PREMIUM,
        monthly_token_quota=5000,
        tokens_used_this_month=1200,
        bonus_tokens=500,
        current_period_start=datetime.utcnow(),
        current_period_end=datetime.utcnow() + timedelta(days=30)
    )
    db_session.add(token_balance)
    db_session.flush()
    
    # Créer plusieurs transactions
    transactions_data = [
        {"amount": -100, "type": "food_analysis", "feature": "nutrition_analysis"},
        {"amount": -75, "type": "chat", "feature": "health_consultation"},
        {"amount": 200, "type": "bonus", "feature": "weekly_bonus"},
    ]
    
    for tx_data in transactions_data:
        transaction = TokenTransaction(
            user_id=test_user.id,
            amount=tx_data["amount"],
            transaction_type=tx_data["type"],
            feature_used=tx_data["feature"],
            plan_type_at_time=PlanType.PREMIUM.value
        )
        db_session.add(transaction)
    
    db_session.commit()
    
    # Tester les relations SQLAlchemy
    from sqlalchemy import select
    
    # Récupérer le balance avec ses transactions
    balance_query = select(UserTokenBalance).where(UserTokenBalance.user_id == test_user.id)
    balance_result = db_session.execute(balance_query)
    balance = balance_result.scalar_one()
    
    # Vérifier que les propriétés fonctionnent
    assert balance.tokens_remaining > 0
    assert balance.usage_percentage > 0
    
    # Récupérer les transactions pour cet utilisateur
    transactions_query = select(TokenTransaction).where(TokenTransaction.user_id == test_user.id)
    transactions_result = db_session.execute(transactions_query)
    transactions = transactions_result.scalars().all()
    
    assert len(transactions) == 3
    
    # Vérifier les types de transactions
    consumption_count = len([tx for tx in transactions if tx.is_consumption])
    credit_count = len([tx for tx in transactions if tx.is_credit])
    
    assert consumption_count == 2  # 2 consommations
    assert credit_count == 1       # 1 crédit
    
    print("✅ Relations SQLAlchemy fonctionnent correctement avec PostgreSQL")
    print(f"   - Balance créé pour l'utilisateur {test_user.email}")
    print(f"   - {len(transactions)} transactions créées")
    print(f"   - Tokens restants: {balance.tokens_remaining}")
    print(f"   - Pourcentage d'utilisation: {balance.usage_percentage:.1f}%")


@pytest.mark.asyncio
async def test_uuid_compatibility(db_session):
    """Test spécifique pour la compatibilité UUID avec PostgreSQL."""
    # Créer plusieurs utilisateurs avec des UUIDs
    user_ids = [uuid4() for _ in range(3)]
    
    for i, user_id in enumerate(user_ids):
        user = User(
            id=user_id,
            email=f"uuid_test_{i}@wellix.ai",
            username=f"uuidtest{i}",
            hashed_password="hashed_password_example",
            is_active=True,
            is_verified=True
        )
        db_session.add(user)
    
    db_session.flush()
    
    # Créer des balances et transactions avec ces UUIDs
    for user_id in user_ids:
        balance = UserTokenBalance(
            user_id=user_id,
            plan_type=PlanType.FREE,
            monthly_token_quota=1000,
            tokens_used_this_month=100,
            bonus_tokens=0,
            current_period_start=datetime.utcnow(),
            current_period_end=datetime.utcnow() + timedelta(days=30)
        )
        db_session.add(balance)
        
        transaction = TokenTransaction(
            user_id=user_id,
            amount=-25,
            transaction_type="test",
            feature_used="uuid_test",
            plan_type_at_time=PlanType.FREE.value
        )
        db_session.add(transaction)
    
    db_session.commit()
    
    # Vérifier que tous les UUIDs sont correctement stockés et récupérés
    from sqlalchemy import select
    
    users_query = select(User)
    users_result = db_session.execute(users_query)
    users = users_result.scalars().all()
    
    balances_query = select(UserTokenBalance)
    balances_result = db_session.execute(balances_query)
    balances = balances_result.scalars().all()
    
    transactions_query = select(TokenTransaction)
    transactions_result = db_session.execute(transactions_query)
    transactions = transactions_result.scalars().all()
    
    # Vérifier que les UUIDs sont préservés
    stored_user_ids = {user.id for user in users}
    original_user_ids = set(user_ids)
    
    assert stored_user_ids.intersection(original_user_ids) == original_user_ids
    assert len(balances) >= 3
    assert len(transactions) >= 3
    
    print("✅ Compatibilité UUID avec PostgreSQL confirmée")
    print(f"   - {len(users)} utilisateurs créés avec UUID")
    print(f"   - {len(balances)} balances créés")
    print(f"   - {len(transactions)} transactions créées")
    print(f"   - UUIDs préservés correctement")


if __name__ == "__main__":
    # Permettre l'exécution directe pour les tests de développement
    pytest.main([__file__, "-v"])