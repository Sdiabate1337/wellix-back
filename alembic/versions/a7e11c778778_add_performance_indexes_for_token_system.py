"""Add performance indexes for token system

Revision ID: a7e11c778778
Revises: 2024_token_system
Create Date: 2025-09-19 08:08:56.942891

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'a7e11c778778'
down_revision = '2024_token_system'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Index pour recherche rapide des transactions par utilisateur et date
    op.create_index(
        'idx_token_transactions_user_date', 
        'token_transactions', 
        ['user_id', 'created_at']
    )
    
    # Index pour recherche par type de transaction (analytics)
    op.create_index(
        'idx_token_transactions_type_date', 
        'token_transactions', 
        ['transaction_type', 'created_at']
    )
    
    # Index pour recherche des transactions par plan (billing analytics)
    op.create_index(
        'idx_token_transactions_plan_date', 
        'token_transactions', 
        ['plan_type_at_time', 'created_at']
    )
    
    # Index pour les analytics d'usage par période
    op.create_index(
        'idx_usage_analytics_date_plan', 
        'usage_analytics', 
        ['date', 'plan_type']
    )
    
    # Index pour recherche rapide des balances actives
    op.create_index(
        'idx_user_token_balances_plan', 
        'user_token_balances', 
        ['plan_type', 'current_period_end']
    )


def downgrade() -> None:
    # Suppression des indexes en ordre inverse
    op.drop_index('idx_user_token_balances_plan')
    op.drop_index('idx_usage_analytics_date_plan')
    op.drop_index('idx_token_transactions_plan_date')
    op.drop_index('idx_token_transactions_type_date')
    op.drop_index('idx_token_transactions_user_date')
