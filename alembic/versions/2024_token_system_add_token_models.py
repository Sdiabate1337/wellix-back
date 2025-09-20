"""Add token system models

Revision ID: 2024_token_system
Revises: 68ab028bbbdd
Create Date: 2025-09-19 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '2024_token_system'
down_revision = '68ab028bbbdd'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create subscription_plans table
    op.create_table('subscription_plans',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('plan_type', sa.String(length=20), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('monthly_token_quota', sa.Integer(), nullable=False),
        sa.Column('bonus_tokens_on_signup', sa.Integer(), nullable=False, default=0),
        sa.Column('price_eur', sa.Float(), nullable=False),
        sa.Column('price_usd', sa.Float(), nullable=False),
        sa.Column('allowed_features', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('max_daily_analyses', sa.Integer(), nullable=True),
        sa.Column('priority_support', sa.Boolean(), nullable=False, default=False),
        sa.Column('api_access', sa.Boolean(), nullable=False, default=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('is_visible', sa.Boolean(), nullable=False, default=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('plan_type')
    )
    op.create_index(op.f('ix_subscription_plans_id'), 'subscription_plans', ['id'], unique=False)
    op.create_index(op.f('ix_subscription_plans_plan_type'), 'subscription_plans', ['plan_type'], unique=True)

    # Create user_token_balances table
    op.create_table('user_token_balances',
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('plan_type', sa.String(length=20), nullable=False, default='free'),
        sa.Column('plan_started_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('monthly_token_quota', sa.Integer(), nullable=False, default=20),
        sa.Column('tokens_used_this_month', sa.Integer(), nullable=False, default=0),
        sa.Column('bonus_tokens', sa.Integer(), nullable=False, default=0),
        sa.Column('current_period_start', sa.DateTime(timezone=True), nullable=False),
        sa.Column('current_period_end', sa.DateTime(timezone=True), nullable=False),
        sa.Column('last_reset_date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('total_tokens_consumed', sa.Integer(), nullable=False, default=0),
        sa.Column('total_analyses_performed', sa.Integer(), nullable=False, default=0),
        sa.Column('auto_renewal', sa.Boolean(), nullable=False, default=True),
        sa.Column('payment_method_id', sa.String(length=100), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('user_id')
    )
    op.create_index(op.f('ix_user_token_balances_user_id'), 'user_token_balances', ['user_id'], unique=False)

    # Create token_transactions table
    op.create_table('token_transactions',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('amount', sa.Integer(), nullable=False),
        sa.Column('transaction_type', sa.String(length=50), nullable=False),
        sa.Column('feature_used', sa.String(length=100), nullable=True),
        sa.Column('analysis_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('chat_session_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('processing_time_ms', sa.Integer(), nullable=True),
        sa.Column('llm_model_used', sa.String(length=50), nullable=True),
        sa.Column('tokens_estimated', sa.Integer(), nullable=True),
        sa.Column('tokens_actual', sa.Integer(), nullable=True),
        sa.Column('user_feedback', sa.Integer(), nullable=True),
        sa.Column('error_occurred', sa.Boolean(), nullable=False, default=False),
        sa.Column('refunded', sa.Boolean(), nullable=False, default=False),
        sa.Column('plan_type_at_time', sa.String(length=20), nullable=False),
        sa.Column('rate_limited', sa.Boolean(), nullable=False, default=False),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.Column('user_agent', sa.String(length=500), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['analysis_id'], ['food_analyses.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_token_transactions_analysis_id'), 'token_transactions', ['analysis_id'], unique=False)
    op.create_index(op.f('ix_token_transactions_created_at'), 'token_transactions', ['created_at'], unique=False)
    op.create_index(op.f('ix_token_transactions_id'), 'token_transactions', ['id'], unique=False)
    op.create_index(op.f('ix_token_transactions_transaction_type'), 'token_transactions', ['transaction_type'], unique=False)
    op.create_index(op.f('ix_token_transactions_user_id'), 'token_transactions', ['user_id'], unique=False)

    # Create usage_analytics table
    op.create_table('usage_analytics',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('period_type', sa.String(length=20), nullable=False),
        sa.Column('plan_type', sa.String(length=20), nullable=True),
        sa.Column('user_segment', sa.String(length=50), nullable=True),
        sa.Column('total_users', sa.Integer(), nullable=True, default=0),
        sa.Column('active_users', sa.Integer(), nullable=True, default=0),
        sa.Column('total_analyses', sa.Integer(), nullable=True, default=0),
        sa.Column('total_tokens_consumed', sa.Integer(), nullable=True, default=0),
        sa.Column('total_revenue', sa.Float(), nullable=True, default=0.0),
        sa.Column('new_subscriptions', sa.Integer(), nullable=True, default=0),
        sa.Column('cancelled_subscriptions', sa.Integer(), nullable=True, default=0),
        sa.Column('feature_usage', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('average_rating', sa.Float(), nullable=True),
        sa.Column('error_rate', sa.Float(), nullable=True),
        sa.Column('refund_rate', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_usage_analytics_date'), 'usage_analytics', ['date'], unique=False)
    op.create_index(op.f('ix_usage_analytics_id'), 'usage_analytics', ['id'], unique=False)
    op.create_index(op.f('ix_usage_analytics_plan_type'), 'usage_analytics', ['plan_type'], unique=False)


def downgrade() -> None:
    # Drop tables in reverse order due to foreign key constraints
    op.drop_index(op.f('ix_usage_analytics_plan_type'), table_name='usage_analytics')
    op.drop_index(op.f('ix_usage_analytics_id'), table_name='usage_analytics')
    op.drop_index(op.f('ix_usage_analytics_date'), table_name='usage_analytics')
    op.drop_table('usage_analytics')
    
    op.drop_index(op.f('ix_token_transactions_user_id'), table_name='token_transactions')
    op.drop_index(op.f('ix_token_transactions_transaction_type'), table_name='token_transactions')
    op.drop_index(op.f('ix_token_transactions_id'), table_name='token_transactions')
    op.drop_index(op.f('ix_token_transactions_created_at'), table_name='token_transactions')
    op.drop_index(op.f('ix_token_transactions_analysis_id'), table_name='token_transactions')
    op.drop_table('token_transactions')
    
    op.drop_index(op.f('ix_user_token_balances_user_id'), table_name='user_token_balances')
    op.drop_table('user_token_balances')
    
    op.drop_index(op.f('ix_subscription_plans_plan_type'), table_name='subscription_plans')
    op.drop_index(op.f('ix_subscription_plans_id'), table_name='subscription_plans')
    op.drop_table('subscription_plans')