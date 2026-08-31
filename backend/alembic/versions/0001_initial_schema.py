"""initial schema

Revision ID: 0001
Revises: 
Create Date: 2026-07-28 09:20:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '0001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade() -> None:
    # Create users table
    op.create_table(
        'users',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    op.create_index(op.f('ix_users_id'), 'users', ['id'], unique=False)

    # Create diagnosis_history table
    op.create_table(
        'diagnosis_history',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('user_id', sa.String(length=36), nullable=False),
        sa.Column('disease_name', sa.String(length=100), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('severity_level', sa.String(length=50), nullable=True),
        sa.Column('temperature', sa.Float(), nullable=True),
        sa.Column('humidity', sa.Float(), nullable=True),
        sa.Column('ph_level', sa.Float(), nullable=True),
        sa.Column('gradcam_heatmap', sa.String(), nullable=True),
        sa.Column('root_cause_json', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_diagnosis_history_id'), 'diagnosis_history', ['id'], unique=False)
    op.create_index(op.f('ix_diagnosis_history_user_id'), 'diagnosis_history', ['user_id'], unique=False)

def downgrade() -> None:
    op.drop_index(op.f('ix_diagnosis_history_user_id'), table_name='diagnosis_history')
    op.drop_index(op.f('ix_diagnosis_history_id'), table_name='diagnosis_history')
    op.drop_table('diagnosis_history')
    op.drop_index(op.f('ix_users_id'), table_name='users')
    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_table('users')
