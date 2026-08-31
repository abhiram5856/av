"""rename severity_level to concern_level and add concern_score

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-30 15:00:00.000000

This migration renames the 'severity_level' column to 'concern_level'
and adds a new 'concern_score' float column in the 'diagnosis_history' table.

This reflects the rename from the legacy SeverityScoringEngine to the
MultimodalConcernScorer which uses 'concern_level' and 'concern_score'
as the canonical output fields.
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0002'
down_revision = '0001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Rename severity_level → concern_level
    op.alter_column(
        'diagnosis_history',
        'severity_level',
        new_column_name='concern_level',
        existing_type=sa.String(length=50),
        existing_nullable=True,
    )
    # Add concern_score column (new field, no prior equivalent)
    op.add_column(
        'diagnosis_history',
        sa.Column('concern_score', sa.Float(), nullable=True)
    )


def downgrade() -> None:
    # Drop concern_score
    op.drop_column('diagnosis_history', 'concern_score')
    # Rename concern_level → severity_level
    op.alter_column(
        'diagnosis_history',
        'concern_level',
        new_column_name='severity_level',
        existing_type=sa.String(length=50),
        existing_nullable=True,
    )
