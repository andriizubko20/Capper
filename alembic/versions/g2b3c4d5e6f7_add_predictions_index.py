"""add index on predictions(match_id, model_version, is_active)

Revision ID: g2b3c4d5e6f7
Revises: f1a2b3c4d5e6
Create Date: 2026-04-22

"""
from alembic import op

revision = 'g2b3c4d5e6f7'
down_revision = 'f1a2b3c4d5e6'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        'ix_predictions_match_model_active',
        'predictions',
        ['match_id', 'model_version', 'is_active'],
    )


def downgrade() -> None:
    op.drop_index('ix_predictions_match_model_active', table_name='predictions')
