"""add result pnl timing snapshots

Revision ID: c3f1a2b4d5e6
Revises: a7dd83965c30
Create Date: 2026-04-19 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'c3f1a2b4d5e6'
down_revision: Union[str, Sequence[str], None] = 'a7dd83965c30'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # predictions: result, pnl, timing, is_active
    op.add_column('predictions', sa.Column('result', sa.Text(), nullable=True))
    op.add_column('predictions', sa.Column('pnl', sa.Float(), nullable=True))
    op.add_column('predictions', sa.Column('timing', sa.String(length=10), nullable=True))
    op.add_column('predictions', sa.Column('is_active', sa.Boolean(), nullable=False,
                                           server_default=sa.text('true')))

    # bankroll_snapshots — нова таблиця
    op.create_table(
        'bankroll_snapshots',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('balance', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False,
                  server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_bankroll_snapshots_user_id', 'bankroll_snapshots', ['user_id'])
    op.create_index('ix_bankroll_snapshots_created_at', 'bankroll_snapshots', ['created_at'])


def downgrade() -> None:
    op.drop_index('ix_bankroll_snapshots_created_at', 'bankroll_snapshots')
    op.drop_index('ix_bankroll_snapshots_user_id', 'bankroll_snapshots')
    op.drop_table('bankroll_snapshots')

    op.drop_column('predictions', 'is_active')
    op.drop_column('predictions', 'timing')
    op.drop_column('predictions', 'pnl')
    op.drop_column('predictions', 'result')
