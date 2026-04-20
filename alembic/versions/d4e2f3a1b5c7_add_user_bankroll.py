"""add user bankroll

Revision ID: d4e2f3a1b5c7
Revises: c3f1a2b4d5e6
Create Date: 2026-04-19 00:01:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'd4e2f3a1b5c7'
down_revision: Union[str, Sequence[str], None] = 'c3f1a2b4d5e6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    result = conn.execute(sa.text(
        "SELECT 1 FROM information_schema.columns WHERE table_name='users' AND column_name='bankroll'"
    ))
    if not result.fetchone():
        op.add_column(
            'users',
            sa.Column('bankroll', sa.Float(), nullable=False, server_default='1000.0'),
        )


def downgrade() -> None:
    op.drop_column('users', 'bankroll')
