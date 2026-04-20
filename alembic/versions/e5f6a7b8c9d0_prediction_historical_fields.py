"""prediction historical fields

Revision ID: e5f6a7b8c9d0
Revises: d4e2f3a1b5c7
Create Date: 2026-04-19 20:00:00.000000

Makes match_id nullable and adds denormalized fields for historical imports
that don't have a corresponding Match record in the DB.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'e5f6a7b8c9d0'
down_revision: Union[str, Sequence[str], None] = 'd4e2f3a1b5c7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Make match_id nullable (historical predictions have no match record)
    op.alter_column('predictions', 'match_id',
                    existing_type=sa.Integer(),
                    nullable=True)

    # Denormalized fields for historical / imported predictions
    op.add_column('predictions', sa.Column('match_date',  sa.Date(),        nullable=True))
    op.add_column('predictions', sa.Column('home_name',   sa.String(100),   nullable=True))
    op.add_column('predictions', sa.Column('away_name',   sa.String(100),   nullable=True))
    op.add_column('predictions', sa.Column('league_name', sa.String(100),   nullable=True))


def downgrade() -> None:
    op.drop_column('predictions', 'league_name')
    op.drop_column('predictions', 'away_name')
    op.drop_column('predictions', 'home_name')
    op.drop_column('predictions', 'match_date')
    op.alter_column('predictions', 'match_id',
                    existing_type=sa.Integer(),
                    nullable=False)
