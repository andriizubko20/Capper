"""add lineups table

Revision ID: j5f6g7h8i9j0
Revises: h3d4e5f6g7h8
Create Date: 2026-04-24

Stores confirmed starting XI per match/side as fetched from API-Football
/fixtures/lineups (~1h before kickoff). Used by lineup_strength feature
for late picks (1.5h before match).
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


revision = "j5f6g7h8i9j0"
down_revision = "h3d4e5f6g7h8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "lineups",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("match_id", sa.Integer(), sa.ForeignKey("matches.id"), nullable=False),
        sa.Column("team_id", sa.Integer(), sa.ForeignKey("teams.id"), nullable=False),
        sa.Column("side", sa.String(4), nullable=False),  # 'home' | 'away'
        sa.Column("formation", sa.String(20), nullable=True),
        sa.Column("starter_player_ids", JSONB, nullable=False),
        sa.Column("fetched_at", sa.DateTime(), nullable=False,
                  server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.UniqueConstraint("match_id", "side", name="uq_lineups_match_side"),
    )
    op.create_index("ix_lineups_match_id", "lineups", ["match_id"])


def downgrade() -> None:
    op.drop_index("ix_lineups_match_id", table_name="lineups")
    op.drop_table("lineups")
