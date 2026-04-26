"""add player_stats table for xg_share / missing_xg_share feature

Revision ID: i4e5f6g7h8i9
Revises: h3d4e5f6g7h8
Create Date: 2026-04-24
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "i4e5f6g7h8i9"
down_revision: Union[str, Sequence[str], None] = "h3d4e5f6g7h8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "player_stats",
        sa.Column("player_id",      sa.Integer(), primary_key=True),  # API-Football player id
        sa.Column("team_id",        sa.Integer(), sa.ForeignKey("teams.id"), nullable=True),
        sa.Column("name",           sa.String(length=200), nullable=True),
        sa.Column("league_id",      sa.Integer(), sa.ForeignKey("leagues.id"), nullable=True),
        sa.Column("season",         sa.Integer(), nullable=True),
        sa.Column("goals",          sa.Integer(), nullable=False, server_default="0"),
        sa.Column("assists",        sa.Integer(), nullable=False, server_default="0"),
        sa.Column("minutes_played", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("xg_share",       sa.Float(),   nullable=False, server_default="0"),
        sa.Column("updated_at",     sa.DateTime(), nullable=False,
                  server_default=sa.text("CURRENT_TIMESTAMP")),
    )
    op.create_index("ix_player_stats_team_id",   "player_stats", ["team_id"])
    op.create_index("ix_player_stats_league_id", "player_stats", ["league_id"])
    op.create_index("ix_player_stats_season",    "player_stats", ["season"])


def downgrade() -> None:
    op.drop_index("ix_player_stats_season",    table_name="player_stats")
    op.drop_index("ix_player_stats_league_id", table_name="player_stats")
    op.drop_index("ix_player_stats_team_id",   table_name="player_stats")
    op.drop_table("player_stats")
