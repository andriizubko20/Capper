"""add team_ratings table

Revision ID: h3d4e5f6g7h8
Revises: g2b3c4d5e6f7
Create Date: 2026-04-26
"""
from alembic import op
import sqlalchemy as sa


revision = "h3d4e5f6g7h8"
down_revision = "g2b3c4d5e6f7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "team_ratings",
        sa.Column("team_id", sa.Integer(), sa.ForeignKey("teams.id"), primary_key=True),
        sa.Column("rating", sa.Float(), nullable=False, server_default="1500"),
        sa.Column("rd",     sa.Float(), nullable=False, server_default="350"),
        sa.Column("volatility", sa.Float(), nullable=False, server_default="0.06"),
        sa.Column("matches_played", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_match_date", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=False,
                  server_default=sa.text("CURRENT_TIMESTAMP")),
    )
    op.create_index("ix_team_ratings_updated_at", "team_ratings", ["updated_at"])


def downgrade() -> None:
    op.drop_index("ix_team_ratings_updated_at", table_name="team_ratings")
    op.drop_table("team_ratings")
