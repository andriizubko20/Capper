"""merge block3+block4 heads

Revision ID: 1ab60f2a8ba6
Revises: i4e5f6g7h8i9, j5f6g7h8i9j0
Create Date: 2026-04-26 09:34:55.139539

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1ab60f2a8ba6'
down_revision: Union[str, Sequence[str], None] = ('i4e5f6g7h8i9', 'j5f6g7h8i9j0')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
