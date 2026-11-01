"""create users conversations messages citations tables

Revision ID: 94cc75c8c47b
Revises: 
Create Date: 2026-11-02 00:13:07.350974

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '94cc75c8c47b'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute(
        """
        CREATE TABLE users (
            id            SERIAL PRIMARY KEY,
            email         TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )

    op.execute(
        """
        CREATE TABLE conversations (
            id         SERIAL PRIMARY KEY,
            user_id    INTEGER NOT NULL REFERENCES users(id),
            title      TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            deleted_at TIMESTAMPTZ
        )
        """
    )
    op.execute("CREATE INDEX conversations_user_id_idx ON conversations(user_id)")

    op.execute(
        """
        CREATE TABLE messages (
            id              SERIAL PRIMARY KEY,
            conversation_id INTEGER NOT NULL REFERENCES conversations(id),
            role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
            content         TEXT NOT NULL,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX messages_conversation_id_idx ON messages(conversation_id)")

    op.execute(
        """
        CREATE TABLE citations (
            id         SERIAL PRIMARY KEY,
            message_id INTEGER NOT NULL REFERENCES messages(id),
            doc_slug   TEXT NOT NULL,
            section    TEXT,
            content    TEXT NOT NULL
        )
        """
    )
    op.execute("CREATE INDEX citations_message_id_idx ON citations(message_id)")


def downgrade() -> None:
    """Downgrade schema."""
    # Reverse of creation order -- each table's foreign key must be
    # dropped before the table it points to.
    op.execute("DROP TABLE citations")
    op.execute("DROP TABLE messages")
    op.execute("DROP TABLE conversations")
    op.execute("DROP TABLE users")
