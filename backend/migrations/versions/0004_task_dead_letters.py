"""Add a durable dead-letter table for terminal Celery failures."""

import sqlalchemy as sa
from alembic import op

revision = "0004_task_dead_letters"
down_revision = "0003_persistent_bm25"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "task_dead_letters",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("task_id", sa.String(length=255), nullable=False),
        sa.Column("task_name", sa.String(length=255), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("exception_type", sa.String(length=255), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=False),
        sa.Column("retry_count", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("task_id"),
    )
    op.create_index(
        "ix_task_dead_letters_status_created",
        "task_dead_letters",
        ["status", "created_at"],
    )
    op.create_index(
        "ix_task_dead_letters_document",
        "task_dead_letters",
        ["document_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_task_dead_letters_document", table_name="task_dead_letters")
    op.drop_index("ix_task_dead_letters_status_created", table_name="task_dead_letters")
    op.drop_table("task_dead_letters")
