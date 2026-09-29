"""Create persistent BM25 document statistics and postings."""

import sqlalchemy as sa
from alembic import op

revision = "0003_persistent_bm25"
down_revision = "0002_content_hash_compatibility"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "bm25_documents",
        sa.Column("chunk_id", sa.Uuid(), nullable=False),
        sa.Column("organization_id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("document_length", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["chunk_id"], ["document_chunks.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("chunk_id"),
    )
    op.create_index(
        "ix_bm25_documents_organization",
        "bm25_documents",
        ["organization_id"],
    )
    op.create_index(
        "ix_bm25_documents_organization_document",
        "bm25_documents",
        ["organization_id", "document_id"],
    )
    op.create_table(
        "bm25_postings",
        sa.Column("organization_id", sa.Uuid(), nullable=False),
        sa.Column("term", sa.String(length=255), nullable=False),
        sa.Column("chunk_id", sa.Uuid(), nullable=False),
        sa.Column("term_frequency", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["chunk_id"], ["bm25_documents.chunk_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("organization_id", "term", "chunk_id"),
    )
    op.create_index(
        "ix_bm25_postings_organization_term",
        "bm25_postings",
        ["organization_id", "term"],
    )
    op.create_index("ix_bm25_postings_chunk", "bm25_postings", ["chunk_id"])


def downgrade() -> None:
    op.drop_index("ix_bm25_postings_chunk", table_name="bm25_postings")
    op.drop_index("ix_bm25_postings_organization_term", table_name="bm25_postings")
    op.drop_table("bm25_postings")
    op.drop_index(
        "ix_bm25_documents_organization_document",
        table_name="bm25_documents",
    )
    op.drop_index("ix_bm25_documents_organization", table_name="bm25_documents")
    op.drop_table("bm25_documents")
