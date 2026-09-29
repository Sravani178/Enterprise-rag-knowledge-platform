"""Add content-hash deduplication to databases created by the earlier baseline."""

import sqlalchemy as sa
from alembic import op

revision = "0002_content_hash_compatibility"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())
    if "documents" not in tables:
        return

    document_columns = {column["name"] for column in inspector.get_columns("documents")}
    if "content_sha256" not in document_columns:
        op.add_column(
            "documents",
            sa.Column("content_sha256", sa.String(length=64), nullable=True),
        )

    constraints = {
        constraint.get("name")
        for constraint in inspector.get_unique_constraints("documents")
    }
    constraint_name = "uq_documents_organization_content_hash"
    if constraint_name not in constraints:
        op.create_unique_constraint(
            constraint_name,
            "documents",
            ["organization_id", "content_sha256"],
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "documents" not in set(inspector.get_table_names()):
        return
    constraints = {
        constraint.get("name")
        for constraint in inspector.get_unique_constraints("documents")
    }
    constraint_name = "uq_documents_organization_content_hash"
    if constraint_name in constraints:
        op.drop_constraint(constraint_name, "documents", type_="unique")
    if "content_sha256" in {column["name"] for column in inspector.get_columns("documents")}:
        op.drop_column("documents", "content_sha256")
