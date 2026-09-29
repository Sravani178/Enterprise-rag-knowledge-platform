from pathlib import Path

from alembic import command
from alembic.config import Config

from app.core.config import get_settings


def init_db() -> None:
    backend_root = Path(__file__).resolve().parents[2]
    config = Config(str(backend_root / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", get_settings().database_url)
    command.upgrade(config, "head")


if __name__ == "__main__":
    init_db()
