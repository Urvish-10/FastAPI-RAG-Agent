"""
Empty all application data in the configured PostgreSQL database.

Clears the app data tables (documents, chat_sessions, chat_messages) and the
LangGraph checkpoint data tables. The `users` table is NEVER cleared, and the
schema and migration bookkeeping tables (alembic_version, checkpoint_migrations)
are left untouched.

By default this is a dry run that only prints row counts. Pass --yes to delete.

Usage:
    uv run python -m scripts.clean_db          # dry run
    uv run python -m scripts.clean_db --yes    # actually delete
"""
import argparse
import sys

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url

from app.core.config import settings
from app.db.database import Base
from app.models import models  # noqa: F401  (registers tables on Base.metadata)

# LangGraph AsyncPostgresSaver data tables (created at runtime, not by Alembic)
CHECKPOINT_DATA_TABLES = ["checkpoint_blobs", "checkpoint_writes", "checkpoints"]
# Tables that must never be wiped
PROTECTED_TABLES = {"users"}
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "postgres", None, ""}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--yes", action="store_true", help="actually delete the data")
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="allow running against a non-local database host",
    )
    args = parser.parse_args()

    url = make_url(settings.DATABASE_URL)
    target = f"{url.host or 'local socket'}:{url.port or 5432}/{url.database}"

    if url.host not in LOCAL_HOSTS and not args.allow_remote:
        print(f"Refusing to touch non-local database {target}. Use --allow-remote to override.")
        return 1

    engine = create_engine(settings.DATABASE_URL)
    existing = set(inspect(engine).get_table_names())

    app_tables = [
        t.name for t in Base.metadata.sorted_tables if t.name not in PROTECTED_TABLES
    ]
    tables = [t for t in app_tables + CHECKPOINT_DATA_TABLES if t in existing]

    print(f"Database: {target}")
    with engine.connect() as conn:
        for t in tables:
            count = conn.execute(text(f'SELECT count(*) FROM "{t}"')).scalar()
            print(f"  {t}: {count} rows")

    if not tables:
        print("No matching tables found; nothing to do.")
        return 0

    if not args.yes:
        print("\nDry run. Re-run with --yes to delete all rows in the tables above.")
        return 0

    assert not PROTECTED_TABLES & set(tables), "protected table in delete list"
    quoted = ", ".join(f'"{t}"' for t in tables)
    with engine.begin() as conn:  # single transaction
        conn.execute(text(f"TRUNCATE TABLE {quoted} RESTART IDENTITY CASCADE"))

    with engine.connect() as conn:
        remaining = {
            t: conn.execute(text(f'SELECT count(*) FROM "{t}"')).scalar() for t in tables
        }
    print("\nCleaned. Remaining rows:", remaining)
    return 0


if __name__ == "__main__":
    sys.exit(main())
