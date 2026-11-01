from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

from cpa.config import get_settings

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# No SQLAlchemy models in this project (store.py and every other module
# talk to Postgres with plain psycopg -- see store.py's own docstring on
# why). Migrations are hand-written raw SQL via op.execute(), so there's
# nothing to autogenerate a diff against.
target_metadata = None


def _sqlalchemy_url() -> str:
    """Resolve the database to migrate.

    alembic.ini intentionally has no sqlalchemy.url (CLAUDE.md: no
    hardcoded paths/secrets) -- when nothing has already set one on this
    Config object, fall back to cpa.config.Settings, so a plain
    `alembic upgrade head` reads the same CPA_DATABASE_URL/.env every
    other script in this repo does. Tests call config.set_main_option(...)
    with a test database URL before invoking alembic's upgrade/downgrade
    commands, which leaves a real URL already in place by the time this
    function runs.

    Every URL in this project is a plain "postgresql://..." string (see
    cpa.config.Settings and tests/conftest.py) -- SQLAlchemy defaults
    that scheme to the psycopg2 driver, which isn't installed here (this
    project uses psycopg 3 everywhere). "+psycopg" picks the psycopg 3
    dialect SQLAlchemy ships support for, with no extra dependency.
    """
    url = config.get_main_option("sqlalchemy.url")
    if not url:
        url = get_settings().database_url
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


config.set_main_option("sqlalchemy.url", _sqlalchemy_url())


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
