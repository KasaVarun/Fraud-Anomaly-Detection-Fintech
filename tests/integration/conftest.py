"""Dedicated PostgreSQL integration fixtures."""

from collections.abc import Iterator

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import make_url

from fraud_detection.config import Settings

_ALEMBIC_TEST_DSN: str | None = None


def _test_dsn() -> str:
    settings = Settings(_env_file=None)
    if settings.test_postgres_dsn is None:
        pytest.skip("FRAUD_TEST_POSTGRES_DSN is not configured")
    dsn = settings.test_postgres_dsn.get_secret_value()
    url = make_url(dsn)
    if url.database is None or not url.database.endswith("_test"):
        pytest.fail("integration tests require a database name ending in _test")
    if url.host not in {"127.0.0.1", "localhost", "::1"}:
        pytest.fail("integration tests require a localhost PostgreSQL database")
    return dsn


def run_alembic(revision: str) -> None:
    if _ALEMBIC_TEST_DSN is None:
        raise RuntimeError("PostgreSQL integration fixture has not initialized Alembic")
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", _ALEMBIC_TEST_DSN)
    command.upgrade(config, revision) if revision != "base" else command.downgrade(config, "base")


def run_alembic_downgrade(revision: str) -> None:
    if _ALEMBIC_TEST_DSN is None:
        raise RuntimeError("PostgreSQL integration fixture has not initialized Alembic")
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", _ALEMBIC_TEST_DSN)
    command.downgrade(config, revision)


@pytest.fixture(scope="session")
def postgres_engine() -> Iterator[Engine]:
    global _ALEMBIC_TEST_DSN
    dsn = _test_dsn()
    _ALEMBIC_TEST_DSN = dsn
    engine = create_engine(dsn, hide_parameters=True)
    with engine.begin() as connection:
        connection.execute(text("DROP SCHEMA public CASCADE"))
        connection.execute(text("CREATE SCHEMA public"))
    run_alembic("head")
    yield engine
    with engine.begin() as connection:
        connection.execute(text("DROP SCHEMA public CASCADE"))
        connection.execute(text("CREATE SCHEMA public"))
    engine.dispose()
    _ALEMBIC_TEST_DSN = None
