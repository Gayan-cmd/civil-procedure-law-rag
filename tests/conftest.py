from __future__ import annotations

import os
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from langchain_core.embeddings import Embeddings

# Any test that writes to the real vector store (build_vectorstore,
# load_vectorstore) must point at this database, never the plain
# CPA_DATABASE_URL one -- these tests truncate and rebuild the `chunks`
# table, which would otherwise wipe real ingested data every pytest run.
# Created once with:
#   docker exec cpa-postgres psql -U cpa -d cpa -c "CREATE DATABASE cpa_test OWNER cpa;"
TEST_DATABASE_URL = os.environ.get(
    "CPA_TEST_DATABASE_URL",
    "postgresql://cpa:cpa_local_dev@127.0.0.1:5432/cpa_test",
)


class FakeEmbeddings(Embeddings):
    """Deterministic, download-free stand-in for a real embedding model.

    Good enough for anything that only cares about wiring (does the store
    persist, does a filter include/exclude the right documents) rather than
    semantic search quality.

    Padded to 384 dimensions -- store.py's Postgres `chunks` table has a
    fixed vector(384) column (tied to the real embedding model), so even a
    fake vector must be exactly that length or the insert fails outright.
    Only the first 3 values carry any real signal; the rest are zeros,
    which doesn't change any two fake vectors' relative distance from each
    other (padding both sides of a comparison with the same zeros changes
    nothing about which one is closer).
    """

    _DIMENSIONS = 384

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        signal = [float(len(text) % 7), float(text.count("a")), 1.0]
        return signal + [0.0] * (self._DIMENSIONS - len(signal))


@pytest.fixture
def fake_embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()


_ALEMBIC_INI = Path(__file__).resolve().parents[1] / "alembic.ini"


def alembic_config_for_test_db() -> Config:
    config = Config(str(_ALEMBIC_INI))
    config.set_main_option("sqlalchemy.url", TEST_DATABASE_URL)
    return config


@pytest.fixture
def migrated_db():
    """Upgrade the test database to head, yield, then downgrade back to
    base -- every test using this fixture starts and ends with no trace
    of the users/conversations/messages/citations tables, the same
    "leave it as you found it" rule build_vectorstore's reset() follows
    for the chunks table.
    """
    config = alembic_config_for_test_db()
    command.upgrade(config, "head")
    try:
        yield
    finally:
        command.downgrade(config, "base")
