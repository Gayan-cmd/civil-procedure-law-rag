"""Vector store read/write, backed by Postgres + pgvector.

Hand-rolled against a literal, inspectable schema -- not LangChain's
PGVector class, which manages its own generic collection/embedding tables
with metadata stuffed into a JSONB column. BUILD_PLAN.md's whole point
for this phase is "metadata filtering becomes plain SQL" -- that's only
true, and only explainable in an interview, if the schema and the actual
queries are things you can read directly, not something a library
abstracts away.
"""

from __future__ import annotations

from typing import Any

import psycopg
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pgvector.psycopg import register_vector

from cpa.config import Settings, get_settings

# Tied to settings.embedding_model (all-MiniLM-L6-v2, 384 dimensions).
# Changing the embedding model means changing this and rebuilding the
# table -- a vector column's dimension is fixed at creation time.
EMBEDDING_DIM = 384

_SCHEMA_SQL = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id            SERIAL PRIMARY KEY,
    doc_slug      TEXT NOT NULL,
    title         TEXT,
    doc_type      TEXT NOT NULL,
    act_number    TEXT,
    year          INTEGER,
    section       TEXT,
    amends        TEXT,
    source_url    TEXT,
    content_role  TEXT,
    superseded_by TEXT,
    content       TEXT NOT NULL,
    embedding     vector({EMBEDDING_DIM}) NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
"""

# The only metadata keys similarity_search's `filter` is allowed to
# reference. Column names are interpolated directly into SQL text (values
# are always parameterized) -- this allowlist is what keeps that safe,
# since every caller in this codebase is trusted, but "trusted today"
# isn't the same guarantee as "can never be a column-name injection."
_FILTERABLE_COLUMNS = frozenset(
    {
        "doc_slug",
        "title",
        "doc_type",
        "act_number",
        "year",
        "section",
        "amends",
        "source_url",
        "content_role",
        "superseded_by",
    }
)

_SELECT_COLUMNS = (
    "doc_slug, title, doc_type, act_number, year, section, "
    "amends, source_url, content_role, superseded_by, content"
)


def _row_to_document(row: tuple[Any, ...]) -> Document:
    (
        doc_slug,
        title,
        doc_type,
        act_number,
        year,
        section,
        amends,
        source_url,
        content_role,
        superseded_by,
        content,
    ) = row
    metadata = {
        "slug": doc_slug,
        "title": title or "",
        "doc_type": doc_type,
        "act_number": act_number or "",
        "year": year if year is not None else "",
        "section": section or "",
        "amends": amends or "",
        "source_url": source_url or "",
        "content_role": content_role or "",
        "superseded_by": superseded_by or "",
    }
    return Document(page_content=content, metadata=metadata)


class PgRetriever:
    """Just enough surface for retrieval.py/chain.py to keep calling
    .invoke(question) exactly as they already do against a Chroma
    retriever -- so switching backends doesn't ripple into either file.
    """

    def __init__(self, store: "PgVectorStore", k: int, filter: dict[str, Any] | None):
        self._store = store
        self._k = k
        self._filter = filter

    def invoke(self, question: str) -> list[Document]:
        return self._store.similarity_search(question, k=self._k, filter=self._filter)


class PgVectorStore:
    """Vector store backed by the `chunks` table above."""

    def __init__(self, database_url: str, embeddings: Embeddings) -> None:
        self._database_url = database_url
        self._embeddings = embeddings
        self._ensure_schema()

    def _connect(self) -> psycopg.Connection:
        conn = psycopg.connect(self._database_url)
        register_vector(conn)
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(_SCHEMA_SQL)
            conn.commit()

    def reset(self) -> None:
        """Clear every row.

        Same "replace, don't append" guarantee as the Chroma version's
        reset_collection() call -- this is what keeps re-running ingestion
        from silently doubling the corpus, the real bug fixed in 1.8.
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE chunks")
            conn.commit()

    def count(self) -> int:
        """How many chunks are currently stored. Mirrors Chroma's
        `._collection.count()`, which tests and ingest/run.py both relied
        on -- kept as a real method here instead of reaching into
        implementation details that don't exist in this backend.
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chunks")
            (count,) = cur.fetchone()
            return count

    def add_documents(self, documents: list[Document]) -> None:
        if not documents:
            return

        vectors = self._embeddings.embed_documents(
            [doc.page_content for doc in documents]
        )
        rows = [
            (
                doc.metadata.get("slug", ""),
                doc.metadata.get("title") or None,
                doc.metadata.get("doc_type", ""),
                doc.metadata.get("act_number") or None,
                doc.metadata.get("year") or None,
                doc.metadata.get("section") or None,
                doc.metadata.get("amends") or None,
                doc.metadata.get("source_url") or None,
                doc.metadata.get("content_role") or None,
                doc.metadata.get("superseded_by") or None,
                doc.page_content,
                vector,
            )
            for doc, vector in zip(documents, vectors)
        ]

        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO chunks
                    (doc_slug, title, doc_type, act_number, year, section,
                     amends, source_url, content_role, superseded_by,
                     content, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                rows,
            )
            conn.commit()

    def similarity_search(
        self, query: str, k: int, filter: dict[str, Any] | None = None
    ) -> list[Document]:
        """Embed the query, then order chunks by pgvector's cosine-distance
        operator (<=>) -- this is BUILD_PLAN.md's "Act-number query becomes
        readable" point, literally: the filter is a plain SQL WHERE clause,
        not a vector store's own filter syntax.
        """
        query_vector = self._embeddings.embed_query(query)

        where_sql = ""
        params: list[Any] = []
        if filter:
            conditions = []
            for column, value in filter.items():
                if column not in _FILTERABLE_COLUMNS:
                    raise ValueError(f"Not a filterable column: {column!r}")
                conditions.append(f"{column} = %s")
                params.append(value)
            where_sql = "WHERE " + " AND ".join(conditions)

        params.append(query_vector)
        params.append(k)

        sql = f"""
            SELECT {_SELECT_COLUMNS}
            FROM chunks
            {where_sql}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        return [_row_to_document(row) for row in rows]

    def as_retriever(
        self, search_kwargs: dict[str, Any] | None = None, **_ignored: Any
    ) -> PgRetriever:
        search_kwargs = search_kwargs or {}
        return PgRetriever(
            self, k=search_kwargs.get("k", 5), filter=search_kwargs.get("filter")
        )


def build_vectorstore(
    chunks: list[Document],
    embeddings: Embeddings,
    settings: Settings | None = None,
) -> PgVectorStore:
    """Embed every chunk and write it to Postgres, replacing whatever was
    there.

    This is the ingestion job's job: a full rebuild from the current
    corpus, not an incremental append. Run it after any change to the
    corpus or to chunking.
    """
    settings = settings or get_settings()
    store = PgVectorStore(settings.database_url, embeddings)
    store.reset()
    store.add_documents(chunks)
    return store


def load_vectorstore(
    embeddings: Embeddings, settings: Settings | None = None
) -> PgVectorStore:
    """Open an already-built vector store without re-embedding anything.

    This is what the API/retrieval side calls -- it must never re-embed
    the corpus just to answer a question.
    """
    settings = settings or get_settings()
    return PgVectorStore(settings.database_url, embeddings)
