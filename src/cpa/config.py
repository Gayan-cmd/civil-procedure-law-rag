"""Settings for the cpa package, read from environment variables.

No other module in this package should read a file path, model name, or
tuning constant directly -- everything comes from here, so the same code
runs unmodified on a laptop, in CI, and in production.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# src/cpa/config.py -> src/cpa -> src -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


@dataclass(frozen=True)
class Settings:
    corpus_manifest_path: Path = field(
        default_factory=lambda: _env_path(
            "CPA_CORPUS_MANIFEST", _REPO_ROOT / "corpus" / "manifest.json"
        )
    )
    corpus_root: Path = field(
        default_factory=lambda: _env_path("CPA_CORPUS_ROOT", _REPO_ROOT)
    )
    chroma_dir: Path = field(
        default_factory=lambda: _env_path("CPA_CHROMA_DIR", _REPO_ROOT / "chroma_db")
    )

    embedding_model: str = field(
        default_factory=lambda: os.environ.get(
            "CPA_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    llm_model: str = field(
        default_factory=lambda: os.environ.get(
            "CPA_LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"
        )
    )

    chunk_size: int = field(default_factory=lambda: _env_int("CPA_CHUNK_SIZE", 800))
    chunk_overlap: int = field(
        default_factory=lambda: _env_int("CPA_CHUNK_OVERLAP", 100)
    )

    retrieval_k: int = field(default_factory=lambda: _env_int("CPA_RETRIEVAL_K", 5))


def get_settings() -> Settings:
    """Build a fresh Settings from the current environment.

    A function rather than a module-level singleton so tests can monkeypatch
    os.environ and get a Settings that reflects it, without import order
    mattering.
    """
    return Settings()
