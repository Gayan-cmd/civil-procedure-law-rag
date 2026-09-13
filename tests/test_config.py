from pathlib import Path

from cpa.config import get_settings


def test_defaults_point_at_repo_corpus() -> None:
    settings = get_settings()
    assert settings.corpus_manifest_path.name == "manifest.json"
    assert settings.corpus_manifest_path.parent.name == "corpus"
    assert settings.chunk_size == 800
    assert settings.chunk_overlap == 100


def test_env_override(monkeypatch) -> None:
    monkeypatch.setenv("CPA_CHUNK_SIZE", "500")
    monkeypatch.setenv("CPA_CHROMA_DIR", "/tmp/custom_chroma")
    settings = get_settings()
    assert settings.chunk_size == 500
    assert settings.chroma_dir == Path("/tmp/custom_chroma")
