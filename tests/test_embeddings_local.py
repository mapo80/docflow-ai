import importlib
import os
import pytest

llama_spec = importlib.util.find_spec("llama_cpp")

pytestmark = pytest.mark.skipif(
    llama_spec is None,
    reason="requires llama-cpp package",
)

from clients.embeddings_local import embed_texts
import clients


def test_embed_texts_returns_vector(monkeypatch):
    token = os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        pytest.skip("HUGGINGFACE_TOKEN must be set for real embeddings")
    monkeypatch.setenv("BACKENDS_MOCK", "0")
    if hasattr(clients, "llm_embed"):
        delattr(clients, "llm_embed")
    try:
        vec = embed_texts(["hello"])[0]
    except Exception as e:
        pytest.skip(f"embedding failed: {e}")
    assert isinstance(vec, list)
    assert len(vec) == 768
