import os, json, importlib
from fastapi.testclient import TestClient
import main, config, retriever
from _pdfutils import make_pdf_text
import numpy as np


def _tpl(fields, txt="Estrai i campi con attenzione"):
    return {"name": "fattura", "fields": fields, "llm_text": txt}


def _client():
    importlib.reload(config)
    importlib.reload(retriever)
    importlib.reload(main)
    return TestClient(main.app)


def test_llm_and_embedding_integration(monkeypatch):
    os.environ["MOCK_LLM"] = "0"
    os.environ["EMBEDDING_BACKEND"] = "st"
    os.environ["LLM_N_CTX"] = "256"
    os.environ["RAG_MIN_SEGMENTS"] = "1"

    called = {"emb": 0, "llm": 0}

    class FakeST:
        def __init__(self, name):
            pass
        def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
            called["emb"] += 1
            return np.zeros((len(texts), 3), dtype="float32")

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", FakeST)

    import llm as _llm
    async def fake_chat_json_async(messages, max_tokens=1024):
        called["llm"] += 1
        return {"choices": [{"message": {"content": json.dumps({"iban": {"value": "IT00A", "confidence": 0.9}})}}]}
    monkeypatch.setattr(_llm, "chat_json_async", fake_chat_json_async)

    c = _client()
    pdf = make_pdf_text(1, "IBAN: IT00A")
    r = c.post(
        "/extract",
        headers={"x-api-key": os.environ["API_KEY"]},
        files={"file": ("doc.pdf", pdf, "application/pdf")},
        data={"template": json.dumps(_tpl(["iban"]))},
    )
    assert r.status_code == 200
    data = r.json()
    assert called["emb"] > 0 and called["llm"] > 0
    assert data["fields"][0]["value"].startswith("IT00")
