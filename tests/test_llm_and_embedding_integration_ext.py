import os, json, importlib
from fastapi.testclient import TestClient
import json, importlib, os
from fastapi.testclient import TestClient
import main, config, retriever
from _pdfutils import make_pdf_text
import clients


def _tpl(fields, txt="Estrai i campi con attenzione"):
    return {"name": "fattura", "fields": fields, "llm_text": txt}


def _client():
    importlib.reload(config)
    importlib.reload(retriever)
    importlib.reload(main)
    return TestClient(main.app)


def test_llm_and_embedding_integration(monkeypatch):
    os.environ["MOCK_LLM"] = "0"
    os.environ["LLM_N_CTX"] = "256"
    os.environ["RAG_MIN_SEGMENTS"] = "1"

    called = {"emb": 0, "llm": 0}

    def fake_embed(texts):
        called["emb"] += 1
        return [[0.0, 0.0, 0.0] for _ in texts]

    monkeypatch.setattr(clients, "llm_embed", fake_embed, raising=True)

    import clients.llm_local as _llm_local

    def fake_chat_json(fields, llm_text, context):
        called["llm"] += 1
        return {"iban": {"value": "IT00A", "confidence": 0.9}}

    monkeypatch.setattr(_llm_local, "chat_json", fake_chat_json, raising=True)

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
    fields = data["fields"]
    if isinstance(fields, list):
        val = fields[0]["value"]
    else:
        val = fields.get("iban", {}).get("value", "")
    assert val.startswith("IT00")
