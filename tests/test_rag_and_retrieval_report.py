import os, json
from fastapi.testclient import TestClient
import main
client = TestClient(main.app)
API = {"x-api-key": os.environ["API_KEY"]}

def test_rag_trigger_and_report(monkeypatch):
    # Force small context to trigger RAG
    os.environ["LLM_N_CTX"] = "512"
    os.environ["RAG_TOPK"] = "3"

    # Fake PP returns empty (not used here)
    from clients import ppstructure_client as ppc
    async def fake_pp(data, filename, pages=None):
        return []
    monkeypatch.setattr(ppc, "analyze_async", fake_pp)

    # Build a long PDF to exceed context
    md = "\n\n".join([f"# Sezione {i}\nContenuto con IBAN e CF e altre info {i}" for i in range(200)])
    # encode as simple text file (will go through markitdown fallback), but we can embed in PDF quickly
    import fitz, io
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72,72), md[:2000])
    pdf = doc.tobytes(); doc.close()

    tpl = {"name":"fattura_long","fields":["iban","cf"], "llm_text":"estrai i campi richiesti"}
    r = client.post("/extract", headers=API, files={"file": ("doc.pdf", pdf, "application/pdf")}, data={"template": json.dumps(tpl)})
    assert r.status_code == 200
    js = r.json()
    rid = js["request_id"]
    rep = client.get(f"/reports/{rid}", headers=API).json()
    # In manifest, context mode should be rag_field_wise
    assert rep["manifest"]["llm_context_mode"] == "rag_field_wise"
    # Each field should have retrieval info
    for f in rep["fields"].values():
        assert "retrieval" in f and isinstance(f["retrieval"], list)
