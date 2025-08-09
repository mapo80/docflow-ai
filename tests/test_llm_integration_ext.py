
import os, json, importlib
from fastapi.testclient import TestClient
import main, config
from _pdfutils import make_pdf_text

def _tpl(fields, txt="Estrai i campi con attenzione"):
    return {"name":"fattura","fields":fields, "llm_text": txt}

def _client():
    importlib.reload(config); importlib.reload(main)
    return TestClient(main.app)

def test_llm_json_clean_and_codeblock(monkeypatch):
    # Force non-mock path by patching chat_json_async directly on llm module
    import llm as _llm
    async def fake_chat_json_async(messages, max_tokens=1024):
        # For single_pass we expect both fields to be returned in one call
        return {"choices":[{"message":{"content": json.dumps({
            "iban":{"value":"IT00A","confidence":0.9},
            "totale":{"value":"123,45","confidence":0.8}
        })}}]}
        return {"choices":[{"message":{"content": "```json\n{\n \"totale\":{\"value\":\"123,45\",\"confidence\":0.8}\n}\n```"}}]}
    monkeypatch.setattr(_llm, "chat_json_async", fake_chat_json_async)

    os.environ["MOCK_LLM"]="0"
    os.environ["LLM_N_CTX"]="1024"
    os.environ["RAG_MIN_SEGMENTS"]="12"   # prefer single_pass on small docs

    c = _client()
    pdf = make_pdf_text(2, "doc breve per single pass")
    r = c.post("/extract", headers={"x-api-key": os.environ["API_KEY"]},
               files={"file": ("sp.pdf", pdf, "application/pdf")},
               data={"template": json.dumps(_tpl(["iban","totale"]))})
    assert r.status_code == 200
    data = r.json()
    vals = {f["key"]: f["value"] for f in data["fields"]}
    assert vals.get("iban","").startswith("IT00") and vals.get("totale","").startswith("123")

def test_llm_rag_fieldwise_long_doc(monkeypatch):
    # Force RAG by shrinking context and segments
    os.environ["MOCK_LLM"]="0"
    os.environ["LLM_N_CTX"]="256"
    os.environ["RAG_MIN_SEGMENTS"]="1"

    import llm as _llm
    async def fake_chat_json_async(messages, max_tokens=1024):
        user = next(m for m in messages if m.get("role")=="user")["content"]
        if "iban" in user:
            payload = {"iban":{"value":"IT99X","confidence":0.8}}
        else:
            payload = {"totale":{"value":"999,00","confidence":0.65}}
        return {"choices":[{"message":{"content": json.dumps(payload)}}]}
    monkeypatch.setattr(_llm, "chat_json_async", fake_chat_json_async)

    c = _client()
    longtext = " ".join(f"token{i}" for i in range(5000))
    pdf = make_pdf_text(3, longtext)
    r = c.post("/extract", headers={"x-api-key": os.environ["API_KEY"]},
               files={"file": ("rag.pdf", pdf, "application/pdf")},
               data={"template": json.dumps(_tpl(["iban","totale"]))})
    assert r.status_code == 200
    rep = c.get(f"/reports/{r.json()['request_id']}", headers={"x-api-key": os.environ["API_KEY"]}).json()
    assert rep["manifest"]["llm_context_mode"] == "rag_field_wise"
