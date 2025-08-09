import json, os
from fastapi.testclient import TestClient
import main

client = TestClient(main.app)
API = {"x-api-key": os.environ["API_KEY"]}

def test_extract_sync():
    tpl={"name":"fattura","fields":["iban","cf"], "llm_text":"trova i campi nel documento"}
    r = client.post("/extract", headers=API,
        files={"file": ("doc.pdf", b"%PDF-1.7\n", "application/pdf")},
        data={"template": json.dumps(tpl)})
    assert r.status_code == 200
    js = r.json()
    assert "request_id" in js and "fields" in js
    # forensic report should exist
    rid = js["request_id"]
    r2 = client.get(f"/reports/{rid}", headers=API)
    assert r2.status_code == 200
