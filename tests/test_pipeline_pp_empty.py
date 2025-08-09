
import os, json, importlib
from fastapi.testclient import TestClient
import main, config

def test_image_pp_returns_empty_still_ok(monkeypatch):
    os.environ["PPSTRUCT_POLICY"]="auto"
    import clients.ppstructure_client as ppc
    async def fake(data, filename, pages=None): return []
    monkeypatch.setattr(ppc, "analyze_async", fake)

    importlib.reload(config); importlib.reload(main)
    c = TestClient(main.app)
    img = b"\x89PNG\r\n\x1a\n" + b"0"*64
    tpl = {"name":"t","fields":["iban","totale"],"llm_text":"estrai"}
    r = c.post("/extract", headers={"x-api-key": os.environ["API_KEY"]},
               files={"file": ("x.png", img, "image/png")},
               data={"template": json.dumps(tpl)})
    assert r.status_code == 200
    js = r.json()
    assert "fields" in js and "request_id" in js
