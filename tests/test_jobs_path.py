
import os, json, importlib, time
from fastapi.testclient import TestClient
import main, config

def test_extract_job_mode_happy_path():
    importlib.reload(config); importlib.reload(main)
    c = TestClient(main.app)
    pdf = b"%PDF-"
    tpl = {"name":"t","fields":["x"],"llm_text":"x"}
    r = c.post("/extract?mode=job", headers={"x-api-key": os.environ["API_KEY"]},
               files={"file": ("a.pdf", pdf, "application/pdf")},
               data={"template": json.dumps(tpl)})
    assert r.status_code in (200, 202)
    js = r.json()
    assert "request_id" in js
