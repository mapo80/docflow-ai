
import os, json, importlib
from fastapi.testclient import TestClient
import main, config

def test_metrics_and_emit_toggle():
    os.environ["EMIT_EVENTS"]="1"
    importlib.reload(config); importlib.reload(main)
    c = TestClient(main.app)
    # Simple request
    pdf = b"%PDF-"
    tpl = {"name":"t","fields":["f"],"llm_text":"x"}
    r = c.post("/extract", headers={"x-api-key": os.environ["API_KEY"]},
               files={"file": ("a.pdf", pdf, "application/pdf")},
               data={"template": json.dumps(tpl)})
    assert r.status_code == 200
    # metrics endpoint available
    m = c.get("/metrics")
    assert m.status_code == 200 and "python_info" in m.text
