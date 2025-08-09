
import os, json
from fastapi.testclient import TestClient
import main
client = TestClient(main.app)

def test_metrics_ok():
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "python_info" in r.text

def test_auth_required_extract():
    pdf = b"%PDF-"
    r = client.post("/extract", files={"file": ("a.pdf", pdf, "application/pdf")}, data={"template": json.dumps({"name":"t","fields":["f"],"llm_text":"x"})})
    assert r.status_code == 401
