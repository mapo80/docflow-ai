import json, pytest
from fastapi.testclient import TestClient
import main

client = TestClient(main.app)

def test_auth_required_jobs():
    r = client.post("/jobs", files={"file": ("t.pdf", b"PDF")}, data={"template": json.dumps({"name":"t","fields":["a"],"llm_text":"x"}), "priority":5})
    assert r.status_code == 401
