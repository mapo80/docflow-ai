
import json
from fastapi.testclient import TestClient
import main
client = TestClient(main.app)

def test_extract_missing_file():
    r = client.post("/extract", headers={"x-api-key": "test-key"}, data={"template": json.dumps({"name":"t","fields":["a"],"llm_text":"x"})})
    assert r.status_code in (400, 422)

def test_extract_unsupported_media():
    r = client.post("/extract", headers={"x-api-key": "test-key"}, files={"file": ("x.xyz", b"abc", "application/octet-stream")}, data={"template": json.dumps({"name":"t","fields":["a"],"llm_text":"x"})})
    # should still respond 200 with best-effort pipeline (naive decode)
    assert r.status_code == 200
