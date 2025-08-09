
import os, json, importlib
from fastapi.testclient import TestClient
import main, config

def _client():
    importlib.reload(config); importlib.reload(main)
    return TestClient(main.app)

def test_extract_missing_file():
    c = _client()
    r = c.post("/extract", headers={"x-api-key": os.environ["API_KEY"]},
               data={"template": json.dumps({"name":"t","fields":["a"],"llm_text":"x"})})
    assert r.status_code in (400, 422)

def test_unsupported_media_best_effort():
    c = _client()
    r = c.post("/extract", headers={"x-api-key": os.environ["API_KEY"]},
               files={"file": ("x.xyz", b"abc", "application/octet-stream")},
               data={"template": json.dumps({"name":"t","fields":["a"],"llm_text":"x"})})
    assert r.status_code == 200
