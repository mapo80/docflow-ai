import json, time
from fastapi.testclient import TestClient
import main, os

client = TestClient(main.app)
API = {"x-api-key": os.environ["API_KEY"]}

def test_submit_and_sse_backlog():
    files={"file": ("x.pdf", b"%PDF-1.7\n", "application/pdf")}
    data={"template": json.dumps({"name":"t","fields":["iban","cf"], "llm_text":"estrai i campi"}) , "priority":"5"}
    r = client.post("/jobs", headers=API, files=files, data=data)
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    # Let worker run
    time.sleep(0.5)
    ev = client.get(f"/jobs/{job_id}/events", headers=API)
    assert ev.status_code == 200
    # Should contain at least queued+started+done in backlog
    text = ev.text
    assert "event: queued" in text
    assert "event: started" in text
    assert "event: done" in text
