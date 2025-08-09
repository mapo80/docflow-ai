import json, time, os
from fastapi.testclient import TestClient
import main
client = TestClient(main.app)
API = {"x-api-key": os.environ["API_KEY"]}

def test_metrics_route():
    r = client.get("/metrics")
    assert r.status_code == 200
