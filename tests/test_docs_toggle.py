import importlib
from fastapi.testclient import TestClient


def test_docs_toggle(monkeypatch):
    # Ensure docs are enabled by default
    monkeypatch.setenv("DOCS_ENABLED", "1")
    import config, main
    importlib.reload(config)
    importlib.reload(main)
    client = TestClient(main.app)
    assert client.get("/docs").status_code == 200

    # Disable docs and reload
    monkeypatch.setenv("DOCS_ENABLED", "0")
    importlib.reload(config)
    importlib.reload(main)
    client = TestClient(main.app)
    assert client.get("/docs").status_code == 404

    # Restore for other tests
    monkeypatch.setenv("DOCS_ENABLED", "1")
    importlib.reload(config)
    importlib.reload(main)

