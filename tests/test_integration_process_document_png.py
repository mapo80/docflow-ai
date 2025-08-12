import os
import time
import httpx
import subprocess
from pathlib import Path

SAMPLE_IMG = Path("dataset/sample_invoice.png")


def wait_for_server(url: str, timeout: int = 60):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = httpx.get(url, timeout=1.0)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("server did not start in time")


def test_process_document_png_integration(tmp_path):
    env = os.environ.copy()
    env.setdefault("MOCK_LLM", "1")
    env.setdefault("BACKENDS_MOCK", "0")
    token = env.get("HUGGINGFACE_TOKEN")
    assert token, "HUGGINGFACE_TOKEN must be set"

    port = "8009"
    proc = subprocess.Popen(
        ["uvicorn", "main:app", "--port", port],
        env=env,
        cwd=str(Path(__file__).resolve().parents[1]),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        wait_for_server(f"http://127.0.0.1:{port}/")
        with SAMPLE_IMG.open("rb") as f:
            files = {"file": ("sample_invoice.png", f, "image/png")}
            data = {"pp_policy": "auto", "overlays": "true"}
            headers = {"x-api-key": env.get("API_KEY", "")}
            r = httpx.post(
                f"http://127.0.0.1:{port}/process-document",
                files=files,
                data=data,
                headers=headers,
                timeout=300.0,
            )
        assert r.status_code == 200
        j = r.json()
        assert j.get("status") == "done"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
