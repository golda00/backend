"""
Smoke tests for the FastAPI application.
Run with: pytest tests/test_health.py -v
"""
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_redirects_to_docs():
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (301, 302, 307, 308)


def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "models_loaded" in data


def test_submit_job_requires_pdfs(tmp_path):
    """Uploading a non-PDF should return 422."""
    fake_file = tmp_path / "not_a_pdf.txt"
    fake_file.write_text("hello")

    with fake_file.open("rb") as f:
        response = client.post(
            "/api/v1/jobs",
            files={
                "before_pdf": ("not_a_pdf.txt", f, "text/plain"),
                "after_pdf":  ("not_a_pdf.txt", f, "text/plain"),
            },
        )
    assert response.status_code == 422


def test_get_unknown_job_returns_404():
    response = client.get("/api/v1/jobs/nonexistent-job-id")
    assert response.status_code == 404
