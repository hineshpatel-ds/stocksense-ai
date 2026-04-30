from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health_check_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["service"] == "stocksense-ai-api"


def test_validate_endpoint_with_sample_data():
    sample_path = Path("data/sample/sample_inventory.csv")

    assert sample_path.exists(), "Sample data file is missing. Run scripts/generate_sample_data.py"

    with sample_path.open("rb") as file:
        response = client.post(
            "/validate",
            files={"file": ("sample_inventory.csv", file, "text/csv")},
        )

    assert response.status_code == 200

    data = response.json()

    assert "is_valid" in data
    assert "data_quality_score" in data
    assert "row_count" in data


def test_analyze_endpoint_with_sample_data():
    sample_path = Path("data/sample/sample_inventory.csv")

    assert sample_path.exists(), "Sample data file is missing. Run scripts/generate_sample_data.py"

    with sample_path.open("rb") as file:
        response = client.post(
            "/analyze",
            files={"file": ("sample_inventory.csv", file, "text/csv")},
        )

    assert response.status_code == 200

    data = response.json()

    assert data["is_valid"] is True
    assert "summary_metrics" in data
    assert "forecast_summary" in data
    assert "recommendation_summary" in data


def test_ask_endpoint_with_sample_data():
    sample_path = Path("data/sample/sample_inventory.csv")

    assert sample_path.exists(), "Sample data file is missing. Run scripts/generate_sample_data.py"

    with sample_path.open("rb") as file:
        response = client.post(
            "/ask",
            data={"question": "Give me inventory summary"},
            files={"file": ("sample_inventory.csv", file, "text/csv")},
        )

    assert response.status_code == 200

    data = response.json()

    assert data["is_valid"] is True
    assert "answer" in data
    assert "intent" in data
    assert "confidence" in data


def test_validate_endpoint_rejects_unsupported_file():
    response = client.post(
        "/validate",
        files={"file": ("bad_file.txt", b"hello world", "text/plain")},
    )

    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]