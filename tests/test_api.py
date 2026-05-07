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

def test_save_upload_endpoint_with_sample_data(monkeypatch, tmp_path):
    test_db_path = tmp_path / "test_api_stocksense.db"
    monkeypatch.setenv("STOCKSENSE_DB_PATH", str(test_db_path))

    sample_path = Path("data/sample/sample_inventory.csv")

    assert sample_path.exists(), "Sample data file is missing. Run scripts/generate_sample_data.py"

    with sample_path.open("rb") as file:
        response = client.post(
            "/uploads/save",
            data={
                "company_name": "API Test Company",
                "industry": "General Inventory",
            },
            files={"file": ("sample_inventory.csv", file, "text/csv")},
        )

    assert response.status_code == 200

    data = response.json()

    assert data["is_saved"] is True
    assert data["is_valid"] is True
    assert "batch_id" in data
    assert data["company_name"] == "API Test Company"


def test_list_uploads_endpoint(monkeypatch, tmp_path):
    test_db_path = tmp_path / "test_api_stocksense.db"
    monkeypatch.setenv("STOCKSENSE_DB_PATH", str(test_db_path))

    sample_path = Path("data/sample/sample_inventory.csv")

    with sample_path.open("rb") as file:
        client.post(
            "/uploads/save",
            data={
                "company_name": "API Test Company",
                "industry": "General Inventory",
            },
            files={"file": ("sample_inventory.csv", file, "text/csv")},
        )

    response = client.get("/uploads")

    assert response.status_code == 200

    data = response.json()

    assert data["count"] >= 1
    assert len(data["batches"]) >= 1


def test_analyze_saved_upload_endpoint(monkeypatch, tmp_path):
    test_db_path = tmp_path / "test_api_stocksense.db"
    monkeypatch.setenv("STOCKSENSE_DB_PATH", str(test_db_path))

    sample_path = Path("data/sample/sample_inventory.csv")

    with sample_path.open("rb") as file:
        save_response = client.post(
            "/uploads/save",
            data={
                "company_name": "API Test Company",
                "industry": "General Inventory",
            },
            files={"file": ("sample_inventory.csv", file, "text/csv")},
        )

    batch_id = save_response.json()["batch_id"]

    response = client.get(f"/uploads/{batch_id}/analyze")

    assert response.status_code == 200

    data = response.json()

    assert data["is_valid"] is True
    assert data["batch_id"] == batch_id
    assert "summary_metrics" in data
    assert "forecast_summary" in data


def test_ask_agent_about_saved_upload_endpoint(monkeypatch, tmp_path):
    test_db_path = tmp_path / "test_api_stocksense.db"
    monkeypatch.setenv("STOCKSENSE_DB_PATH", str(test_db_path))

    sample_path = Path("data/sample/sample_inventory.csv")

    with sample_path.open("rb") as file:
        save_response = client.post(
            "/uploads/save",
            data={
                "company_name": "API Test Company",
                "industry": "General Inventory",
            },
            files={"file": ("sample_inventory.csv", file, "text/csv")},
        )

    batch_id = save_response.json()["batch_id"]

    response = client.post(
        f"/uploads/{batch_id}/ask",
        data={"question": "Give me inventory summary"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["is_valid"] is True
    assert data["batch_id"] == batch_id
    assert "answer" in data
    assert "intent" in data


def test_analyze_missing_batch_returns_404(monkeypatch, tmp_path):
    test_db_path = tmp_path / "test_api_stocksense.db"
    monkeypatch.setenv("STOCKSENSE_DB_PATH", str(test_db_path))

    response = client.get("/uploads/999999/analyze")

    assert response.status_code == 404