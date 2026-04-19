import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "version": "2.0-static"}

def test_stats():
    response = client.get("/api/stats")
    # Even if DB is empty or mock, it should return 200
    assert response.status_code == 200

def test_drugs():
    response = client.get("/api/drugs?limit=5")
    assert response.status_code == 200
    assert "drugs" in response.json()

def test_proteins():
    response = client.get("/api/proteins?limit=5")
    assert response.status_code == 200
    assert "proteins" in response.json()
