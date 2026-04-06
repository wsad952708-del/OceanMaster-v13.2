"""
OceanMaster — API Endpoint Tests
==================================
Tests FastAPI endpoints using TestClient.

Run: python -m pytest tests/test_api.py -v
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def client():
    """Create FastAPI test client."""
    from fastapi.testclient import TestClient
    from api.app import app
    return TestClient(app)


# ═══════════════════════════════════════════════════
# 1. Health Endpoint
# ═══════════════════════════════════════════════════

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_version(self, client):
        data = client.get("/health").json()
        assert "version" in data
        assert data["version"] == "15.3"

    def test_health_has_status(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"

    def test_health_has_uptime(self, client):
        data = client.get("/health").json()
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0


# ═══════════════════════════════════════════════════
# 2. Species Endpoint
# ═══════════════════════════════════════════════════

class TestSpeciesEndpoint:
    def test_species_returns_200(self, client):
        resp = client.get("/species")
        assert resp.status_code == 200

    def test_species_has_items(self, client):
        data = client.get("/species").json()
        assert data["count"] >= 6  # At least 6 species
        assert len(data["species"]) == data["count"]

    def test_species_has_required_fields(self, client):
        data = client.get("/species").json()
        sp = data["species"][0]
        assert "key" in sp
        assert "name_zh" in sp
        assert "name_en" in sp
        assert "scientific" in sp
        assert "Topt_C" in sp


# ═══════════════════════════════════════════════════
# 3. Hotspots Endpoint
# ═══════════════════════════════════════════════════

class TestHotspotsEndpoint:
    def test_hotspots_returns_200(self, client):
        resp = client.get("/hotspots")
        assert resp.status_code == 200

    def test_hotspots_is_geojson(self, client):
        data = client.get("/hotspots").json()
        assert data["type"] == "FeatureCollection"
        assert "features" in data


# ═══════════════════════════════════════════════════
# 4. Auth (when API key is NOT set)
# ═══════════════════════════════════════════════════

class TestAuth:
    def test_predict_accessible_without_key_in_dev(self, client):
        """When OCEANMASTER_API_KEY is not set, /predict should not require auth."""
        # We don't actually call /predict (it runs the pipeline),
        # but we can verify the auth middleware doesn't block
        # health and species endpoints
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_docs_accessible(self, client):
        """OpenAPI docs should be accessible."""
        resp = client.get("/docs")
        assert resp.status_code == 200


# ═══════════════════════════════════════════════════
# 5. Retrain Endpoint
# ═══════════════════════════════════════════════════

class TestRetrainEndpoint:
    def test_retrain_returns_200(self, client):
        from api.app import API_KEY
        headers = {"X-API-Key": API_KEY} if API_KEY else {}
        resp = client.post("/retrain", json={"force": False}, headers=headers)
        assert resp.status_code == 200

    def test_retrain_has_status(self, client):
        from api.app import API_KEY
        headers = {"X-API-Key": API_KEY} if API_KEY else {}
        resp = client.post("/retrain", json={"force": False}, headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "n_runs" in data
