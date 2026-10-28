from fastapi.testclient import TestClient
from app.main import app, feature_names

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True

def test_features():
    response = client.get("/features")
    assert response.status_code == 200
    assert response.json()["count"] == 30

def test_predict_valid_payload():
    # Typical numeric values; used only to verify API shape, not for clinical interpretation.
    payload = {"features": {name: 1.0 for name in feature_names}}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in ("benign", "malignant")
    assert set(body["probabilities"]) == {"benign", "malignant"}

def test_predict_missing_feature_rejected():
    response = client.post("/predict", json={"features": {"mean radius": 10.0}})
    assert response.status_code == 422
