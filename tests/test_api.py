from fastapi.testclient import TestClient
from main import api

client = TestClient(api)

def test_status():
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"status": "L'API fonctionne correctement"}

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Bienvenue" in response.json()["message"]

def test_predict():
    sample_text = {"text": "exemple de texte"}  # adapte ce texte si besoin
    response = client.post("/predict", json=sample_text)
    assert response.status_code == 200
    assert "predicted_class" in response.json()
    assert "class_label" in response.json()

