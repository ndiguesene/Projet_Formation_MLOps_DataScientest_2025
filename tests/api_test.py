import os
from fastapi.testclient import TestClient
from src.models.serve.serve_model_fastapi import app
from src.models.serve.serve_model_fastapi import verify_token

client = TestClient(app)


def test_status():
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"message": "Model Serving via FastAPI is running !"}

#def test_root():
#    response = client.get("/")
#    assert response.status_code == 200
#    data = response.json()
#    assert "message" in data
#    assert "Bienvenue" in data["message"]

def test_predict(monkeypatch):
    # Mock verify_token dependency to bypass authentication
    def mock_verify_token():
        return {"username": "testuser"}
    app.dependency_overrides[verify_token] = mock_verify_token

    # Prepare a dummy file
    file_content = b"dummy image data"
    files = {"image": ("test.jpg", file_content, "image/jpeg")}
    data = {
        "product_identifier": "id1",
        "designation": "test",
        "description": "desc",
        "product_id": "pid",
        "imageid": "imgid"
    }
    response = client.post("/predict", data=data, files=files)
    resp_json = response.json
    assert response.status_code == 200
    assert "predictions" in response.json()
    assert resp_json["message"] == "Prediction successful"

    # Clean up dependency override
    app.dependency_overrides = {}
