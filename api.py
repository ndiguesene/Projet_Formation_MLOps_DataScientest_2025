# src/main.py

from fastapi import FastAPI

app = FastAPI()

@app.get("/status")
def status():
    return {"message": "API is working"}

@app.post("/predict")
def predict(data: dict):
    # Prédiction avec ton modèle ici
    return {"prediction": "result"}
