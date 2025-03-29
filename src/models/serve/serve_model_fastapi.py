from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from typing import Optional
from PIL import Image
import numpy as np
import pandas as pd
import io
import tensorflow as tf
from io import BytesIO
from predict_logic import make_prediction, preprocess_image


app = FastAPI()

class PredictionResponse(BaseModel):
    code: str # product code predicted
    confidence: float # confidence level
    message: str # response message
    prediction: str # prediction result

@app.get("/status")
def status():
    return {"message": "Model Serving via FastAPI is running !"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_code(
    text: str = Form(...),  # Receive text as form data
    image: UploadFile = File(...)  # Receive image as file)
):
    # Process the image input as bytes
    image_data = await image.read()
    processed_image = preprocess_image(image_data)

    # Convert to TensorFlow tensor (optional, for model input)
    #image_tensor = tf.convert_to_tensor([processed_image], dtype=tf.float32)
    
    # construct a df with textual data including description
    df = pd.DataFrame({"description": [text]})

    predictions = make_prediction(df, processed_image)

    # logique de prédiction : 
    return {"prediction": predictions["prediction"], "code": predictions["label"], "confidence": predictions["confidence"], "message": "Prediction successful"}