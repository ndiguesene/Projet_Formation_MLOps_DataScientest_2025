from fastapi import FastAPI, File, UploadFile, Form, Request, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Optional
from PIL import Image
import json
from tensorflow import keras
import numpy as np
import pandas as pd
import io
import tensorflow as tf
from io import BytesIO
from src.models.serve.predict_logic import make_prediction, preprocess_image
from src.models.serve.auth_utils import verify_token, get_token
from datetime import timedelta
import logging
from logging.handlers import RotatingFileHandler
from logging import Formatter, getLogger
import os
from dotenv import load_dotenv
from uuid import uuid4
import time
from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi import FastAPI, Request, Form
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse
from slowapi.middleware import SlowAPIMiddleware

app = FastAPI()

class SubPrediction(BaseModel):
    label: str
    confidence: float
    prediction: int

class PredictionResponse(BaseModel):
    predictions: list[SubPrediction] # prediction result
    message: str # response message

# Securing API 1 : logging handler and formatter
# Load environment variables from .env file
    load_dotenv()

logger = logging.getLogger(__name__)
log_file_path=os.environ.get("SERVING_LOGGER_PATH", "../../../logs/serving_logger.log")
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
fileHandler = RotatingFileHandler(log_file_path, maxBytes=5 * 1024 * 1024, backupCount=3)
    
formatter = Formatter(
    "%(asctime)s [%(levelname)s] %(message)s"
)
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)
logger.setLevel(logging.INFO)

@app.on_event("startup")
async def load_models():
    global model
    global tokenizer
    global lstm
    global vgg16
    global best_weights
    global mapper
    
    # Load environment variables from .env file
    load_dotenv()

    # Load paths from environment variables
    model_path = os.environ.get("CONCATENATED_MODEL_PATH", "../../../models/concatenate.h5")
    #model = tf.keras.models.load_model(model_path)
    # Load tokenizer and model configurations
    tokenizer_config_path = os.environ.get("TOKENIZER_CONFIG_PATH", "../../../models/tokenizer_config.json")
    lstm_model_path = os.environ.get("LSTM_MODEL_PATH", "../../../models/best_lstm_model.h5")
    vgg16_model_path = os.environ.get("VGG16_MODEL_PATH", "../../../models/best_vgg16_model.h5")
    best_weights_path = os.environ.get("BEST_WEIGHTS_PATH", "../../../models/best_weights.json")
    mapper_path = os.environ.get("MAPPER_PATH", "../../../models/mapper.json")

    # Charger les configurations et modèles
    with open(tokenizer_config_path, "r", encoding="utf-8") as json_file:
        tokenizer_config = json_file.read()
    tokenizer = keras.preprocessing.text.tokenizer_from_json(tokenizer_config)

    lstm = keras.models.load_model(lstm_model_path)
    vgg16 = keras.models.load_model(vgg16_model_path)

    with open(best_weights_path, "r") as json_file:
        best_weights = json.load(json_file)

    with open(mapper_path, "r") as json_file:
        mapper = json.load(json_file)

    #model = tf.keras.models.load_model(model_path)
    logger.info("Models and files loaded successfully.")

# Securing API 2 : Add a global exception handler for rate limiting
# Initialize the Limiter
limiter = Limiter(key_func=get_remote_address)
@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"message": "Rate limit exceeded. Please try again later."},
    )
# Global exception handler for unhandled exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "An internal server error occurred."},
    )

# Apply rate limiting to the /status endpoint
@app.get("/status")
@limiter.limit("5/minute")  # Allow 5 requests per minute per client
def get_status(request: Request):
    """
    Secured endpoint to check the status of the service.
    Requires a valid token for access.
    """
    return {"message": "Model Serving via FastAPI is running !"}

# Securing API 3 : Token endpoint
@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    token = get_token(form_data.username, form_data.password)
    return token


# Apply rate limiting to the /predict endpoint
@app.post("/predict", response_model=PredictionResponse)
@limiter.limit("10/minute")  # Allow 10 requests per minute per client
async def predict_code(
    request: Request,
    product_identifier: str = Form(...), # An integer ID for the product. This ID is used to associate the product with its corresponding product type code.
    designation: str = Form(...), # The product title, a short text summarizing the product.
    description: str = Form(...),  # A more detailed text describing the product. Not all the merchants use this field, so to retain originality of the data, the description field can contain NaN value for many products.
    product_id: str = Form (...), # An unique ID for the product.
    imageid: str = Form (...), # An unique ID for the image associated with the product.
    image: UploadFile = File(...),  # Receive image as file
    token: str = Depends(verify_token),  # Secure the endpoint using the token
):
     
    current_user: dict = token # Secure the endpoint
     # Log the authenticated user
    logger.info(f"Prediction requested by user: {current_user['username']}")
    
    # Process the image input as bytes
    image_data = await image.read()
    processed_image = preprocess_image(image_data)
    
    # construct a df with textual data including description
    df = pd.DataFrame({"product_identifier":[product_identifier], "description": [description], "designation": [designation], "product_id": [product_id], "imageid": [imageid]})

    # handling potential errors : model loading, image processing, prediction
    try:
        predictions = make_prediction(df, processed_image, tokenizer, lstm, vgg16, best_weights, mapper)
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred during prediction.",
        )

    # Convert the list of dictionaries to a list of SubPrediction objects
    sub_predictions = [SubPrediction(**prediction) for prediction in predictions]

    # endpoint return value : {"prediction": predictions["prediction"], "code": predictions["label"], "confidence": predictions["confidence"], "message": "Prediction successful"}
    return {
        "predictions": sub_predictions,
        "message": "Prediction successful"
    }

@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Generate a unique request ID for tracking
    request_id = str(uuid4())
    
    # Log the start of the request
    logger.info(f"Request ID={request_id} - Start request: {request.method} {request.url}")
    
    # Record the start time

    start_time = time.time()
    
    # Process the request
    response = await call_next(request)
    
    # Calculate the processing time
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # Log the end of the request
    logger.info(
        f"Request ID={request_id} - Completed request: {request.method} {request.url} "
        f"Status code={response.status_code} - Process time={process_time:.4f}s"
    )
    
    return response