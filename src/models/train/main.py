from src.features.build_features import DataImporter, TextPreprocessor, ImagePreprocessor
from train_model import TextLSTMModel, ImageVGG16Model, concatenate
from tensorflow import keras
import pickle
import tensorflow as tf
import os
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler
from logging import Formatter, getLogger

load_dotenv()
# Load paths from environment variables
tokenizer_config_path = os.environ.get("TOKENIZER_CONFIG_PATH", "../../../models/tokenizer_config.json")
lstm_model_path = os.environ.get("LSTM_MODEL_PATH", "../../../models/best_lstm_model.h5")
vgg16_model_path = os.environ.get("VGG16_MODEL_PATH", "../../../models/best_vgg16_model.h5")
mapper_path_pkl = os.environ.get("MAPPER_PATH_PKL", "mapper.pkl")
best_weights_path_pkl = os.environ.get("BEST_WEIGHTS_PATH_PKL", "../../../models/best_weights.pkl")
data_path = os.environ.get("DATA_PATH", "../../../data/raw")
images_path = os.environ.get("IMAGES_PATH", "../../../data/raw/image_train")
CONCATENATED_MODEL_PATH = os.environ.get("CONCATENATED_MODEL_PATH", "../../../models/concatenate.h5")

logger = logging.getLogger(__name__)
log_file_path=os.environ.get("TRAIN_MODEL_LOGGER_PATH", "../../../logs/train_model_logger.log")
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
fileHandler = RotatingFileHandler(log_file_path, maxBytes=5 * 1024 * 1024, backupCount=3)
    
formatter = Formatter(
    "%(asctime)s [%(levelname)s] %(message)s"
)
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)
logger.setLevel(logging.INFO)


data_importer = DataImporter(filepath=data_path, mapper_path=mapper_path_pkl)
df = data_importer.load_data()
X_train, X_val, _, y_train, y_val, _ = data_importer.split_train_test(df)
 
# Preprocess text and images
text_preprocessor = TextPreprocessor()
image_preprocessor = ImagePreprocessor(filepath=images_path)
text_preprocessor.preprocess_text_in_df(X_train, columns=["description"])
text_preprocessor.preprocess_text_in_df(X_val, columns=["description"])
image_preprocessor.preprocess_images_in_df(X_train)
image_preprocessor.preprocess_images_in_df(X_val)
 
# Train LSTM model
logger.info("Training LSTM Model")
text_lstm_model = TextLSTMModel()
text_lstm_model.preprocess_and_fit(X_train, y_train, X_val, y_val)
logger.info("Finished training LSTM")
 
logger.info("Training VGG")
# Train VGG16 model
image_vgg16_model = ImageVGG16Model()
image_vgg16_model.preprocess_and_fit(X_train, y_train, X_val, y_val)
logger.info("Finished training VGG")

with open(tokenizer_config_path, "r", encoding="utf-8") as json_file: # Load the tokenizer configuration : "/app/models/tokenizer_config.jso
    tokenizer_config = json_file.read()
tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_config)
lstm = keras.models.load_model(lstm_model_path)
vgg16 = keras.models.load_model(vgg16_model_path)

logger.info("Training the concatenate model")
model_concatenate = concatenate(tokenizer, lstm, vgg16)
lstm_proba, vgg16_proba, new_y_train = model_concatenate.predict(X_train, y_train)
best_weights = model_concatenate.optimize(lstm_proba, vgg16_proba, new_y_train)
logger.info("Finished training concatenate model")
 
with open(best_weights_path_pkl, "wb") as file:
    pickle.dump(best_weights, file)
 
num_classes = 27
 
proba_lstm = keras.layers.Input(shape=(num_classes,))
proba_vgg16 = keras.layers.Input(shape=(num_classes,))
 
weighted_proba = keras.layers.Lambda(
    lambda x: best_weights[0] * x[0] + best_weights[1] * x[1]
)([proba_lstm, proba_vgg16])
 
concatenate_model = keras.models.Model(
    inputs=[proba_lstm, proba_vgg16], outputs=weighted_proba
)
 
# Enregistrer le modèle au format h5
concatenate_model.save(CONCATENATED_MODEL_PATH)
logger.info("Finished saving concatenate model")
