from src.features.build_features import DataImporter, TextPreprocessor, ImagePreprocessor
from train_model import TextLSTMModel, ImageVGG16Model, concatenate
from tensorflow import keras
import pickle
import tensorflow as tf
import os
from dotenv import load_dotenv


load_dotenv()
# Load paths from environment variables
tokenizer_config_path = os.environ.get("TOKENIZER_CONFIG_PATH", "../../../models/tokenizer_config.json")
lstm_model_path = os.environ.get("LSTM_MODEL_PATH", "../../../models/best_lstm_model.h5")
vgg16_model_path = os.environ.get("VGG16_MODEL_PATH", "../../../models/best_vgg16_model.h5")
best_weights_path_pkl = os.environ.get("BEST_WEIGHTS_PATH_PKL", "../../../models/best_weights.pkl")
data_path = os.environ.get("DATA_PATH", "../../../data/raw")
images_path = os.environ.get("IMAGES_PATH", "../../../data/raw/image_train")
CONCATENATED_MODEL_PATH = os.environ.get("CONCATENATED_MODEL_PATH", "../../../models/concatenate.h5")

data_importer = DataImporter(filepath=data_path)
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
print("Training LSTM Model")
text_lstm_model = TextLSTMModel()
text_lstm_model.preprocess_and_fit(X_train, y_train, X_val, y_val)
print("Finished training LSTM")
 
print("Training VGG")
# Train VGG16 model
image_vgg16_model = ImageVGG16Model()
image_vgg16_model.preprocess_and_fit(X_train, y_train, X_val, y_val)
print("Finished training VGG")
<<<<<<< HEAD:src/main.py
 
with open("models/tokenizer_config.json", "r", encoding="utf-8") as json_file:
    tokenizer_config = json_file.read()
tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_config)
lstm = keras.models.load_model("models/best_lstm_model.h5")
vgg16 = keras.models.load_model("models/best_vgg16_model.h5")
 
=======

with open(tokenizer_config_path, "r", encoding="utf-8") as json_file: # Load the tokenizer configuration : "/app/models/tokenizer_config.jso
    tokenizer_config = json_file.read()
tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_config)
lstm = keras.models.load_model(lstm_model_path)
vgg16 = keras.models.load_model(vgg16_model_path)

>>>>>>> origin/awa/restructure_folders:src/models/train/main.py
print("Training the concatenate model")
model_concatenate = concatenate(tokenizer, lstm, vgg16)
lstm_proba, vgg16_proba, new_y_train = model_concatenate.predict(X_train, y_train)
best_weights = model_concatenate.optimize(lstm_proba, vgg16_proba, new_y_train)
print("Finished training concatenate model")
<<<<<<< HEAD:src/main.py
 
with open("models/best_weights.pkl", "wb") as file:
=======

with open(best_weights_path_pkl, "wb") as file:
>>>>>>> origin/awa/restructure_folders:src/models/train/main.py
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
<<<<<<< HEAD:src/main.py
concatenate_model.save("models/concatenate.h5")
=======
concatenate_model.save(CONCATENATED_MODEL_PATH)
>>>>>>> origin/awa/restructure_folders:src/models/train/main.py
