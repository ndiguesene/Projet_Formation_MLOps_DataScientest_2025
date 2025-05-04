import mlflow
import dvc.api
from tensorflow.keras.models import load_model
import tempfile
import os

# Config MLflow (peut être adapté selon ton setup)
mlflow.set_tracking_uri("http://localhost:5000")  # ou un URI distant
mlflow.set_experiment("model_tracking_experiment")

# Récupération du modèle suivi par DVC (à la bonne révision)
model_path = "models/best_lstm_model.h5"
revision = "f11c5a5"  # le commit git correspondant, tu peux automatiser ça si besoin

with dvc.api.open(model_path, mode='rb', rev=revision) as fd:
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        tmp_file.write(fd.read())
        temp_model_path = tmp_file.name

# Chargement du modèle Keras
model = load_model(temp_model_path)

# Log dans MLflow
with mlflow.start_run():
    mlflow.tensorflow.log_model(model, artifact_path="model")

    # Exemple de log de paramètres et métriques
    mlflow.log_param("model_type", "LSTM")
    mlflow.log_param("revision", revision)
    mlflow.log_metric("example_accuracy", 0.83)  

# Nettoyage optionnel du fichier temporaire
os.remove(temp_model_path)
