import dagshub
import mlflow
import mlflow.keras
from tensorflow.keras.models import load_model

# Initialisation de DagsHub avec l'intégration de MLflow
dagshub.init(repo_owner='mariamanadia',
             repo_name='Projet_Formation_MLOps_DataScientest_2025',
             mlflow=True)

# Définir l'URI du serveur MLflow si tu utilises un serveur local
mlflow.set_tracking_uri("http://localhost:5000")

# Charger ton modèle Keras
model = load_model("models/best_lstm_model.h5")

# Démarrer un run MLflow pour enregistrer les hyperparamètres, les métriques et le modèle
with mlflow.start_run(run_name="Log_Existing_LSTM_Model"):

    # Log des hyperparamètres (par exemple, la configuration de ton modèle)
    mlflow.log_param("max_words", 10000)
    mlflow.log_param("max_sequence_length", 10)

    # Log des métriques (par exemple, la précision d'entraînement)
    mlflow.log_metric("train_loss", 0.18)
    mlflow.log_metric("train_accuracy", 0.82)

    # Log des artefacts (par exemple, des fichiers associés au modèle)
    mlflow.log_artifact("models/tokenizer_config.json")
    mlflow.log_artifact("models/mapper.pkl")

    # Log du modèle Keras dans MLflow
    mlflow.keras.log_model(model, "lstm_model_logged")

    # Loguer le modèle DVC (si tu l'utilises avec DVC pour le versionning des modèles)
    mlflow.log_artifact("models/best_lstm_model.h5.dvc")

    print("Le modèle a été enregistré avec succès dans MLflow et DagsHub !")
