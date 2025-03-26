import os
import requests
import logging
import zipfile

# Liste des fichiers à télécharger
FILES = [
    "X_test_update.csv",
    "X_train_update.csv",
    "Y_train_CVw08PX.csv",
    "images_low.zip",
]

# URL de base pour télécharger les fichiers
BASE_URL = "https://huggingface.co/datasets/ndiguesene/ml-datasets-image-rakuten-ecommerce/resolve/main/"

# Dossier où stocker les fichiers téléchargés
RAW_DATA_PATH = "./data/raw"

# Configurer le logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def download_file(url, output_path):
    """Télécharge un fichier depuis une URL et l'enregistre à output_path."""
    if os.path.exists(output_path):
        logger.info(f"Le fichier {output_path} existe déjà, téléchargement ignoré.")
        return

    logger.info(f"Téléchargement de {url} vers {output_path}...")
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with open(output_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1 Mo
                if chunk:
                    file.write(chunk)
        logger.info(f"Téléchargement terminé : {output_path}")
    else:
        logger.error(f"Erreur lors du téléchargement ({response.status_code}): {url}")

def unzip_file(zip_path, extract_to):
    """Décompresse un fichier ZIP vers un dossier donné."""
    extract_folder = os.path.splitext(zip_path)[0]  # Ex: ./data/raw/images_low
    if os.path.exists(extract_folder):
        logger.info(f"Le dossier {extract_folder} existe déjà, extraction ignorée.")
        return

    logger.info(f"Décompression de {zip_path} vers {extract_to}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    logger.info(f"Décompression terminée.")

def main():
    """Télécharge les fichiers CSV et ZIP, puis décompresse le fichier ZIP."""
    os.makedirs(RAW_DATA_PATH, exist_ok=True)  # Créer le dossier si nécessaire

    for filename in FILES:
        file_url = BASE_URL + filename
        file_path = os.path.join(RAW_DATA_PATH, filename)

        # Télécharger chaque fichier
        download_file(file_url, file_path)

        # Décompresser le ZIP après téléchargement
        if filename.endswith(".zip"):
            unzip_file(file_path, RAW_DATA_PATH)

if __name__ == "__main__":
    main()