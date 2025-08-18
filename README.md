# Phase 1
 
- **Définir les objectifs et métriques du projet**
- **Mettre en place l'environnement de développement avec Docker ou de simples environnements virtuels**
- **Créer les premiers pipelines de données et d'entraînement**
- **Mise en place Git pour le code et premiers tests automatisés**
- **Créer une API basique pour le modèle**
## Cadrage du Projet MLOps
Cataloguer les produits selon des données différentes (textes et images) est important pour les e-commerces puisque cela permet de réaliser des applications diverses telles que la recommandation de produits et la recherche personnalisée. Il s’agit alors de prédire le code type des produits à partir de données textuelles (désignation et description des produits) ainsi que des données d'images (image du produit).
Le client est le site internet de Rakuten, et plus particulièrement les administrateurs de ce site.
### Étapes pour cadrer le projet MLOps (Rakuten)
1. **Objectifs et Problématique**
   1. Pourquoi mettre en place un pipeline MLOps ? (automatisation, scalabilité, fiabilité…)
   2. Quels sont les objectifs ?
      1. Construire un modèle de classification d’images pour catégoriser des produits e-commerce.
      2. Déploiement du modèle deep learning en production.
   3. Quels sont les KPIs pour mesurer la performance du modèle et du pipeline MLOps ? (ex. : précision, latence, coût d’inférence…)
      4. Métriques :
         5. Accuracy : Évaluer la proportion de bonnes prédictions 
         6. F1-score : Prendre en compte le déséquilibre des classes 
         7. Temps d’inférence : Mesurer la rapidité des prédictions du modèle
2. **Données et Infrastructure**
   1. **Sources des données** : stockés dans HuggingFace: https://huggingface.co/datasets/ndiguesene/ml-datasets-image-rakuten-ecommerce/resolve/main/
      - Données textuelles (~60 MB)
      - Données images (~2.5 GB)
   2. **Stockage** : fichiers plats (CSV)
   3. **Nettoyage & Préparation** : Stratégie de gestion des données manquantes et transformation.
   4. **Infrastructure** : À définir.
3. **Développement du Modèle ML**
   1. **Choix des algorithmes** : Deep learning
   2. **Frameworks utilisés** : TensorFlow, Scikit-learn
4. **Industrialisation avec MLOps**
   1. **CI/CD pour le ML** : GitHub, GitHub Actions
   2. **Gestion des modèles** : MLflow
   3. **Orchestration** : Airflow
   4. **Monitoring & Observabilité** : Prometheus, Grafana
5. **Déploiement et Scalabilité**
   1. **Mode de déploiement** : Batch (REST API)
   2. **Infrastructure de déploiement** : Docker, Kubernetes (à confirmer)
   3. **Gestion du drift** : Retrain automatique, A/B testing, monitoring des performances
6. **Sécurité et Gouvernance**
   1. Sécurisé les APIs exposées
## Mettre en place l'environnement de développement reproductible
### Création de l'environnement 
#### - Sans Docker
Créez l'environnement virtuel avec les étapes suivantes sur Linux :
 
`python -m venv rakuten-project-mlops`
 
Cela va créer un dossier `rakuten-project-mlops/` contenant l’environnement virtuel.
 
### Installer les dépendances
 
Installez les dépendances nécessaires :
 
`pip install -r requirements.txt`
 
#### - Avec Docker
- Créer un Dockerfile et un `docker-compose.yml`
- Construire l’image avec `docker build -t ml_env .`
- Lancer le conteneur avec `docker-compose up`
### Importer les données dans `raw/data`
 
Ensuite, exécutez le script pour importer les données depuis le repo HuggingFace :
`python3 src/data/import_raw_data.py`
 
Arborescence des données
```
    data/raw/
    │── image_train/
    │   ├── image_123456789_product_987654321.jpg
    │   ├── image_987654321_product_123456789.jpg
    │   └── ...
    │── image_test/
    │   ├── image_111111111_product_222222222.jpg
    │   ├── image_222222222_product_111111111.jpg
    │   └── ...
```
---
## Copiez les données brutes dans `data/preprocessed/` :
 
`python3 src/data/make_dataset.py data/raw data/preprocessed`
 
## Entraînez les modèles sur l’ensemble de données et enregistrez-les dans le dossier models
 
`python3 python src/main.py`
## Démarrer les pratiques de versionning et des tests automatisés
### Suivi et gestion des versions des données et des modèles de données
1. **Initialiser DVC dans le projet**
```bash
  dvc init
```
2. **Ajouter les fichiers volumineux avec DVC**
```bash
    dvc add data/raw/image_train
    dvc add data/raw/image_test
    dvc add data/preprocessed/X_train_update.csv
    dvc add data/preprocessed/X_test_update.csv
```
3. **Créer un fichier `.gitignore` pour ignorer les fichiers volumineux suivis par DVC**
```bash
  echo "*.dvc" > .gitignore
```
### Configurer un stockage distant avec DVC
On utilisera **S3 d’AWS** pour sauvegarder les données. Pour cela, il faut d'abord installer S3 à l’aide de la commande suivante :

 
`pip install "dvc[S3]"`
 
 
Une fois S3 installé, il faudra mettre à jour le fichier de configuration `.dvc/config` pour ajouter DagsHub comme notre stockage distant.
---
## Implémenter une API basique

PYTHONPATH=$(pwd) pytest tests/test_api.py



---
## Implémenbtation d'une API basique


## Partie 1 : serving, dockerisation et tests unitaires
### Restructuration des répertoires et fichiers

#### data : build
Ici, nous regroupons les logiques de récupération des données depuis le dépôt distant. Les données structurées sont récupérées depuis AWS DataScientest. Les données d'images sont récupérées depuis HiggingFace.

#### train
Ici, nous entraînons le modèle (contient les fichiers main.py et train_model.py). Le répertoire contiendra un Dockerfile spécifique pour lancer l'entraînement du modèle à la demande. Les résultats seront enregistrés dans `models` à la racine

#### predict 
Ici, on applique le modèle au données d'entraînement (contient le fichier predict.py). Le répertoire contiendra un Dockerfile spécifique pour lancer le test du modèle sur les données d'entraînement; Les résultats seront enregistrés sur data/predictions à la racine

#### serve
Ici, on utilisera dans un premier temps requests ou FastAPI pour créer une API qui va interroger le modèle avec des données préalablement renseignées (fichier serve.py à créer) et retournera des résultats que l'on pourra mettre dans data/predictions à la racine avec un nom de fichier spécifique
Niveaux de serving : 
1. Usage de FastAPI pour créer un contneur basique de serving
2. Usage de BentoML pour automatiser le déploiement, serving etc

#### Variables d'environnement : vous devez rajouter un fichier .env sur la racine du projet
Les chemins des modèles (poids, mapper etc) ne sont plus codés en dur. Les modifications suivantes ont été apportées : 
1. ajout fichier `.env` contenant les variables d'environnement (**ce fichier ne doit pas être poussé sur le git, il peut renfermer des informations sensibles tels clés d'api etc**)
2. ajout fichier `.env.example` sur le git : fichier d'exemple montrant comment renseigner le fichier .env ( quelles sont les variables d'environnement etc)

## Conteneurisation
Docker est utilisé pour conteneuriser les différents services de l'application. Chaque service dispose d'un répertoire spécifique sur les sources ainsi qu'un Dockerfile spécifique.

### Variables d'environnement
Le fichier `.env` à la racine du projet, doit être mis en jour avec les répertoires de volume créés dans le conteneur (Exemple : `TOKENIZER_CONFIG_PATH=/app/models/tokenizer_config.json`).

### Services applicatifs
1. Service de données
Ici, le service effectue le téléchargement des données depuis le dépôt HuggingFace : `https://huggingface.co/datasets/ndiguesene/ml-datasets-image-rakuten-ecommerce/resolve/main/`.

`Package` : src/data/build_data/

2. Service d'entraînement

`Package` : src/models/train
Contient les fichiers : 
- .dockerignore : fichier précisant les fichiers et répertoires à ignorer lors de la construction de l'image
- Dockerfile : commandes pour créer l'image Docker pour ce service uniquement
- main.py : script principal contenant la logique d'entrainement global 
- train_model.py  : script contenant la logique d'entrainement des modèles
- up.sh : script permettant de lancer un conteneur docker pour ce service uniquement. A lancer depuis la racine du projet.
- requirements.txt : contient les dépendences requis pour créer l'image 

3. Service de test

Ici, nous utilisons les données d'entraînement pour tester l'usage du modèle.
`Package` : src/models/predict
Contient les fichiers : 
- .dockerignore : fichier précisant les fichiers et répertoires à ignorer lors de la construction de l'image
- Dockerfile : commandes pour créer l'image Docker pour ce service uniquement
- predic.py : script principal contenant la logique de test du modèle en utilisant le modèle entraîné et sauvegardé sur le service d'entraînement
- up.sh : script permettant de lancer un conteneur docker pour ce service uniquement. A lancer depuis la racine du projet.
- requirements.txt : contient les dépendences requis pour créer l'image 

4. Service de serving

Ici, nous créons une application permettant d'intérroger le modèle via API en utilisant FastAPI.
`Package` : src/models/serve
Contient les fichiers : 
- .dockerignore : fichier précisant les fichiers et répertoires à ignorer lors de la construction de l'image
- Dockerfile : commandes pour créer l'image Docker pour ce service uniquement
- predic_logic.py : wrapper créé pour contenir les fonctions requis pour effectuer les prédictions (traitement de l'image, traitement des données textuelles)
- serve_model_fastapi.py : script principal contenant l'application FastAPI (les endpoints de prédiction)
- up.sh : script permettant de lancer un conteneur docker pour ce service uniquement. A lancer depuis la racine du projet.
- requirements.txt : contient les dépendences requis pour créer l'image

### API de prédiction
Le service de prédiction est composé pour le moment de deux endpoints : 
- `/status` : pour obtenir le statut de l'application. Retourne un message si l'application est up.
- `/predict` :  pour obtenir une classifiction d'un code produit en fonction des informations suivantes : </br>
`product_identifier`: str = Form(...), # Un identifiant entier pour le produit. Cet identifiant est utilisé pour associer le produit à son code de type de produit correspondant. </br>
`designation`: str = Form(...), # Le titre du produit, un court texte résumant le produit </br>
`description`: str = Form(...),  # Un texte plus détaillé décrivant le produit. Tous les commerçants n'utilisent pas ce champ, donc, afin de conserver l’originalité des données, le champ description peut contenir des valeurs NaN pour de nombreux produits. </br>
`product_id`: str = Form (...), # Un identifiant unique pour ce produit </br>
`imageid`: str = Form (...), # Un identifinat unique de l'image associé à ce produit. </br>
`image`: UploadFile = File(...)  # L'image correspondant au produit </br>

![Endpoint /predict](https://github.com/ndiguesene/Projet_Formation_MLOps_DataScientest_2025/blob/awa/restructure_folders/reports/predict_endpoint_input.png)
![Retour /predict](https://github.com/ndiguesene/Projet_Formation_MLOps_DataScientest_2025/blob/awa/restructure_folders/reports/predict_endpoint_return.png)

### Services : sécurisation et logging

##### Découplage en services
Chaque fonctionnalité du système (téléchargement et traitement des données, entraînement du modèle concaténé, test du modèle entraîné et serving) est défini en service respectivement situés dans les répertoires : src/data, src/models/train, src/models/predict, src/models/serve.
Chaque service est utilisable de façon découplé via des conteneurs Docker ainsi que tous les packages requis pour son bon fonctionnement.

##### Limitation taux de requête
La limitation du débit permet d'éviter les abus et de garantir une utilisation équitable de l'API.
La limite de 10 requêtes par minutes sur /predict est appliquée.
Un gestionnaire d'exception global pour RateLimitExceeded garantit que les utilisateurs reçoivent un message d'erreur clair lorsque les limites sont dépassées.

##### Authentification et authorisation
Nous avons mis en œuvre OAuth2 avec JWT pour sécuriser les points de terminaison. Les clés d'authenication sont tous sauvegardés comme variables d'environnement.
Le point de terminaison `/token` génère des jetons d'accès, et la dépendance `get_current_user` garantit que seuls les utilisateurs authentifiés peuvent accéder aux points de terminaison protégés comme `/predict`.
La protection des points de terminaison sensibles comme `/predict` garantit que seuls les utilisateurs autorisés peuvent accéder aux modèles. 
La logique de sécurité est séparée dans security_logic.py, ce qui rend la base de code plus modulaire et réutilisable.
Dans cette version, les utilisateurs ne sont pas dans une base de données.
- `security_logic.py` : wrapper contenant les logiques d'authentification et d'authorisation utilisés par le script principal.

##### Logging
Nous avons mis en place un intergiciel qui enregistre chaque demande avec un identifiant unique, une méthode, une URL, un code d'état et un temps de traitement.
La journalisation est essentielle pour le débogage, la surveillance et l'audit.
L'inclusion des identifiants des requêtes et des temps de traitement facilite le suivi et l'optimisation des performances. 

- `import_data_logger.log`
- `train_model_logger.log`
- `test_model_logger.log`
- `serving_logger.log`

Nous avons inclu un (RotateFileHandler) pour gérer la limitation de la taille des fichiers de logs.

##### Chargement des modèles au démarrage
Nous avons rajouté une méthode au démarrage de l'application de prédiction pour charger tous les modèles et fichiers de configuration pour éviter leur chargement à chaque demande de prédiction.

##### Orchestration
Le fichier `docker-compose.xml` à la racine du projet est utilisé pour orchestrer tous les services du projet. Nous utilisons le fichier .env pour assurer la réusabilité des différents environnemenst (production, staging etc) et la modularité.

##### Technologies utilisées
1. Docker : utilisé pour créer chaque service de l'application en un conteneur indépendant
2. Logging et FileHandler : nous utilisons un logging centralisé pour chaque service (téléchargement et traitement des données, entraînement du modèle concaténé, test du modèle entraîné et serving).

Chaque service retourne un fichier log spécifique disponible dans le répertoire logs/ à la racine du projet.
3. Slowapi : utilisé pour limiter le taux de requête sur les routes.
4. Oauth2 et JWT : sécurisation des routes d'API en terme d'authentification et d'authorisation.
5. bcrypt : encryption des informations utilisateurs (mots de passe) pour comparaison et validation des informations

##### Usage : 

- Demander un token : `/token`
`curl -X POST "http://127.0.0.1:8000/token" -d "username=user1&password=43SoYourAreADataEngineer34"`
Exemple retours : 
{
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "token_type": "bearer"
}
- Utiliser le token pour intérroger l'API de prédiction : `/predict_code`

![Endpoint /predict](https://github.com/ndiguesene/Projet_Formation_MLOps_DataScientest_2025/blob/securing_apis/reports/securing_api_authorized.png)

---------

## Tests Unitaires
## Documentation des tests unitaires

Cette partie de la documentation décrit les tests unitaires définis pour l’API FastAPI. Ils couvrent les trois endpoints principaux : racine (`/`), `GET /status` et `POST /predict`.

---

## Prérequis

- Python 3.9
- FastAPI
- Uvicorn (pour lancer l’API si besoin)
- pytest
- httpx (installé automatiquement avec FastAPI)

Installer les dépendances :

```bash
  pip install fastapi uvicorn pytest httpx
```

## Lancer les tests

Depuis la racine du projet (là où se trouve src/ et tests/), exécuter :

```bash
   pytest tests/test_api.py
```
## Structure des tests

Le fichier tests/test_api.py contient trois fonctions :
```python
from fastapi.testclient import TestClient
from Projet_Formation_MLOps_DataScientest_2025.api import app

client = TestClient(app)
```
Chaque test utilise un TestClient pour simuler des requêtes HTTP vers l’application FastAPI.

1. **test_status**
   2. Objectif : vérifier que l’endpoint GET /status renvoie bien un code HTTP 200 et le message attendu. 
   3. script
      ```python
      def test_status():
          response = client.get("/status")
          assert response.status_code == 200
          assert response.json() == {"status": "L'API fonctionne correctement"}
      ```
   3. Requête : 
   ```http request
   GET /status
   ```
   4. Assertions :
      - `response.status_code == 200`
      - `response.json() == {"status": "L'API fonctionne correctement"}`

   2. **test_root**
      3. Objectif : valider que la racine / renvoie un message de bienvenue.
      4. script
      ```python
      def test_root():
          response = client.get("/")
          assert response.status_code == 200
          data = response.json()
          assert "message" in data
          assert "Bienvenue" in data["message"]
         ```
      4. Requête :
      ```http request
      GET /
      ```
      5. Assertions
         - `response.status_code == 200`
         - `Le corps JSON contient une clé message`
         - `La chaîne "Bienvenue" apparaît dans la valeur de message`

3. **test_predict**
   4. Objectif : s’assurer que l’endpoint POST /predict accepte un payload JSON arbitraire et retourne un JSON avec les clés predicted et label.
   5. script
   ```python
   def test_predict():
       sample_text = {"predicted": "1", "label": "label_1"}
       response = client.post("/predict", json=sample_text)
       assert response.status_code == 200
       data = response.json()
       assert "predicted" in data
       assert "label" in data
   ```
   5. Requête :
   ```http request
   POST /predict
   Content-Type: application/json
   {"predicted": "1", "label": "label_1"}
   ```
   6. Assertions :
    - response.status_code == 200 
    - La réponse JSON contient les champs predicted et label

---------

## Automatisation DVC/DgasHub/MLFlow

A partir de là, nous travaillons exclusvement avec Docker.
Donc, il faut obligatoirement disposer d'un fichier .env à la racine du projet. Exemple de contenu .env disponible sur le fichier `.env.example``.

1. Après avoir cloner le code, créer un fichier .env (ne pas oublier de mettre à jour vos informations dagshub (token))
```bash
touch .env

```

2. Données nécessaires : fichiers artefacts non dispoinibles sur Git
Les fichiers .h5, .json, .pkl (best_lstm_model, best_vgg16, best_weights, mapper, tokenizer) doivent être disponibles dans le dossier `models`.
Ils ont utilisé dans l'entraînement des modèles.
A la récupération du projet, si ces fichiers ne sont pas disponibles, faire une commande : 

```bash
dvc pull
```

3. Exécuter la pipeline sans Airflow (tests uniquement)

```bash
docker compose up

```

!! Le fichier à la racine `run_compose_options.sh` contient des examples de commandes pour exécuter chaque service en particulier.

4. Exécuter la pipeline via Airflow en mode standalone

Lancer une instance airflow en local (juste pour des tets) : mettre à jour le fichier sous airflow/dags/initialize_airflow.sh et exécuter le fichier

```bash
export AIRFLOW_HOME=`répertoire absolu du projet`/airflow  
airflow db migrate

# Démarrer airflow en mode standalone (crée un utilisateur admin automatiquement, les credentials sont enregsitrés sous : airflow/simple_auth_manager_passwords.json.generated )
airflow standalone

```

5. Exécuter la pipeline via Airflow et Docker
Dans cette partie du projet, nous entrons dans l'industrialisation.  Voir le point suivant.

---------

### Architecture MLOps
Pour automatiser l'exécution des tâches end-to-end, nous mettons en contribution Airflow(gestion de l'ordonnancement et exécution automatique des tâches), Docker(encapsulation des logiques avec toutes les dépendances) et DVC(gestionnaire du versionnage des données).

#### Motivations
Dans une pipeline de MLOps comme celui qui nous concerne, il est bien de s'assurer de certains aspects : 
- les données utilisées à chaque exécution du modèle sont connus et récupérables, ceci rejoint l'aspect reproductibilité
- la version entraînée du modèle est connu, suavegardé et récupérable, toujours sur l'aspect reproductibilité
- les artefacts du modèle correspondant à cette version d'exécution sont enregistrés et récupérables
- la version générale du code est enregistrée et récupérable
- les exécutions sont agnostiques de la plateforme utilisée

#### Outils

- Airlfow nous permet de gérer l'ordonnancement et l'exécution automatique des différentes tâches requises pour entraîner le modèle
- MLfow nous permet de sauvegarder le modèle, les métriques ainsi que les artefacts produits après chaque entraînement
- DVC nous permet d'assurer la sauvegarde et le versionning des données utilisées lors de l'entraînement, le modèle produit, les fichiers artefacts produits
- Docker nous permet d'assurer l'encapsulation de touts les prérequis à l'exécution de tous les stages de notre pipeline d'entraînement
- Git nous permet d'assurer la sauvegarde de tous les fichiers de code requis pour la bonne exécution et la création des images et conteneurs Docker

#### Configurations

##### Fichier .env
Le fichier .env est un fichier très important pour le projet se trouvant sur la racine. Il contient toutes les variables de configuration nécessaires pour la bonne exécution de la pipeline MLFlow.
Il contient les liens vers les fichiers requis à l'entraînement du modèle (poids, tokenizer etc). Il contient également, les répertoires d'enregitrement des données ainsi que du modèle.
Ces derniers sont utilisés notamment par DVC pour le versionning des données.
Il contient également les secret pour le module d'authentification.
Ce fichier n'est pas versionné mais un fichier exemple (`.enf.example`) est disponible sur le répertoire distant.

##### DVC et MLflow
Nous utilisons DVC via DagsHub S3 en plus du tracking distant MLflow lié à DagsHub.
Les commandes de configuration DVC à appliquer en local est disponible sur l'interface DagsHub.

Les configurations suivantes doivent être renseignées sur le fichier .env
`MLFLOW_TRACKING_URI` : correspond au lien distant MLflow lié à DagsHub.
`MLFLOW_EXPERIMENT_NAME` : par ProductCodeClassifier
`DAGSHUB_USERNAME`: Nom d'utilisateur de votre compte DagsHub créé
`DAGSHUB_TOKEN` : Votre token DagsHub

Ici, nous démarrons la connexion à MLflow distant (src/models/train/main.py) :

```python
with mlflow.start_run(run_name="Train_Concatenate_Model") as run:
  mlflow.set_tag("source", "airflow")
  mlflow.set_tag("version", run_time)
  mlflow.set_tag("model_type", "VGG16+LSTM")
```

Après entraînement, les artefacts et le modèle sont sauvegardés :
```python
 # Log artefacts
  mlflow.log_artifact(tokenizer_config_path)
  mlflow.log_artifact(mapper_path_pkl)
  mlflow.log_artifact( best_weights_path_pkl)
  mlflow.log_artifact(BEST_WEIGHTS_PATH)
  mlflow.log_artifact(MAPPER_PATH)

  # Log the models
  mlflow.keras.log_model(lstm, "lstm_model")
  mlflow.keras.log_model(vgg16, "vgg16_model")
  mlflow.keras.log_model(concatenate_model, "concatenate_model")
```

##### Docker
Comme présenté plus haut, les différents stages de notre pipeline sont dockerisés. Chaque stage dispose d'un Dockerfile permettant de créer automatiquement une image.

##### Airflow
Nous utilisons Ariflow sous Docker, nous récupérons donc le docker-compose officiel dédié sur le site de Airflow (https://airflow.apache.org/docs/apache-airflow/3.0.4/docker-compose.yaml).
Ce fichier a été mis à jour et est disponible à la racine du projet : `docker-compose.yml`.
Il a été mis à jour suivant : 
- les volumes : nous lions le projet en local aux différents conteneurs démarrés par Ariflow pour permettre la disponbilité des logiques et fichiers de code (` - ./:/opt/airflow/mlops-project `) nécessaires à la bonne exécution des stages en particulier des logiques pour créer les images des stages (Dockerfile).
- l'utilisateur admin airflow

L'exécution de  ce fichier produit les conteneurs/services suivants :  
- la base de  données PostgreSQL
- la base de données Redis
- le serveur d'api de Airflow
- le scheduler de Airflow
- le dag processor de Ariflow
- un worker
- un Triggerer

#### Stages de la pipeline MLflow

- Définition 
La définition de la pipeline sous forme de dag Airflow est disponible sous `airflow/dags/ml_pipeline.py`.
Les opérateurs Docker : `BashOperator` et `DockerOperator` ont été utilisés.

Les différents stages de notre pipeline ont été regroupés et se présentent comme suit : 

![Pipeline global Airflow](https://github.com/ndiguesene/Projet_Formation_MLOps_DataScientest_2025/blob/automating_dvc_mlflow_docker/reports/pipeline.png)

1. Récupération des données : 3 opérateurs Airflow
- construction de l'image Docker contenant la logique de récupération des données
- création et lancement du conteneur de récupération des données
- enregistrement des données récupérées sur DVC

2. Entraînement du modèle : 3 opérateurs Airflow
- création de l'image Docker contenant la logique d'entraînement du modèle
- création et lancement du conteneur d'entraînement du modèle - enregistrement du modèle et des artéfacts sur MLflow
- enregistrement du modèle et des fichiers sur DVC

3. Test du modèle : 3 opérateurs Airflow
- création de l'image de test
- création et lancement du conteneur de test
- enregistrement des prédictions résultats sur DVC

4. Service d'exposition de l'API : 4 opérateurs Airflow
- création image pour l'authentification
- création et lancement du conteneur d'authentification : ce conteneur s'exécute en backgroud après lancement
- création de l'image d'exposition de l'API
- création et lancement du conteneur d'exposition de l'API 

#### Lancement
- Premier lancement : initialise les bases Postgres SQL et Redis
```bash
make init-airflow
```
- Lancement : 
```bash
make start
```

![Pipeline global Airflow](https://github.com/ndiguesene/Projet_Formation_MLOps_DataScientest_2025/blob/automating_dvc_mlflow_docker/reports/detailled_pipeline.png)

#### Ports ouverts requis par la pipeline

Le service d'authentification est ouvert sur le port 8011:8011.
Le service d'API est ouvert sur le port 8000:8000.

