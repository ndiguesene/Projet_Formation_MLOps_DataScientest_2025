- [Contexte et Objectifs](#contexte-et-objectifs)
  - [Objectifs et métriques](#objectifs-et-métriques)
- [Données et Infrastructure](#données-et-infrastructure)
- [Vous êtes préssés ? Démarrez rapidement ici](#vous-%C3%AAtes-pr%C3%A9ss%C3%A9s--d%C3%A9marrez-rapidement-ici)
  - [Fichier .env](#fichier-env)
  - [Dépendances](#dépendances)
  - [Configurer un stockage distant avec DVC](#configurer-un-stockage-distant-avec-dvc)
  - [Récupérer les artefacts et modèles et données déjà existants](#récupérer-les-artefacts-et-modèles-et-données-déjà-existants)
  - [Lancer la pipeline Airflow](#lancer-la-pipeline-airflow)
  - [Comment tester les endpoints](#comment-tester-les-endpoints)
- [Structure du projet](#structure-du-projet)
  - [Structure globale](#structure-globale)
  - [Struture du code des stages : data, train, auth, predict](#struture-du-code-des-stages--data-train-auth-predict)
- [Détails d'implémentation](#détails-dimplémentation)
  - [Serving, dockerisation et tests unitaires](#serving-dockerisation-et-tests-unitaires)
  - [Conteneurisation](#conteneurisation)
  - [API de prédiction](#api-de-prédiction)
  - [Services : sécurisation et logging](#services--sécurisation-et-logging)
- [Tests Unitaires](#tests-unitaires)
  - [Documentation des tests unitaires](#documentation-des-tests-unitaires)
- [Automatisation Airflow/DVC/DagsHub/MLFlow](#automatisation-airflowdvcdagshubmlflow)
- [Monitoring](#monitoring)

# Contexte et Objectifs
 
## Objectifs et métriques
Ce projet ayant pour objectifs de cataloguer les produits selon des données différentes (textes et images) est important pour les e-commerces puisque cela permet de réaliser des applications diverses telles que la recommandation de produits et la recherche personnalisée. Il s’agit alors de prédire le code type des produits à partir de données textuelles (désignation et description des produits) ainsi que des données d'images (image du produit). Construire un modèle de classification d’images pour catégoriser des produits e-commerce.
Le client est le site internet de Rakuten, et plus particulièrement les administrateurs de ce site.
Nous mettons en place une pipeline MLOps pour permettre l'automatisation, la scalabilité, et la fiabilité des modèles mis en place, et ce, dans le long terme.

Dans ce projet, nous entraînons un modèle multimodal utilisant deux modèles `VGG16` et `LSTM` déjà existants et versionnés sur DasgHub. Il vous faudra donc les récupérer.

Les métriques suivantes sont programmées pour mesurer la performance du modèle et du pipeline MLOps:
         5. Accuracy : Évaluer la proportion de bonnes prédictions 
         6. F1-score : Prendre en compte le déséquilibre des classes 
         7. Temps d’inférence : Mesurer la rapidité des prédictions du modèle

## Données et Infrastructure
1. **Sources des données** : stockés dans HuggingFace: https://huggingface.co/datasets/ndiguesene/ml-datasets-image-rakuten-ecommerce/resolve/main/
      - Données textuelles (~60 MB)
      - Données images (~2.5 GB)
   2. **Stockage** : fichiers plats (CSV)
   3. **Nettoyage & Préparation** : Stratégie de gestion des données manquantes et transformation.
   4. **Infrastructure** : l'infrastructure de déploiement mis en place se base entièrement et exclusivement sur Apache Airflow et Docker.
3. **Développement du Modèle ML**
   1. **Choix des algorithmes** : Deep learning
   2. **Frameworks utilisés** : TensorFlow, Scikit-learn
4. **Industrialisation avec MLOps**
   1. **Gestion du versionnement des modèles** : MLflow
   2. **Versionnage des données, des métriques et artefacts** : DVC et DagsHub 
   3. **Orchestration** : Airflow
   4. **Monitoring & Observabilité** : Prometheus, Grafana
5. **Déploiement et Scalabilité**
   1. **Mode de déploiement** : Batch (REST API)
   2. **Infrastructure de déploiement** : Docker et Airflow
   3. **Gestion du drift** : Non encore implémenté
6. **Sécurité et Gouvernance**
   1. Routes d'API sécurisées via l'usage du Json Web Token (JWT)


---------

# Vous êtes préssés ? Démarrez rapidement ici
L'usage exclusif de Docker sur ce projet nous permet d'assurer la reproductibilité. Il est donc primordiale que vous ayez un Docker Engine up.
Il vous est, tout de même, conseillé de créer un environnement virtuel Python lorsque vous souhaitez lancer ce projet.
Créez l'environnement virtuel (exemple : `rakuten-project-mlops`) avec les étapes suivantes sur Linux :
 
```bash
python -m venv rakuten-project-mlops
```

Cela va créer un dossier `rakuten-project-mlops/` contenant l’environnement virtuel.

Ensuite utilisez cet environnement pour toutes les commandes futures.

### Fichier .env
Le projet dispose d'un fichier `.env` non versionné contenant les informations nécessaires à son bon fonctionnement. Vous pouvez vous inspirer du fichier `.env.example` pour créer votre fichier `.env`.
Les informations requises :

```
TOKENIZER_CONFIG_PATH=`Répertoire et nom du fichier artefact tokenizer format json`
LSTM_MODEL_PATH=`Répertoire et nom du modèle pré-entraîné lstm format h5` 
VGG16_MODEL_PATH=`Répertoire et nom du modèle pré-entraîné image VGG16 format h5`
BEST_WEIGHTS_PATH=`Répertoire et nom du fichier d'artefacts des poids précalculés format json` 
BEST_WEIGHTS_PATH_PKL=`Répertoire et nom du fichier d'artefacts des poids précalculés format pickle` 
MAPPER_PATH=`Répertoire et nom du fichier d'artefacts mapper sous format json` 
MAPPER_PATH_PKL=`Répertoire et nom du fichier d'artefacts mapper sous format pickle`
DATASET_PATH=`Répertoire d'enregistrement des données textuelles d'entraînement de test, doit être dans $DATA_PATH`
DATA_PATH=`Répertoire d'enregistrement des données d'entraînement`
IMAGES_PATH=`Répertoire d'enregistrement des images d'entraînement de test, doit être dans $DATA_PATH`
PREDICTIONS_PATH=`Répertoire d'enregistrement des prédictions test effectués après entrainement`
CONCATENATED_MODEL_PATH=`Répertoire d'enregistrement du modèle concaténé`
SERVING_LOGGER_PATH=`Répertoire et nom du fichier de logs de l'API dans le conteneur`
SECRET_KEY=`Clé secrete utilisée pour le hachage`
ALGORITHM=`Algorithme de hachage pour génération toke JWT, défaut HS256`
ACCESS_TOKEN_EXPIRE_MINUTES=`Délai d'expiration du token JWT, défaut 30 minutes`
TEST_MODEL_LOGGER_PATH=`Répertoire et nom du fichier de logs de test du modèle dans le conteneur`
TRAIN_MODEL_LOGGER_PATH=`Répertoire et nom du fichier de logs d'entraînement dans le conteneur`
IMPORT_DATA_LOGGER_PATH=`Répertoire et nom du fichier de logs dans le conteneur`
AUTH_SERVICE_LOGGER_PATH=`Répertoire et nom du fichier de logs dans le conteneur`
MLFLOW_TRACKING_URI=https://dagshub.com/MariamaNadia/Projet_Formation_MLOps_DataScientest_2025.mlflow
MLFLOW_EXPERIMENT_NAME=ProductCodeClassifier
DAGSHUB_USERNAME=`Votre identifiant DagsHub`
DAGSHUB_TOKEN=`Votre token DagsHub`
AIRFLOW_UID=0
PROJECT_HOST_PATH=`Répertoire absolu de votre projet, requis pour l'usage de l'opérateur Airflow DockerOperator`
TEST_USER=`Login de l'utilisateur de l'API`
TEST_USER_FULLNAME=`Nom de l'utilisateur de l'API`
TEST_USER_PASSWORD=`Mot de passe de l'utilisateur de l'API`
```

### Dépendances
Chaque module de ce projet contient ses dépendances dans un fichier requirements qui lui est propre. Ce projet utilise Git pour gérer le versionnement du code. Il est donc constitué de plusieurs branches. Les données requises pour le développement du modèle (si vous entraînez le modèle from scratch)  sont versionnées via DVC DagsHub avec option d'enregistrement sur S3. Les étapes suivantes vous montrent comment configurer ces outils. Les étapes suivantes sont effectuées sur le répertoire racine du projet.

### Configurer un stockage distant avec DVC
Nous utilisons **S3 d’AWS** pour sauvegarder les données. Pour cela, il faut d'abord installer S3 à l’aide de la commande suivante :
 
```bash
pip install "dvc[S3]"
```

**Initialiser DVC dans le projet**

```bash
  dvc init
```

**Configurer DagsHub comme stockage distant** 
Une fois le support de S3 installé, il faudra mettre à jour le fichier de configuration `.dvc/config` pour ajouter DagsHub comme votre stockage distant. Les commandes à suivre sont les suivantes : 

```bash
dvc remote add origin s3://dvc
dvc remote modify origin endpointurl https://dagshub.com/MariamaNadia/Projet_Formation_MLOps_DataScientest_2025.s3

```
Ici, `votre_token(*)` est à récupérer sur l'interface de DagsHub. Il sera par ailleurs utilisé dans votre pipeline Airflow pour pousser les données automatiquement vers DVC.

```bash
dvc remote modify origin --local access_key_id votre_koken(*)
dvc remote modify origin --local secret_access_key votre_token(*)
```

Votre fichier `.dvc/config`(créé sur initialisation de dvc sur le répertoire du projet)  devrait ressembler à ceci : 

```bash
[core]
    remote = origin
['remote "origin"']
    url = s3://dvc
    endpointurl = https://dagshub.com/MariamaNadia/Projet_Formation_MLOps_DataScientest_2025.s3
```

### Récupérer les artefacts et modèles et données déjà existants
`Attention`, cette étape est très importante. Sans elle, vous ne pouvez avoir les artefacts et les modèles pour poursuivre le traitement.
Dans ce projet, nous entraînons un modèle multimodal utilisant deux modèles VGG16 et LSTM déjà existants et versionnés sur DasgHub. Il vous faudra donc les récupérer.
Vous pouvez récupéree les données lorsque vous souhaitez réentraîner le modèle.

```bash
dvc pull
```

### Lancer la pipeline Airflow
Pour un premier lancement, les bases de Airflow seront créées aisni que leurs volumes associés, les répertoires nécessaires à Aifrlow en local sont également créés.

```bash
make init-airflow
```

Pour lancer la pipeline, utilisez la commande suivante :

```bash
make start
```

Pour arrêter la pipeline, utilisez la commande suivante :

```bash
make stop
```

### Comment tester les endpoints
Un utilisateur a été créé par défaut pour pouvoir tester. Vous pouvez configurer ses accès dans le fichier `.env` à travers les paramètres (`TEST_USER,TEST_USER_FULLNAME,TEST_USER_PASSWORD`).

---------
## Structure du projet

### Structure globale
```
.
├── LICENSE
├── Makefile
├── REAMDME.md
├── __init__.py
├── airflow
│   ├── Dockerfile
│   ├── config
│   │   └── airflow.cfg
│   ├── dags
│   │   └── ml_pipeline.py
│   ├── initialize_airflow.sh
│   ├── logs
│   │   ├── dag_id=ml_pipeline_dvc
│   │   └── dag_processor
│   ├── plugins
│   └── requirements.txt
├── data
│   ├── predictions
│   │   ├── predictions.json
│   │   └── predictions.json.dvc
│   └── raw
│       ├── X_test_update.csv
│       ├── X_test_update.csv.dvc
│       ├── X_train_update.csv
│       ├── X_train_update.csv.dvc
│       ├── Y_train_CVw08PX.csv
│       ├── Y_train_CVw08PX.csv.dvc
│       ├── __MACOSX
│       ├── image_test
│       ├── image_train
│       └── images_low.zip
├── docker-compose.yml
├── image.png
├── logs
│   ├── api.log
│   ├── auth_service_logger.log
│   ├── import_data_logger.log
│   ├── lstm_accuracy_curves.png
│   ├── lstm_loss_curves.png
│   ├── serving_logger.log
│   ├── test_model_logger.log
│   ├── train
│   ├── train_model_logger.log
│   ├── validation
│   ├── vgg16_accuracy_curves.png
│   └── vgg16_loss_curves.png
├── mlartifacts
├── mlruns
├── models
│   ├── best_lstm_model.h5
│   ├── best_lstm_model.h5.dvc
│   ├── best_vgg16_model.h5
│   ├── best_vgg16_model.h5.dvc
│   ├── best_weights.json
│   ├── best_weights.json.dvc
│   ├── best_weights.pkl
│   ├── best_weights.pkl.dvc
│   ├── concatenate.h5
│   ├── concatenate.h5.dvc
│   ├── mapper.json
│   ├── mapper.json.dvc
│   ├── mapper.pkl
│   ├── mapper.pkl.dvc
│   ├── tokenizer_config.json
│   └── tokenizer_config.json.dvc
├── monitoring
│   ├── grafana
│   │   ├── Product_Classifier_Monitoring_1756323298849.json
│   │   ├── fastapi_metrics.json
│   │   └── provisioning
│   ├── node_exporter-1.8.1.linux-amd64
│   │   ├── LICENSE
│   │   ├── NOTICE
│   │   └── node_exporter
│   ├── node_exporter-1.8.1.linux-amd64.tar.gz
│   └── prometheus
│       └── prometheus.yml
├── notebooks
│   └── Rakuten.ipynb
├── reports
│   ├── detailled_pipeline.png
│   ├── endpoint_up_targets.PNG
│   ├── grafana_prometheus_data_source.PNG
│   ├── metriques_graph_prometheus.PNG
│   ├── mlflow_metrics.png
│   ├── mlfow_experiment.png
│   ├── pipeline.png
│   ├── predict_endpoint_input.png
│   ├── predict_endpoint_return.png
│   └── securing_api_authorized.png
├── requirements.txt
├── run_compose_options.sh
├── setup.py
├── src
│   ├── __init__.py
│   ├── auth_service
│   │   ├── Dockerfile
│   │   ├── main.py
│   │   ├── requirements.txt
│   │   └── security_logic.py
│   ├── config
│   ├── data
│   │   ├── __init__.py
│   │   └── build_data
│   ├── features
│   │   ├── __init__.py
│   │   └── build_features.py
│   └── models
│       ├── __init__.py
│       ├── predict
│       ├── serve
│       └── train
└── tests
    ├── __init__.py
    └── test_api.py
```

### Struture du code des stages : data, train, auth, predict

```
.
├── __init__.py
├── auth_service
│   ├── Dockerfile
│   ├── main.py
│   ├── requirements.txt
│   └── security_logic.py
├── config
├── data
│   ├── __init__.py
│   └── build_data
│       ├── Dockerfile
│       ├── __init__.py
│       ├── check_structure.py
│       ├── import_raw_data.py
│       ├── make_dataset.py
│       ├── requirements.txt
│       └── up.sh
├── features
│   ├── __init__.py
│   └── build_features.py
└── models
    ├── __init__.py
    ├── predict
    │   ├── Dockerfile
    │   ├── __init__.py
    │   ├── predict.py
    │   ├── requirements.txt
    │   └── up.sh
    ├── serve
    │   ├── Dockerfile
    │   ├── __init__.py
    │   ├── auth_utils.py
    │   ├── predict_logic.py
    │   ├── requirements.txt
    │   ├── serve_model_fastapi.py
    │   ├── serve_model_fastapi_old.py
    │   └── up.sh
    └── train
        ├── Dockerfile
        ├── __init__.py
        ├── main.py
        ├── requirements.txt
        ├── train_model.py
        └── up.sh
```


---------
## Détails d'implémentation

## Serving, dockerisation et tests unitaires
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
Le fichier `docker-compose.xml` à la racine du projet est utilisé pour orchestrer tous les services du projet. Nous utilisons le fichier `.env` pour assurer la réusabilité des différents environnemenst (production, staging etc) et la modularité.

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

## Automatisation Airflow/DVC/DagsHub/MLFlow

A partir de là, nous travaillons exclusvement avec Docker.
Donc, il faut obligatoirement disposer d'un fichier `.env` à la racine du projet. Exemple de contenu `.env` disponible sur le fichier `.env.example``.

1. Après avoir cloner le code, créer un fichier `.env` (ne pas oublier de mettre à jour vos informations dagshub (token))
```bash
touch .env

```

2. Données nécessaires : fichiers artefacts non disponibles sur Git
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

!! Le fichier à la racine `run_compose_options.sh` contient des examples de commandes pour exécuter chaque service en particulier(Tests unqiuement).

4. Exécuter la pipeline via Airflow en mode standalone

Lancer une instance airflow en local (juste pour des tests) : mettre à jour le fichier sous airflow/dags/initialize_airflow.sh et exécuter le fichier

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
- Premier lancement : initialise les bases Postgres SQL et Redis :
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

# Monitoring

Intègrer un système de monitoring complet basé sur **Prometheus** et **Node Exporter**, avec exposition des métriques de l'application **FastAPI**.

## 1.  Installation des dépendances
Les services de monitoring sont intégrés également dans notre docker compose.

```yaml
...
  prometheus:
    image: prom/prometheus
    container_name: prometheus
    networks:
      - product_classier
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
  
  grafana:
    image: grafana/grafana-enterprise
    container_name: grafana
    networks:
      - product_classier
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
    depends_on:
      - prometheus
...
```

## 2.  Mise à jour du code FastAPI

Ajout de l’instrumentation Prometheus dans le fichier `src.models.serve.serve_model_fastapi.py` :

```python
from prometheus_fastapi_instrumentator import Instrumentator

# Ajout après la création de l'application FastAPI
app = FastAPI()

# Instrumentation Prometheus
Instrumentator().instrument(app).expose(app)

```
## 3. Configuration prometheus.yml
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "prometheus"
    static_configs:
      - targets: ["localhost:9090"]

  - job_name: "fastapi"
    static_configs:
      - targets: ["serving_service_container:8000"]

  - job_name: "node"
    static_configs:
      - targets: ["localhost:9100"]
```

## 4.  Vérification de l’endpoint /metrics , prometheus, node exporter

Ouvrir dans le navigateur ou via curl :

```bash
http://localhost:8000/metrics

```


```bash
//http://localhost:9090

```


```bash
http://http://localhost:9100
```

## 6. Métriques collectées (exemple) :
- `http_requests_total{job="fastapi"}` — nombre de requêtes HTTP par endpoint, méthode, code
- `process_cpu_seconds_total` — temps CPU utilisé
- `node_memory_Active_bytes` — RAM active

### → Installer et configurer Grafana, le connecter à Prometheus et visualiser les métriques (FastAPI, Node Exporter) avec des dashboards.
Étape 1 : Lancer Grafana avec Docker

``` bash
docker run -d \
  -p 3000:3000 \
  --name=grafana \
  grafana/grafana

```
Demarrer grafana si déja crée
``` bash
docker start grafana
docker restart grafana

```
Étape 2 : Ajouter Prometheus comme source de données
Accèdez à Grafana : http://localhost:3000
Menu latéral gauche → ⚙️ Configuration → Data Sources
Cliquez sur `Add data source`
Choisissez Prometheus
Dans le champ URL, mettez :http://prometheus:9090

``` bash
 output
 Successfully queried the Prometheus API.
Next, you can start to visualize data by building a dashboard, or by querying data in the Explore view.
```

Étape 3 : Importer un Dashboard Node Exporter + FastAPI
🔹 Option A : Dashboard Node Exporter (prêt à l’emploi)
Menu gauche → 📊 Dashboards → New Dashboards → ADD visualisation → choisir la source de données prometheus 
Menu gauche → 📊 Dashboards → New Dashboards → ADD visualisation → choisir la source de données prometheus → 

