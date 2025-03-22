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
---
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