from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id="ml_pipeline_dvc",
    start_date=datetime(2025, 6, 23),
    schedule=None,
    catchup=False
) as dag:

    fetch_data = BashOperator(
        task_id="fetch_data",
        bash_command="""

          #cd /Users/tiam028713/Documents/Formations/Projet_2025_MLOps/Projet_Formation_MLOps_DataScientest_2025/ && \
          #docker compose run --rm data_service

        """
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command="""

        #cd /Users/tiam028713/Documents/Formations/Projet_2025_MLOps/Projet_Formation_MLOps_DataScientest_2025/ && \
        docker compose run --rm training_service && \
        
        # Versionner les artefacts produits
        dvc add models/*.h5 && \
        
        dvc add models/*.pkl && \
        dvc add models/*.json && \

        # Pousser vers DVC remote
        git add models/*.dvc && \
        dvc push

        """
    )

    # créer un bashoperator avec des arguments pour pousser les données et qui contiendra

    run_prediction = BashOperator(
        task_id="predict_model",
        bash_command="docker compose run --rm prediction_service && dvc add data/predictions/predictions.json && git add data/predictions/predictions.json.dvc && dvc push"
    )

    start_auth = BashOperator(
        task_id="auth_service_up",
        bash_command="docker compose up -d auth_service"
    )

    start_serving = BashOperator(
        task_id="serving_service_up",
        bash_command="docker compose up -d serving_service"
    )

    #push_to_git = BashOperator(
    #    task_id="push_to_git",
    #    bash_command="""
    #      git commit -m "Update model/artifacts and input data from Airflow run"
    #      git commit -m "Update DVC tracked files"
    #      git push origin main
    #    """
    #)

    fetch_data >> train_model >> run_prediction >> start_auth >> start_serving
