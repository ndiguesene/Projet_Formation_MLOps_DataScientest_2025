from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime
import os
from dotenv import load_dotenv
from airflow.providers.docker.operators.docker import DockerOperator

load_dotenv(dotenv_path="/opt/airflow/mlops-project/.env")

with DAG(
    dag_id="ml_pipeline_dvc",
    start_date=datetime(2025, 6, 23),
    schedule=None,
    catchup=False
) as dag:

    # -*- we are building the image for the data service to fetch data from the remote server and store it in the local filesystem -*-
    build_image_fetch_data = BashOperator(
        task_id="build_image_fetch_data",
        bash_command="""

          cd /opt/airflow/mlops-project && \
          docker build -f ./src/data/build_data/Dockerfile -t data_service .

        """
    )
    # -*- we are creating the actual container to fetch data from the remote server and store it in the local filesystem -*-. Note: The DockerOperator is used to run the container that fetches data.
    fetch_data = DockerOperator(
        task_id="fetch_data",
        image="data_service",
        container_name="data_service_container",
        api_version="auto",
        environment={
            "DATA_PATH": os.getenv("DATA_PATH"),
            "IMPORT_DATA_LOGGER_PATH": os.getenv("IMPORT_DATA_LOGGER_PATH"),
            "PYTHONUNBUFFERED": "1"
        },
        docker_url="unix://var/run/docker.sock"#,
        #volumes=[
        #    "/opt/airflow/mlops-project/logs:/app/logs",
        #    "/opt/airflow/mlops-project/data:/app/data"
        #],
        #auto_remove=True,
    )

    # -*- here we are pushing data to dvc -*-. 
    push_input_data = BashOperator(
        task_id="push_input_data",
        bash_command="""

        cd /opt/airflow/mlops-project && \
        
        # Versionner les données d'entrée
        dvc add ./data/raw/*.csv && \
        git add ./data/raw/*.dvc && \
        dvc push

        """
    )


    train_model = BashOperator(
        task_id="train_model",
        bash_command="""

        cd /opt/airflow/mlops-project && \
        docker compose run --rm training_service && \

        """
    )

    push_model_artifacts = BashOperator(
        task_id="push_model_artifacts",
        bash_command="""

        cd /opt/airflow/mlops-project && \
        
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

    build_image_fetch_data >> fetch_data >> push_input_data
    # >> train_model >> run_prediction >> start_auth >> start_serving
