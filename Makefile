# First launch of Airflow
init-airflow: 
	mkdir -p ./airflow/dags ./airflow/logs ./airflow/plugins ./airflow/config
		# Copy model files into the 'models' volume
	docker run --rm \
  	-v models:/app/models \
  	-v ./models:/source \
  	busybox sh -c "cp /source/* /app/models"

# Copy data files into the 'data' volume
	docker run --rm \
  	-v data:/app/data \
  	-v ./data:/source \
  	busybox sh -c "cp -r /source/* /app/data"

# Copy logs into the 'logs' volume
	docker run --rm \
  	-v logs:/app/logs \
  	-v ./logs:/source \
  	busybox sh -c "cp -r /source/* /app/logs"
	@echo AIRFLOW_UID=$(shell id -u) >> .env
	docker compose up airflow-init

start:
	docker compose up
#docker compose up airflow-apiserver airflow-scheduler airflow-dag-processor airflow-worker airflow-triggerer
#docker compose up

stop:
	docker compose down

restart:
	docker compose up --build

airflow-logs:
	docker-compose logs airflow-webserver

del-containers-and-images:
	docker compose down --volumes --rmi all
	docker compose down --volumes --remove-orphans
	docker images | grep airflow | awk '{print $3}' | xargs docker rmi -f

#docker stop $(docker ps -q)
#docker rm $(docker ps -aq)
#docker volume rm $(docker volume ls -q)

free-space:
	df -h