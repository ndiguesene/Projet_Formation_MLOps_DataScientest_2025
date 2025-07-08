init-airflow:
	mkdir -p ./airflow/dags ./airflow/logs ./airflow/plugins ./airflow/config
	@echo AIRFLOW_UID=$(shell id -u) > .env
	docker compose up airflow-init airflow-apiserver airflow-scheduler airflow-dag-processor airflow-worker airflow-triggerer airflow-cli

start:
	docker compose up

stop:
	docker compose down

restart:
	docker compose up --build

airflow-logs:
	docker-compose logs airflow-webserver

#del-containers-and-images:
#	docker stop $(docker ps -q)
#	docker rm $(docker ps -aq)
#	docker volume rm $(docker volume ls -q)

free-space:
	df -h