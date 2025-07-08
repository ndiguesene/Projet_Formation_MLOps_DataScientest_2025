init-airflow:
	mkdir -p ./airflow/dags ./airflow/logs ./airflow/plugins ./airflow/config
	@echo AIRFLOW_UID=$(shell id -u) > .env
	docker compose up airflow-init 
	docker compose up airflow-apiserver airflow-scheduler airflow-dag-processor airflow-worker airflow-triggerer

start:
	docker compose up

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
	rm -rf ./airflow/dags/
	rm -rf ./airflow/logs/
	rm -rf ./airflow/plugins/
	rm -rf ./airflow/config/

#docker stop $(docker ps -q)
#docker rm $(docker ps -aq)
#docker volume rm $(docker volume ls -q)

free-space:
	df -h