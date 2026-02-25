#!/bin/bash
set -e

airflow db migrate
airflow users create -u admin -p admin -f Admin -l User -r Admin -e admin@example.com || true

mlflow server \
  --backend-store-uri sqlite:///opt/airflow/mlflow_data/mlflow.db \
  --host 0.0.0.0 --port 5000 &

airflow scheduler &

exec airflow webserver --port 8080