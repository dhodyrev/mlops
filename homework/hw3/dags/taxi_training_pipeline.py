import os
import pickle
from datetime import datetime

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

import mlflow

from airflow import DAG
from airflow.operators.python import PythonOperator

DATA_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/"
    "yellow_tripdata_2023-03.parquet"
)
MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT = "nyc-taxi-hw3"
DATA_DIR = "/tmp/hw3_data"
os.makedirs(DATA_DIR, exist_ok=True)


def ingest(**kwargs):
    df = pd.read_parquet(DATA_URL)
    raw_path = f"{DATA_DIR}/raw.parquet"
    df.to_parquet(raw_path)

    row_count = len(df)
    print(f"Q3 → Loaded {row_count:,} records")
    kwargs["ti"].xcom_push(key="raw_path", value=raw_path)


def prepare(**kwargs):
    raw_path = kwargs["ti"].xcom_pull(key="raw_path")
    df = pd.read_parquet(raw_path)

    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df["duration"] = df["duration"].dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    prepared_path = f"{DATA_DIR}/prepared.parquet"
    df.to_parquet(prepared_path)

    print(f"Q4 → After filtering: {len(df):,} records")
    kwargs["ti"].xcom_push(key="prepared_path", value=prepared_path)


def train(**kwargs):
    prepared_path = kwargs["ti"].xcom_pull(key="prepared_path")
    df = pd.read_parquet(prepared_path)

    categorical = ["PULocationID", "DOLocationID"]
    dicts = df[categorical].to_dict(orient="records")

    dv = DictVectorizer(sparse=True)
    X = dv.fit_transform(dicts)
    y = df["duration"].values

    lr = LinearRegression()
    lr.fit(X, y)

    y_pred = lr.predict(X)
    rmse = root_mean_squared_error(y, y_pred)

    print(f"Q5 → Intercept: {lr.intercept_:.2f}")
    print(f"     RMSE: {rmse:.4f}")

    with open(f"{DATA_DIR}/dv.pkl", "wb") as f:
        pickle.dump(dv, f)
    with open(f"{DATA_DIR}/lr.pkl", "wb") as f:
        pickle.dump(lr, f)

    kwargs["ti"].xcom_push(key="rmse", value=rmse)
    kwargs["ti"].xcom_push(key="intercept", value=lr.intercept_)


def register(**kwargs):
    rmse = kwargs["ti"].xcom_pull(key="rmse")
    intercept = kwargs["ti"].xcom_pull(key="intercept")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run():
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("intercept", intercept)

        mlflow.log_artifact(f"{DATA_DIR}/dv.pkl", artifact_path="preprocessor")

        with open(f"{DATA_DIR}/lr.pkl", "rb") as f:
            lr = pickle.load(f)
        mlflow.sklearn.log_model(lr, artifact_path="model")

        print("Q6 → Model logged. Check MLflow UI → Artifacts → model/MLmodel")
        print("     Look for model_size_bytes field")


with DAG(
    dag_id="taxi_training_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["mlops-zoomcamp", "hw3"],
) as dag:

    t_ingest  = PythonOperator(task_id="ingest",   python_callable=ingest)
    t_prepare = PythonOperator(task_id="prepare",  python_callable=prepare)
    t_train   = PythonOperator(task_id="train",    python_callable=train)
    t_register = PythonOperator(task_id="register", python_callable=register)

    t_ingest >> t_prepare >> t_train >> t_register