#!/usr/bin/env python
# coding: utf-8

import datetime
import time
import logging
import pandas as pd
import psycopg

from joblib import load
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
    ColumnQuantileMetric,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)

SEND_TIMEOUT = 10

# PostgreSQL table
create_table_statement = """
DROP TABLE IF EXISTS hw_metrics;
CREATE TABLE hw_metrics(
    timestamp TIMESTAMP,
    prediction_drift FLOAT,
    num_drifted_columns INTEGER,
    share_missing_values FLOAT,
    fare_amount_quantile_05 FLOAT
)
"""

# Feature definitions (same as baseline notebook)
target = "duration_min"
num_features = ["passenger_count", "trip_distance", "fare_amount", "total_amount"]
cat_features = ["PULocationID", "DOLocationID"]

column_mapping = ColumnMapping(
    target=None,
    prediction="prediction",
    numerical_features=num_features,
    categorical_features=cat_features,
)


def prep_db():
    """Create the metrics table in PostgreSQL."""
    with psycopg.connect(
        "host=localhost port=5433 user=postgres password=example", autocommit=True
    ) as conn:
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
        if res.fetchone() is None:
            conn.execute("CREATE DATABASE test;")

    with psycopg.connect(
        "host=localhost port=5433 dbname=test user=postgres password=example",
        autocommit=True,
    ) as conn:
        conn.execute(create_table_statement)


def prep_data():
    """Load reference data, March 2024 data, model, and generate predictions."""
    ref_data = pd.read_parquet("data/reference.parquet")

    raw_data = pd.read_parquet("data/green_tripdata_2024-03.parquet")

    # Create target column
    raw_data["duration_min"] = (
        raw_data.lpep_dropoff_datetime - raw_data.lpep_pickup_datetime
    )
    raw_data["duration_min"] = raw_data["duration_min"].apply(
        lambda td: float(td.total_seconds()) / 60
    )

    # Load model and generate predictions
    with open("models/lin_reg.bin", "rb") as f:
        model = load(f)

    features = raw_data[num_features + cat_features].fillna(0)
    raw_data["prediction"] = model.predict(features)

    logging.info(f"Reference data shape: {ref_data.shape}")
    logging.info(f"March 2024 data shape: {raw_data.shape}")
    logging.info(f"Predictions NaN count: {raw_data['prediction'].isna().sum()}")

    return ref_data, raw_data


def calculate_metrics(current_data, ref_data):
    """Run Evidently report and extract metrics."""
    report = Report(
        metrics=[
            ColumnDriftMetric(column_name="prediction"),
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
            ColumnQuantileMetric(column_name="fare_amount", quantile=0.5),
        ]
    )

    report.run(
        reference_data=ref_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )

    result = report.as_dict()

    prediction_drift = result["metrics"][0]["result"]["drift_score"]
    num_drifted = result["metrics"][1]["result"]["number_of_drifted_columns"]
    share_missing = result["metrics"][2]["result"]["current"]["share_of_missing_values"]
    fare_quantile = result["metrics"][3]["result"]["current"]["value"]

    return prediction_drift, num_drifted, share_missing, fare_quantile


def main():
    prep_db()
    ref_data, raw_data = prep_data()

    quantile_values = []

    with psycopg.connect(
        "host=localhost port=5433 dbname=test user=postgres password=example",
        autocommit=True,
    ) as conn:
        for day in range(1, 32):
            day_start = f"2024-03-{day:02d}"
            day_end = f"2024-03-{day + 1:02d}" if day < 31 else "2024-04-01"

            current = raw_data.loc[
                raw_data.lpep_pickup_datetime.between(
                    day_start, day_end, inclusive="left"
                )
            ]

            if len(current) == 0:
                logging.info(f"Day {day}: no data, skipping")
                continue

            # Drop rows with missing predictions (edge cases)
            current = current.dropna(subset=["prediction"])

            if len(current) == 0:
                logging.info(f"Day {day}: no valid predictions, skipping")
                continue

            try:
                prediction_drift, num_drifted, share_missing, fare_quantile = (
                    calculate_metrics(current, ref_data)
                )

                conn.execute(
                    """
                    INSERT INTO hw_metrics(
                        timestamp, prediction_drift, num_drifted_columns,
                        share_missing_values, fare_amount_quantile_05
                    ) VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        datetime.datetime(2024, 3, day),
                        prediction_drift,
                        num_drifted,
                        share_missing,
                        fare_quantile,
                    ),
                )

                quantile_values.append(fare_quantile)
                logging.info(
                    f"Day {day}: rows={len(current)}, "
                    f"drift={prediction_drift:.4f}, "
                    f"drifted_cols={num_drifted}, "
                    f"missing={share_missing:.4f}, "
                    f"fare_quantile_0.5={fare_quantile}"
                )

            except Exception as e:
                logging.error(f"Day {day}: failed ({e})")

            time.sleep(SEND_TIMEOUT)

    if quantile_values:
        logging.info(
            f"\n{'='*50}\n"
            f"Q3 ANSWER: Max fare_amount quantile(0.5) = {max(quantile_values)}\n"
            f"{'='*50}"
        )


if __name__ == "__main__":
    main()