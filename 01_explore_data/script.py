"""
Exploratory Data Analysis (EDA) for Taxi Tip Prediction Challenge
-----------------------------------------------------------------

This script loads the training and test datasets, inspects dataset shapes,
analyzes target variable distribution, summarizes numerical and categorical
features, and performs interaction analysis between key features.

No model training occurs here.
"""

import os
import random
import numpy as np
import polars as pl
from polars.datatypes import Datetime, Date

np.random.seed(42)
random.seed(42)

pl.Config.set_tbl_rows(300)
pl.Config.set_tbl_cols(50)


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
TRAIN_FILE = os.path.join(BASE_DIR, "train.parquet")
TEST_FILE = os.path.join(BASE_DIR, "test.parquet")
TARGET_COLUMN = "has_tipped_over_20"


def load_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    print("Loading datasets...")
    df_train = pl.read_parquet(TRAIN_FILE)
    df_test = pl.read_parquet(TEST_FILE)
    return df_train, df_test


def analyze_target_distribution(df_train: pl.DataFrame) -> pl.DataFrame:
    print("\n=== TARGET DISTRIBUTION ===")
    dist = df_train.select(df_train.get_column(TARGET_COLUMN).value_counts(sort=True))
    imbalance_ratio = dist.row(0)[1] / dist.row(1)[1]
    print("Imbalance ratio:", imbalance_ratio)
    print(dist)
    return dist


def analyze_passenger_count_effect(df_train: pl.DataFrame) -> pl.DataFrame:
    print("\n=== PASSENGER COUNT EFFECT ON TIPPING ===")
    result = (
        df_train.group_by("passenger_count")
        .agg([
            pl.mean(TARGET_COLUMN).alias("tip_over_20_rate"),
            pl.count().alias("num_rides"),
        ])
        .sort("passenger_count")
    )
    print(result)
    return result


def interaction_passenger_payment(df: pl.DataFrame) -> pl.DataFrame:
    result = (
        df.group_by(["passenger_count", "payment_type"])
        .agg([
            pl.col(TARGET_COLUMN).mean().alias("tip_rate"),
            pl.len().alias("num_rides")
        ])
        .sort(["passenger_count", "payment_type"])
    )
    print("\n=== passenger_count × payment_type ===")
    print(result)
    return result


def interaction_distance_payment(df: pl.DataFrame) -> pl.DataFrame:
    df_binned = df.with_columns(
        pl.when(pl.col("trip_distance") < 1).then(pl.lit("very_short"))
        .when(pl.col("trip_distance") < 3).then(pl.lit("short"))
        .when(pl.col("trip_distance") < 7).then(pl.lit("medium"))
        .when(pl.col("trip_distance") < 15).then(pl.lit("long"))
        .otherwise(pl.lit("very_long"))
        .alias("distance_bucket")
    )

    result = (
        df_binned.group_by(["distance_bucket", "payment_type"])
        .agg([
            pl.col(TARGET_COLUMN).mean().alias("tip_rate"),
            pl.len().alias("num_rides")
        ])
        .sort(["distance_bucket", "payment_type"])
    )

    print("\n=== trip_distance × payment_type ===")
    print(result)
    return result



def interaction_hour_payment(df: pl.DataFrame) -> pl.DataFrame:
    result = (
        df.group_by(["hour_of_day", "payment_type"])
        .agg([
            pl.col(TARGET_COLUMN).mean().alias("tip_rate"),
            pl.len().alias("num_rides")
        ])
        .sort(["hour_of_day", "payment_type"])
    )
    print("\n=== hour_of_day × payment_type ===")
    print(result)
    return result


def interaction_airport_payment(df: pl.DataFrame) -> pl.DataFrame:
    result = (
        df.group_by(["is_airport_trip", "payment_type"])
        .agg([
            pl.col(TARGET_COLUMN).mean().alias("tip_rate"),
            pl.len().alias("num_rides")
        ])
        .sort(["is_airport_trip", "payment_type"])
    )
    print("\n=== is_airport_trip × payment_type ===")
    print(result)
    return result


def interaction_pickup_payment(df: pl.DataFrame) -> pl.DataFrame:
    top_locations = (
        df.group_by("PULocationID")
        .agg(pl.len().alias("num_rides"))
        .sort("num_rides", descending=True)
        .head(20)
        .select("PULocationID")
    )
    df_top = df.join(top_locations, on="PULocationID", how="inner")
    result = (
        df_top.group_by(["PULocationID", "payment_type"])
        .agg([
            pl.col(TARGET_COLUMN).mean().alias("tip_rate"),
            pl.len().alias("num_rides")
        ])
        .sort(["PULocationID", "payment_type"])
    )
    print("\n=== PULocationID × payment_type (Top 20) ===")
    print(result)
    return result


def main():
    df_train, df_test = load_data()

    print(df_train.columns)

    df_train = df_train.with_columns([
        pl.col("tpep_pickup_datetime").dt.hour().alias("hour_of_day"),
        (pl.col("airport_fee") > 0).cast(pl.Int8).alias("is_airport_trip")
    ])

    print("\nDataset:", df_train.shape)

    analyze_target_distribution(df_train)
    analyze_passenger_count_effect(df_train)

    showInteactions = False

    if showInteactions :
        interaction_passenger_payment(df_train)
        interaction_distance_payment(df_train)
        interaction_hour_payment(df_train)
        interaction_airport_payment(df_train)
        interaction_pickup_payment(df_train)

    result = (
        df_train
        .group_by("payment_type")
        .agg([
            pl.mean(TARGET_COLUMN).alias("tip_over_20_rate"),  # Anteil >20%
            pl.count().alias("num_rides")  # Anzahl Fahrten
        ])
        .sort("tip_over_20_rate", descending=True)  # nach Tipp-Verhalten sortieren
    )

    print(result)



if __name__ == "__main__":
    main()
