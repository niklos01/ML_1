# ---------------------------------------------------------------------
# NYC Taxi Tip Prediction Challenge (Optimized Version)
# ---------------------------------------------------------------------

import polars as pl
import numpy as np
import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
TARGET_VAR = "has_tipped_over_20"

# ---------------------------------------------------------------------
# SEED SETUP
# ---------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)


# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------
def load_data():
    df_train = pl.read_parquet("train.parquet")
    df_test = pl.read_parquet("test.parquet")
    return df_train, df_test

# ---------------------------------------------------------------------
# CLEANING
# ---------------------------------------------------------------------
def clean_data(df: pl.DataFrame) -> pl.DataFrame:

    df = df.drop_nulls(["trip_distance", "fare_amount", "passenger_count"])

    # Fill fee/tax columns
    fee_columns = [
        "airport_fee", "tolls_amount", "extra", "mta_tax",
        "congestion_surcharge", "improvement_surcharge", "cbd_congestion_fee"
    ]
    df = df.with_columns([pl.col(c).fill_null(0) for c in fee_columns])

    # Remove physically impossible trips
    df = df.filter(pl.col("trip_distance").is_between(0.1, 100))
    df = df.filter(pl.col("passenger_count").is_between(1, 8))

    # Remove bad fare logic (fare_per_km sanity check)
    df = df.with_columns((pl.col("fare_amount") / pl.col("trip_distance")).alias("fare_per_km"))
    df = df.filter(pl.col("fare_per_km").is_between(0.05, 200))
    df = df.drop("fare_per_km")

    # ✅ Memory optimization
    df = df.with_columns([
        pl.col("PULocationID").cast(pl.Int16),
        pl.col("DOLocationID").cast(pl.Int16),
        pl.col("payment_type").cast(pl.Int8)
    ])

    return df

# ---------------------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------------------
def engineer_features(df: pl.DataFrame) -> pl.DataFrame:
    # Ensure datetime dtype
    if df["tpep_pickup_datetime"].dtype == pl.String:
        df = df.with_columns(
            pl.col("tpep_pickup_datetime").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
        )

    # Create time features
    df = df.with_columns([
        pl.col("tpep_pickup_datetime").dt.hour().alias("hour_of_day"),
        pl.col("tpep_pickup_datetime").dt.weekday().alias("weekday")
    ])

    # Weekend flag
    df = df.with_columns(
        (pl.col("weekday") >= 5).cast(pl.Int8).alias("is_weekend")
    )

    # Airport pickup flag
    df = df.with_columns(
        pl.col("PULocationID").is_in([132, 138]).cast(pl.Int8).alias("is_airport_trip")
    )

    df = df.with_columns([
        (pl.col("fare_amount") / pl.col("trip_distance")).alias("fare_per_km"),
        (pl.col("airport_fee") + pl.col("tolls_amount") + pl.col("extra") +
         pl.col("congestion_surcharge")).alias("extra_fees_total"),
        pl.when(pl.col("trip_distance") < 2).then(0)
        .when(pl.col("trip_distance") < 8).then(1)
        .otherwise(2)
        .alias("distance_bucket"),
        ((pl.col("hour_of_day").is_between(7, 10)) |
         (pl.col("hour_of_day").is_between(16, 19))).cast(pl.Int8).alias("rush_hour"),
        (pl.col("hour_of_day").is_between(22, 5)).cast(pl.Int8).alias("night_ride")
    ])


    return df


# ---------------------------------------------------------------------
# LOCATION CLUSTERING
# ---------------------------------------------------------------------
def compute_location_tip_clusters(df_train):
    #PROBLEM: areas with low num_trips -> needs spatial smoothing
    return (
        df_train
        .group_by("PULocationID")
        .agg(pl.col(TARGET_VAR).mean().alias("tip_rate_per_locatoin"))
    )

def cluster_locations(location_stats, n_clusters=10):
    km = KMeans(n_clusters=n_clusters, random_state=SEED)
    labels = km.fit_predict(location_stats["tip_rate_per_locatoin"].to_numpy().reshape(-1, 1))

    return location_stats.with_columns(
        pl.Series("pickup_cluster", labels).cast(pl.Int8)
    )

def apply_location_clusters(df, clusters):
    return df.join(clusters, on="PULocationID", how="left")

# ---------------------------------------------------------------------
# TRAINING & VALIDATION
# ---------------------------------------------------------------------
def train_and_evaluate(df_train, sample_size=None):

    # ✅ OPTION: Speed up testing by sampling
    if sample_size is not None:
        df_train = df_train.sample(n=sample_size, seed=SEED)

    features = [
        "trip_distance", "fare_amount", "hour_of_day", "is_weekend",
        "is_airport_trip", "pickup_cluster", "payment_type"
    ]

    X = df_train[features].to_pandas()
    y = df_train[TARGET_VAR].to_pandas()

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1))
    ])

    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    print("Validation F1 Score:", f1_score(y_val, preds))
    return model

# ---------------------------------------------------------------------
# CREATE SUBMISSION
# ---------------------------------------------------------------------
def create_submission(model, df_test):
    features = [
        "trip_distance", "fare_amount", "hour_of_day", "is_weekend",
        "is_airport_trip", "pickup_cluster", "payment_type"
    ]
    X_test = df_test[features].to_pandas()
    preds = model.predict(X_test)

    pl.DataFrame({
        "row_nr": df_test["row_nr"],
        TARGET_VAR: preds.astype(int)
    }).write_csv("submission.csv")

    print("✅ submission.csv successfully written!")

# ---------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------
def main():
    df_train, df_test = load_data()

    df_train = clean_data(df_train)

    df_train = engineer_features(df_train)
    df_test = engineer_features(df_test)

    clusters = compute_location_tip_clusters(df_train)
    clusters = cluster_locations(clusters, n_clusters=10)
    print(clusters)

    #df_train = apply_location_clusters(df_train, clusters)
    #df_test = apply_location_clusters(df_test, clusters)

    #model = train_and_evaluate(df_train, sample_size=1000)  # ✅ much faster during dev
    #create_submission(model, df_test)

if __name__ == "__main__":
    main()
