import polars as pl
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

TARGET_VAR = "has_tipped_over_20"

LOCATION_CONFIG = {
    "pul": {
        "id_col": "PULocationID",
        "rate_col": "pul_tip_rate",
        "cluster_col": "pul_cluster",
    },
    "dol": {
        "id_col": "DOLocationID",
        "rate_col": "dol_tip_rate",
        "cluster_col": "dol_cluster",
    },
}

FEATURE_LIST  = [
    "trip_distance",
    "trip_minutes",
    "speed_kmh",
    "fare_amount",
    "fare_per_km_net",
    "fare_per_passenger",
    "extra_fees_total",
    "extra_fee_ratio",
    "distance_bucket",
    "hour_of_day",
    "hour_sin",
    "hour_cos",
    "weekday",
    "is_weekend",
    "rush_hour",
    "night_ride",
    "weekend_night",
    "rush_distance",
    "is_airport_trip",
    LOCATION_CONFIG["pul"]["cluster_col"],
    LOCATION_CONFIG["dol"]["cluster_col"],
    "payment_type",
]


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
    # Remove trips with invalid distance or fare values
    df = df.filter((pl.col("trip_distance") >= 0.1) & (pl.col("trip_distance") <= 80))
    df = df.filter((pl.col("fare_amount") >= 2) & (pl.col("fare_amount") <= 500))

    # Remove extreme outliers based on fare per kilometer
    df = df.with_columns(
        (pl.col("fare_amount") / pl.col("trip_distance")).alias("fare_per_km")
    )
    q_low = df["fare_per_km"].quantile(0.01)
    q_high = df["fare_per_km"].quantile(0.99)
    df = df.filter(pl.col("fare_per_km").is_between(q_low, q_high))
    df = df.drop("fare_per_km")

    # Replace negative monetary values with zero
    money_columns = [
        "fare_amount", "tip_amount", "total_amount", "tolls_amount",
        "extra", "mta_tax", "congestion_surcharge"
    ]
    df = df.with_columns([
        pl.when(pl.col(c) < 0).then(0).otherwise(pl.col(c)).alias(c)
        for c in money_columns if c in df.columns
    ])

    # Keep only trips with valid pickup and dropoff times and reasonable durations
    if "tpep_dropoff_datetime" in df.columns:
        df = df.filter(pl.col("tpep_dropoff_datetime") > pl.col("tpep_pickup_datetime"))
        df = df.with_columns((
            (pl.col("tpep_dropoff_datetime") - pl.col("tpep_pickup_datetime"))
            .dt.total_microseconds() / 60_000_000
        ).alias("trip_minutes"))
        df = df.filter((pl.col("trip_minutes") >= 1) & (pl.col("trip_minutes") <= 240))
        df = df.drop("trip_minutes")

    # Optimize memory usage and ensure correct data types
    df = df.with_columns([
        pl.col("PULocationID").cast(pl.Int16),
        pl.col("DOLocationID").cast(pl.Int16),
        pl.col("payment_type").fill_null(0).cast(pl.Int8),
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
    if "tpep_dropoff_datetime" in df.columns and df["tpep_dropoff_datetime"].dtype == pl.String:
        df = df.with_columns(
            pl.col("tpep_dropoff_datetime").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
        )

    # Time features
    df = df.with_columns([
        pl.col("tpep_pickup_datetime").dt.hour().alias("hour_of_day"),
        pl.col("tpep_pickup_datetime").dt.weekday().alias("weekday"),
    ])

    # Weekend flag
    df = df.with_columns((pl.col("weekday") >= 5).cast(pl.Int8).alias("is_weekend"))

    # Airport pickup flag
    df = df.with_columns(
        pl.col("PULocationID").is_in([132, 138]).cast(pl.Int8).alias("is_airport_trip")
    )

    # Extra fees total
    df = df.with_columns([
        (
            pl.col("airport_fee").fill_null(0)
            + pl.col("tolls_amount").fill_null(0)
            + pl.col("extra").fill_null(0)
            + pl.col("congestion_surcharge").fill_null(0)
        ).alias("extra_fees_total")
    ])

    # Net fare per km
    df = df.with_columns(
        ((pl.col("fare_amount") - pl.col("extra_fees_total")) / pl.col("trip_distance"))
        .clip(0, 200)
        .alias("fare_per_km_net")
    )

    # Fare per passenger
    # Fare per passenger (mit Clip für Division durch Null)
    if "passenger_count" in df.columns:
        df = df.with_columns(
            (pl.col("fare_amount") / pl.col("passenger_count").clip(1, None))
            .fill_nan(0)
            .alias("fare_per_passenger")
        )

    # Duration & Speed
    if "tpep_dropoff_datetime" in df.columns:
        df = df.with_columns(
            ((pl.col("tpep_dropoff_datetime") - pl.col("tpep_pickup_datetime"))
             .dt.total_microseconds() / 60_000_000).alias("trip_minutes")
        )
        df = df.with_columns(
            (pl.col("trip_distance") / (pl.col("trip_minutes") / 60))
            .clip(0, 120)
            .alias("speed_kmh")
        )

    # Distance bucket
    df = df.with_columns([
        pl.when(pl.col("trip_distance") < 2)
        .then(0)
        .when(pl.col("trip_distance") < 8)
        .then(1)
        .otherwise(2)
        .alias("distance_bucket")
    ])

    # Rush hour and night ride
    df = df.with_columns([
        (
            (pl.col("hour_of_day").is_between(7, 10))
            | (pl.col("hour_of_day").is_between(16, 19))
        ).cast(pl.Int8).alias("rush_hour"),
        ((pl.col("hour_of_day") >= 22) | (pl.col("hour_of_day") <= 5))
        .cast(pl.Int8)
        .alias("night_ride"),
    ])

    # Weekend-night combination
    df = df.with_columns(
        ((pl.col("is_weekend") == 1) & (pl.col("night_ride") == 1))
        .cast(pl.Int8)
        .alias("weekend_night")
    )

    # Rush distance interaction
    df = df.with_columns(
        (pl.col("trip_distance") * pl.col("rush_hour"))
        .alias("rush_distance")
    )

    # Extra fee ratio
    df = df.with_columns(
        (pl.col("extra_fees_total") / pl.col("fare_amount").clip(1, None))
        .clip(0, 1)
        .alias("extra_fee_ratio")
    )

    # Circular time encoding
    hour_rad = 2 * np.pi * pl.col("hour_of_day") / 24
    df = df.with_columns([
        hour_rad.sin().alias("hour_sin"),
        hour_rad.cos().alias("hour_cos"),
    ])

    return df
# ---------------------------------------------------------------------
# LOCATION CLUSTERING
# ---------------------------------------------------------------------

def compute_location_tip_clusters(df, type="pul"):
    # PROBLEM: areas with low num_trips -> needs spatial smoothing
    if type == "pul":
        feature = "PULocationID"
        new_feature = "pul_tip_rate"
    else:
        feature = "DOLocationID"
        new_feature = "dol_tip_rate"

    return df.group_by(feature).agg(pl.col(TARGET_VAR).mean().alias(new_feature))


def cluster_locations(location_stats, type="pul", n_clusters=10):
    cfg = LOCATION_CONFIG[type]
    km = KMeans(n_clusters=n_clusters, random_state=SEED)
    labels = km.fit_predict(location_stats[cfg["rate_col"]].to_numpy().reshape(-1, 1))
    return location_stats.with_columns(pl.Series(cfg["cluster_col"], labels).cast(pl.Int8))


def apply_location_clusters(df, clusters, type="pul"):
    cfg = LOCATION_CONFIG[type]

    # Drop existing cluster column if present
    if cfg["cluster_col"] in df.columns:
        df = df.drop(cfg["cluster_col"])

    return df.join(clusters, on=cfg["id_col"], how="left")

# ---------------------------------------------------------------------
# TRAINING & VALIDATION
# ---------------------------------------------------------------------

def train_and_evaluate(df_train, sample_size=None):
    # Speed up testing by sampling
    if sample_size is not None:
        df_train = df_train.sample(n=sample_size, seed=SEED)

    features = FEATURE_LIST

    X = df_train[features].to_pandas()
    y = df_train[TARGET_VAR].to_pandas()

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)

    model = HistGradientBoostingClassifier(
        max_depth=7,
        learning_rate=0.08,
        max_iter=350,
        min_samples_leaf=20,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=SEED
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    print("Validation F1 Score:", f1_score(y_val, preds))

    return model

# ---------------------------------------------------------------------
# CREATE SUBMISSION
# ---------------------------------------------------------------------

def create_submission(model, df_test):
    features = FEATURE_LIST

    X_test = df_test[features].to_pandas()
    preds = model.predict(X_test)

    pl.DataFrame({
        "row_nr": df_test["row_nr"],
        TARGET_VAR: preds.astype(int),
    }).write_csv("submission.csv")

    print("submission was created")

# ---------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------

def main():
    df_train, df_test = load_data()
    df_train = clean_data(df_train)
    df_train = engineer_features(df_train)
    df_test = engineer_features(df_test)

    pul_stats = compute_location_tip_clusters(df_train, "pul")
    dol_stats = compute_location_tip_clusters(df_train, "dol")

    pul_clusters = cluster_locations(pul_stats, type="pul", n_clusters=7)
    dol_clusters = cluster_locations(dol_stats, type="dol", n_clusters=7)

    df_train = apply_location_clusters(df_train, pul_clusters, "pul")
    df_train = apply_location_clusters(df_train, dol_clusters, "dol")

    df_test = apply_location_clusters(df_test, pul_clusters, "pul")
    df_test = apply_location_clusters(df_test, dol_clusters, "dol")

    model = train_and_evaluate(df_train)
    create_submission(model, df_test)


if __name__ == "__main__":
    main()