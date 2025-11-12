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
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

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

FEATURE_LIST = [
    # Original dataset features
    "payment_type",           # payment method
    #"trip_distance",          # trip length in miles
    LOCATION_CONFIG["pul"]["cluster_col"],  # pickup location cluster
    LOCATION_CONFIG["dol"]["cluster_col"],  # dropoff location cluster
    "fare_amount",            # total fare amount

    # Engineered features
    "trip_minutes",           # trip duration in minutes
    "speed_kmh",              # average trip speed
    "fare_per_km_net",        # fare per km excluding extra fees
    "fare_per_passenger",     # fare divided by number of passengers
    "extra_fees_total",       # total of all extra fees
    "extra_fee_ratio",        # ratio of extra fees to total fare
    "distance_bucket",        # distance category
    "hour_of_day",            # pickup hour (0–23)
    "weekday",                # day of the week (0=Mon, 6=Sun)
    "is_weekend",             # 1 if weekend
    #"rush_hour",              # 1 if rush hour (7–10 or 16–19)
    "night_ride",             # 1 if night (22–5)
    "weekend_night",          # 1 if weekend and night
    #"rush_distance",          # distance x rush_hour interaction
    "is_airport_trip",        # 1 if airport pickup/dropoff
]


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

    # pickup hour (0–23) and day of the week (0=Mon, 6=Sun)
    df = df.with_columns([
        pl.col("tpep_pickup_datetime").dt.hour().alias("hour_of_day"),
        pl.col("tpep_pickup_datetime").dt.weekday().alias("weekday"),
    ])

    # 1 if weekend
    df = df.with_columns((pl.col("weekday") >= 5).cast(pl.Int8).alias("is_weekend"))

    # 1 if airport pickup/dropoff
    df = df.with_columns(
        pl.col("PULocationID").is_in([132, 138]).cast(pl.Int8).alias("is_airport_trip")
    )

    # total of all extra fees
    df = df.with_columns([
        (
            pl.col("airport_fee").fill_null(0)
            + pl.col("tolls_amount").fill_null(0)
            + pl.col("extra").fill_null(0)
            + pl.col("congestion_surcharge").fill_null(0)
        ).alias("extra_fees_total")
    ])

    # fare per km excluding extra fees
    df = df.with_columns(
        ((pl.col("fare_amount") - pl.col("extra_fees_total")) / pl.col("trip_distance"))
        .clip(0, 200)
        .alias("fare_per_km_net")
    )

    # fare divided by number of passengers
    if "passenger_count" in df.columns:
        df = df.with_columns(
            (pl.col("fare_amount") / pl.col("passenger_count").clip(1, None))
            .fill_nan(0)
            .alias("fare_per_passenger")
        )

    # trip duration in minutes and average trip speed
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

    # distance category
    df = df.with_columns([
        pl.when(pl.col("trip_distance") < 2)
        .then(0)
        .when(pl.col("trip_distance") < 8)
        .then(1)
        .otherwise(2)
        .alias("distance_bucket")
    ])


    # 1 if rush hour (7–10 or 16–19), 1 if night (22–5)
    df = df.with_columns([
        (
            (pl.col("hour_of_day").is_between(7, 10))
            | (pl.col("hour_of_day").is_between(16, 19))
        ).cast(pl.Int8).alias("rush_hour"),
        ((pl.col("hour_of_day") >= 22) | (pl.col("hour_of_day") <= 5))
        .cast(pl.Int8)
        .alias("night_ride"),
    ])

    # Rush distance interaction
    df = df.with_columns(
        (pl.col("trip_distance") * pl.col("rush_hour"))
        .alias("rush_distance")
    )

    # 1 if weekend and night
    df = df.with_columns(
        ((pl.col("is_weekend") == 1) & (pl.col("night_ride") == 1))
        .cast(pl.Int8)
        .alias("weekend_night")
    )

    # ratio of extra fees to total fare
    df = df.with_columns(
        (pl.col("extra_fees_total") / pl.col("fare_amount").clip(1, None))
        .clip(0, 1)
        .alias("extra_fee_ratio")
    )

    return df

# ---------------------------------------------------------------------
# LOCATION CLUSTERING
# ---------------------------------------------------------------------

def compute_location_tip_clusters(df, location_type="pul"):
    cfg = LOCATION_CONFIG[location_type]

    # Compute mean tip rate per location
    return df.group_by(cfg["id_col"]).agg(pl.col(TARGET_VAR).mean().alias(cfg["rate_col"]))

def cluster_locations(location_stats, location_type="pul", n_clusters=7):
    # Cluster locations by smoothed tip rate
    cfg = LOCATION_CONFIG[location_type]
    km = KMeans(n_clusters=n_clusters, random_state=SEED)
    labels = km.fit_predict(location_stats[cfg["rate_col"]].to_numpy().reshape(-1, 1))

    # Add cluster labels
    return location_stats.with_columns(pl.Series(cfg["cluster_col"], labels).cast(pl.Int8))


def apply_location_clusters(df, clusters, location_type="pul"):
    # Merge cluster info back into main dataset
    cfg = LOCATION_CONFIG[location_type]

    # Drop old cluster column if it exists
    if cfg["cluster_col"] in df.columns:
        df = df.drop(cfg["cluster_col"])

    # Join by location ID
    return df.join(clusters, on=cfg["id_col"], how="left")

# ---------------------------------------------------------------------
# TRAINING & VALIDATION
# ---------------------------------------------------------------------

def train_and_evaluate(df_train, sample_size=None):
    # Speed up testing by sampling
    if sample_size is not None:
        df_train = df_train.sample(n=sample_size, seed=SEED)

    features = FEATURE_LIST

    x = df_train[features].to_pandas()
    y = df_train[TARGET_VAR].to_pandas()

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=SEED)

    model = HistGradientBoostingClassifier(
        max_depth=7,
        learning_rate=0.08,
        max_iter=350,
        min_samples_leaf=10,
        l2_regularization=0.3,
        class_weight="balanced",
        random_state=SEED
    )

    model.fit(x_train, y_train)
    preds = model.predict(x_val)
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

    pul_clusters = cluster_locations(pul_stats, location_type="pul", n_clusters=5)
    dol_clusters = cluster_locations(dol_stats, location_type="dol", n_clusters=5)

    df_train = apply_location_clusters(df_train, pul_clusters, "pul")
    df_train = apply_location_clusters(df_train, dol_clusters, "dol")

    df_test = apply_location_clusters(df_test, pul_clusters, "pul")
    df_test = apply_location_clusters(df_test, dol_clusters, "dol")

    model = train_and_evaluate(df_train)
    create_submission(model, df_test)


if __name__ == "__main__":
    main()
