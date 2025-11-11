# 123456_challenge_01.py
# ------------------------------------------------------------
# Machine Learning Challenge: Predict if a taxi ride received
# more than 20% tip
# ------------------------------------------------------------
# Allowed libraries: polars, scikit-learn (+ numpy, pyarrow)
# ------------------------------------------------------------

import polars as pl
import numpy as np
import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, roc_auc_score

# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ------------------------------------------------------------
# Load Data
# ------------------------------------------------------------
train = pl.read_parquet("train.parquet")
test = pl.read_parquet("test.parquet")
TARGET = "has_tipped_over_20"
assert TARGET in train.columns, "Target missing!"

# ------------------------------------------------------------
# Date / Duration Features
# ------------------------------------------------------------
def add_datetime_features(df: pl.DataFrame) -> pl.DataFrame:

    # Sicherstellen: Pickup Datetime
    if df["tpep_pickup_datetime"].dtype != pl.Datetime:
        df = df.with_columns(
            pl.col("tpep_pickup_datetime").str.strptime(pl.Datetime, strict=False).alias("pickup_dt")
        )
    else:
        df = df.rename({"tpep_pickup_datetime": "pickup_dt"})

    # Sicherstellen: Dropoff Datetime
    if df["tpep_dropoff_datetime"].dtype != pl.Datetime:
        df = df.with_columns(
            pl.col("tpep_dropoff_datetime").str.strptime(pl.Datetime, strict=False).alias("dropoff_dt")
        )
    else:
        df = df.rename({"tpep_dropoff_datetime": "dropoff_dt"})

    # Zeitfeatures & Dauer
    df = df.with_columns([
        pl.col("pickup_dt").dt.hour().alias("pickup_hour"),
        pl.col("pickup_dt").dt.weekday().alias("pickup_weekday"),
        ((pl.col("dropoff_dt") - pl.col("pickup_dt")).dt.total_seconds() / 60).alias("trip_duration_min"),
    ])

    return df


train = add_datetime_features(train)
test = add_datetime_features(test)

# ------------------------------------------------------------
# Additional Feature Engineering
# ------------------------------------------------------------
if "airport_fee" in train.columns:
    train = train.with_columns((pl.col("airport_fee") > 0).cast(pl.Int8).alias("is_airport_trip"))
    test = test.with_columns((pl.col("airport_fee") > 0).cast(pl.Int8).alias("is_airport_trip"))
else:
    train = train.with_columns(pl.lit(0).alias("is_airport_trip"))
    test = test.with_columns(pl.lit(0).alias("is_airport_trip"))

train = train.with_columns((pl.col("fare_amount") / (pl.col("trip_duration_min") + 1)).alias("fare_per_minute"))
test = test.with_columns((pl.col("fare_amount") / (pl.col("trip_duration_min") + 1)).alias("fare_per_minute"))

train = train.with_columns((pl.col("fare_amount") / (pl.col("trip_distance") + 0.1)).alias("fare_per_mile"))
test = test.with_columns((pl.col("fare_amount") / (pl.col("trip_distance") + 0.1)).alias("fare_per_mile"))

# ------------------------------------------------------------
# Segment by payment_type
# ------------------------------------------------------------
CASH_CODES = [0, 2, 3, 4, 5]
train_card = train.filter(~pl.col("payment_type").is_in(CASH_CODES))
test_card = test.filter(~pl.col("payment_type").is_in(CASH_CODES))
test_cash = test.filter(pl.col("payment_type").is_in(CASH_CODES))

# ------------------------------------------------------------
# Feature List
# ------------------------------------------------------------
features = [
    "passenger_count",
    "trip_distance",
    "fare_amount",
    "pickup_hour",
    "pickup_weekday",
    "trip_duration_min",
    "is_airport_trip",
    "fare_per_minute",
    "fare_per_mile",
]
features = [f for f in features if f in train.columns]

# ------------------------------------------------------------
# TIME-BASED VALIDATION SPLIT (important!)
# ------------------------------------------------------------
train_card = train_card.sort("pickup_dt")
split_idx = int(len(train_card) * 0.8)

train_df = train_card[:split_idx]
val_df = train_card[split_idx:]

X_train = train_df.select(features).to_numpy()
y_train = train_df[TARGET].to_numpy()
X_val = val_df.select(features).to_numpy()
y_val = val_df[TARGET].to_numpy()

X_train = np.nan_to_num(X_train, nan=0.0)
X_val = np.nan_to_num(X_val, nan=0.0)

# ------------------------------------------------------------
# Scale Features
# ------------------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# ------------------------------------------------------------
# Train Model (improved RF)
# ------------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=350,
    max_depth=None,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=SEED,
    n_jobs=-1,
)
model.fit(X_train, y_train)

val_prob = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, val_prob)
print(f"Validation AUROC: {auc:.4f}")

# ------------------------------------------------------------
# Find Best Threshold via F1
# ------------------------------------------------------------
prec, rec, th = precision_recall_curve(y_val, val_prob)
f1 = 2 * (prec * rec) / (prec + rec + 1e-6)
best_threshold = th[np.nanargmax(f1)]
print(f"Best threshold = {best_threshold:.3f}")

# ------------------------------------------------------------
# Predict test set
# ------------------------------------------------------------
X_test_card = test_card.select(features).to_numpy()
X_test_card = scaler.transform(np.nan_to_num(X_test_card, nan=0.0))
card_pred = (model.predict_proba(X_test_card)[:, 1] >= best_threshold).astype(int)

cash_pred = np.zeros(len(test_cash), dtype=int)

# ------------------------------------------------------------
# Create submission.csv
# ------------------------------------------------------------
if "row_nr" not in test.columns:
    test = test.with_columns(pl.arange(1, len(test) + 1).alias("row_nr"))

sub_card = test_card.select("row_nr").with_columns(pl.Series("has_tipped_over_20", card_pred))
sub_cash = test_cash.select("row_nr").with_columns(pl.Series("has_tipped_over_20", cash_pred))

submission = pl.concat([sub_card, sub_cash]).sort("row_nr")
submission.write_csv("submission.csv")

print("✅ submission.csv created successfully!")
