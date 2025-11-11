# 123456_challenge_01.py
# ------------------------------------------------------------
# ML Challenge: Predict if a taxi ride received > 20% tip
# Allowed: polars, scikit-learn, numpy
# ------------------------------------------------------------

import polars as pl
import numpy as np
import random

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ----------------------------
# 1. Reproducibility
# ----------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ----------------------------
# 2. Load Data
# ----------------------------
train = pl.read_parquet("train.parquet")
test = pl.read_parquet("test.parquet")

TARGET = "has_tipped_over_20"
assert TARGET in train.columns

# ----------------------------
# 3. Datetime Features
# ----------------------------
def add_datetime_features(df: pl.DataFrame) -> pl.DataFrame:
    # Parse datetime only if needed
    if df["tpep_pickup_datetime"].dtype != pl.Datetime:
        df = df.with_columns([
            pl.col("tpep_pickup_datetime").str.strptime(pl.Datetime, strict=False).alias("pickup_dt"),
            pl.col("tpep_dropoff_datetime").str.strptime(pl.Datetime, strict=False).alias("dropoff_dt"),
        ])
    else:
        df = df.rename({
            "tpep_pickup_datetime": "pickup_dt",
            "tpep_dropoff_datetime": "dropoff_dt",
        })

    return df.with_columns([
        pl.col("pickup_dt").dt.hour().alias("pickup_hour"),
        pl.col("pickup_dt").dt.weekday().alias("pickup_weekday"),
        ((pl.col("dropoff_dt") - pl.col("pickup_dt")).dt.total_seconds() / 60).alias("trip_duration_min"),
    ])

train = add_datetime_features(train)
test = add_datetime_features(test)

# ----------------------------
# 4. Additional Features
# ----------------------------
train = train.with_columns((pl.col("airport_fee") > 0).cast(pl.Int8).alias("is_airport_trip") if "airport_fee" in train.columns else pl.lit(0).alias("is_airport_trip"))
test = test.with_columns((pl.col("airport_fee") > 0).cast(pl.Int8).alias("is_airport_trip") if "airport_fee" in test.columns else pl.lit(0).alias("is_airport_trip"))

train = train.with_columns((pl.col("fare_amount") / (pl.col("trip_distance") + 1e-6)).alias("price_per_mile"))
test = test.with_columns((pl.col("fare_amount") / (pl.col("trip_distance") + 1e-6)).alias("price_per_mile"))

# ----------------------------
# 5. Split Cash vs Card
# ----------------------------
CASH_CODES = [0, 2, 3, 4, 5]
train_card = train.filter(~pl.col("payment_type").is_in(CASH_CODES))
test_card = test.filter(~pl.col("payment_type").is_in(CASH_CODES))
test_cash = test.filter(pl.col("payment_type").is_in(CASH_CODES))

# ----------------------------
# 6. Feature Selection
# ----------------------------
feature_cols = [
    "passenger_count",
    "trip_distance",
    "fare_amount",
    "pickup_hour",
    "pickup_weekday",
    "trip_duration_min",
    "is_airport_trip",
    "price_per_mile",
]
feature_cols = [c for c in feature_cols if c in train.columns]

# ----------------------------
# 7. Training (Card only)
# ----------------------------
X = train_card.select(feature_cols).to_numpy()
y = train_card[TARGET].to_numpy()
X = np.nan_to_num(X, nan=0.0)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

model = HistGradientBoostingClassifier(
    learning_rate=0.08,
    max_depth=6,
    max_leaf_nodes=64,
    min_samples_leaf=25,
    l2_regularization=1.0,
    random_state=SEED,
)

model.fit(X_train, y_train)
val_pred = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, val_pred)
print(f"Validation AUROC (cards): {auc:.4f}")

# ----------------------------
# 8. Predict Test
# ----------------------------
X_test_card = test_card.select(feature_cols).to_numpy()
X_test_card = np.nan_to_num(X_test_card, nan=0.0)
X_test_card = scaler.transform(X_test_card)

card_pred = (model.predict_proba(X_test_card)[:, 1] >= 0.5).astype(int)
cash_pred = np.zeros(len(test_cash), dtype=int)

# ----------------------------
# 9. Submission
# ----------------------------
if "row_nr" not in test.columns:
    test = test.with_columns(pl.arange(1, len(test) + 1).alias("row_nr"))

sub_card = test_card.select("row_nr").with_columns(pl.Series("has_tipped_over_20", card_pred))
sub_cash = test_cash.select("row_nr").with_columns(pl.Series("has_tipped_over_20", cash_pred))

submission = pl.concat([sub_card, sub_cash]).sort("row_nr")
submission.write_csv("submission.csv")

print("🎉 submission.csv created (HistGB only, binary output)")
