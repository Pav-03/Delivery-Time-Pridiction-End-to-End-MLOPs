# src/features/feature_eta.py
import logging
from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np
import json

from src.common.params import load_params
from src.common.utils import to_python_types

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORDERS_PATH = PROJECT_ROOT / "data" / "interim" / "orders.csv"
RESTAURANTS_PATH = PROJECT_ROOT / "data" / "interim" / "restaurants_clean.csv"
USERS_PATH = PROJECT_ROOT / "data" / "interim" / "users.csv"
ROUTES_PATH = PROJECT_ROOT / "data" / "interim" / "routes_eta.csv"
FEATURES_DIR = PROJECT_ROOT / "data" / "processed"
METRICS_PATH = PROJECT_ROOT / "metrics" / "features.json"

logger = logging.getLogger(__name__)

# ==================== FEATURE ENGINEERING PIPELINE ====================

def engineer_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features with cyclical encoding."""
    df = df.copy()
    
    # Basic temporal
    df["hour"] = df["order_created_at"].dt.hour
    df["day_of_week"] = df["order_created_at"].dt.dayofweek
    df["month"] = df["order_created_at"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_rush_hour"] = df["hour"].isin([7,8,9,12,13,18,19,20,21,22]).astype(int)
    
    # Cyclical encoding (hour → sin/cos for pattern learning)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    
    return df

def engineer_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create user behavioral aggregations (target: delivery time)."""
    user_stats = df.groupby("user_id").agg({
        "actual_delivery_time_min": ["mean", "std", "count"],
        "basket_value": "mean",
        "order_status": lambda x: (x == "completed").mean(),
        "distance_km": "mean"
    }).round(2)
    
    user_stats.columns = [
        "user_avg_delivery_time",
        "user_std_delivery_time",
        "user_order_count",
        "user_avg_basket",
        "user_completion_rate",
        "user_avg_distance"
    ]
    
    return df.merge(user_stats, left_on="user_id", right_index=True, how="left")

def engineer_restaurant_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create restaurant operational features."""
    rest_stats = df.groupby("restaurant_id").agg({
        "actual_delivery_time_min": ["mean", "std"],
        "order_status": lambda x: (x == "completed").mean(),
        "order_created_at": "count",
        "basket_value": "mean"
    }).round(2)
    
    rest_stats.columns = [
        "rest_avg_delivery_time",
        "rest_std_delivery_time",
        "rest_completion_rate",
        "rest_order_volume",
        "rest_avg_basket"
    ]
    
    return df.merge(rest_stats, left_on="restaurant_id", right_index=True, how="left")

def engineer_route_features(df: pd.DataFrame) -> pd.DataFrame:
    """Parse route_key with validation and error handling."""
    df = df.copy()
    
    # DEBUG: Check for malformed route_keys
    malformed = df[~df["route_key"].str.contains("_to_")]
    if not malformed.empty:
        logger.error("Found %d malformed route_keys without '_to_':\n%s", 
                     len(malformed), malformed["route_key"].head().to_string())
        raise ValueError("route_key must contain '_to_'. See log for examples.")
    
    # Split into (origin, dest)
    route_parts = df["route_key"].str.split("_to_", expand=True, n=1)
    if route_parts.shape[1] < 2:
        raise ValueError("split('_to_') did not produce 2 columns")
    
    # Extract dest_area
    df["dest_area"] = route_parts[1].fillna("UNKNOWN")
    
    # Split origin into (city, origin_area)
    origin_parts = route_parts[0].str.split("_", expand=True, n=1)
    if origin_parts.shape[1] < 2:
        logger.warning("Some origins don't have '_': %s", origin_parts.head())
        # If no underscore, treat entire string as city, origin_area = UNKNOWN
        df["city"] = origin_parts[0]
        df["origin_area"] = "UNKNOWN"
    else:
        df["city"] = origin_parts[0]
        df["origin_area"] = origin_parts[1].fillna("UNKNOWN")
    
    # Route statistics
    route_stats = df.groupby("route_key").agg({
        "traffic_level": ["mean", "std"],
        "actual_delivery_time_min": "mean",
        "distance_km": "first",
        "base_eta_minutes": "first"
    }).round(2)
    
    route_stats.columns = [
        "route_avg_traffic",
        "route_std_traffic",
        "route_historical_delivery_time",
        "route_distance",
        "route_base_eta"
    ]
    
    return df.merge(route_stats, left_on="route_key", right_index=True, how="left")

def engineer_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction features (where real signal lives)."""
    df = df.copy()
    
    # 1. Rush hour × Traffic (non-linear delay)
    df["rush_traffic_interaction"] = df["is_rush_hour"] * df["traffic_level"]
    
    # 2. Distance × Rain (exponential effect)
    df["distance_rain_interaction"] = df["distance_km"] * df["rain"]
    
    # 3. Restaurant load × Order complexity
    df["restaurant_complexity"] = df["rest_order_volume"] * df["num_items"]
    
    # 4. Traffic volatility (sudden spikes)
    df["traffic_squared"] = df["traffic_level"] ** 2
    
    # 5. High-value orders get priority (negative delay)
    df["is_high_value"] = (df["basket_value"] > 1500).astype(int)
    
    return df

def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create rolling window features for time-series patterns."""
    df = df.copy()
    
    # CRITICAL: Sort by datetime FIRST (before setting index)
    df = df.sort_values("order_created_at")
    
    # CRITICAL: Convert to datetime index for time-based rolling
    df = df.set_index("order_created_at")
    
    # Create a dummy column for counting orders
    df["rest_order_counter"] = 1
    
    # Restaurant load in last hour (kitchen pressure)
    df["rest_orders_last_1h"] = df.groupby("restaurant_id")["rest_order_counter"].transform(
        lambda x: x.rolling("1h", closed="left").sum()
    )
    
    # Route performance last 3h (trend)
    df["route_avg_delivery_last_3h"] = df.groupby("route_key")["actual_delivery_time_min"].transform(
        lambda x: x.rolling("3h", closed="left").mean()
    )
    
    # Traffic moving average last 2h
    df["traffic_ma_2h"] = df.groupby("route_key")["traffic_level"].transform(
        lambda x: x.rolling("2h", closed="left").mean()
    )
    
    # Drop the temporary counter
    df = df.drop(columns=["rest_order_counter"])
    
    # Reset index to restore original structure
    df = df.reset_index()
    
    # ✅ FILL NaN VALUES HERE (before validation)
    # First orders have no history → 0 is sensible (kitchen was idle)
    lag_features = ["rest_orders_last_1h", "route_avg_delivery_last_3h", "traffic_ma_2h"]
    df[lag_features] = df[lag_features].fillna(0)
    
    return df

def target_encode_time_series(df: pd.DataFrame, col: str, target: str) -> pd.DataFrame:
    """
    Target encode WITHOUT leakage (use only past data).
    This is CRITICAL for time series.
    """
    df = df.sort_values("order_created_at")
    df[f"{col}_encoded"] = np.nan
    
    # Compute rolling mean for each group
    for idx, group in df.groupby(col, sort=False):
        group_target = group[target].shift(1).expanding().mean()  # Shift 1 = only past
        df.loc[group.index, f"{col}_encoded"] = group_target
    
    # Fill NaN (first occurrence) with global mean
    global_mean = df[target].mean()
    df[f"{col}_encoded"] = df[f"{col}_encoded"].fillna(global_mean)
    
    return df

def validate_features(df: pd.DataFrame, feature_cols: list):
    """Validate feature quality before saving (only numeric columns)."""
    # Select only numeric columns for validation
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
    
    # Check for infinite values in numeric columns only
    if not numeric_cols.empty:
        has_inf = np.isinf(df[numeric_cols]).any().any()
        assert not has_inf, f"Infinite values found in numeric features"
    
    # Check for impossible delivery times (target column)
    assert df["actual_delivery_time_min"].min() >= 5, "Impossible delivery time detected"
    
    # Check feature coverage (missing values)
    null_pct = df[feature_cols].isnull().mean()
    if null_pct.max() > 0.3:
        raise ValueError(f"Feature {null_pct.idxmax()} has >30% missing values")
    
    # Log success
    logger.info("✅ Feature validation passed (checked %d numeric cols)", len(numeric_cols))

def main():
    """Main feature engineering pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    
    # Load data
    logger.info("Loading orders, users, restaurants, routes...")
    orders = pd.read_csv(ORDERS_PATH, parse_dates=["order_created_at"])
    restaurants = pd.read_csv(RESTAURANTS_PATH)
    users = pd.read_csv(USERS_PATH)
    routes = pd.read_csv(ROUTES_PATH, parse_dates=["timestamp"])
    
    initial_rows = len(orders)
    
    # Merge data with explicit suffixes and deduplication
    logger.info("Merging restaurant & user data...")
    
    # First merge: orders + restaurants
    df = orders.merge(restaurants, on="restaurant_id", how="left", suffixes=("", "_restaurant"))
    
    # Second merge: df + users
    df = df.merge(users, on="user_id", how="left", suffixes=("", "_user"))
    
    # ✅ Deduplicate columns (critical fix)
    duplicate_cols = df.columns[df.columns.duplicated()].tolist()
    if duplicate_cols:
        logger.warning(f"Removing duplicate columns: {duplicate_cols}")
        df = df.loc[:, ~df.columns.duplicated()]
    
    # Create features
    logger.info("Engineering temporal features...")
    df = engineer_temporal_features(df)
    
    logger.info("Engineering user features...")
    df = engineer_user_features(df)
    
    logger.info("Engineering restaurant features...")
    df = engineer_restaurant_features(df)
    
    logger.info("Engineering route features...")
    df = engineer_route_features(df)
    
    logger.info("Engineering interaction features...")
    df = engineer_interactions(df)
    
    logger.info("Engineering lag features...")
    df = create_lag_features(df)
    
    logger.info("Target encoding (no leakage)...")
    df = target_encode_time_series(df, "restaurant_id", "actual_delivery_time_min")
    df = target_encode_time_series(df, "user_id", "actual_delivery_time_min")
    
    # Define feature columns
    feature_cols = [col for col in df.columns if col not in [
    "order_id", "order_created_at", "route_key", "order_status",
    "user_id", "restaurant_id", "actual_delivery_time_min"]]
    
    # Validate features
    validate_features(df, feature_cols)
    
    # Time-based split (CRITICAL for ETA prediction)
    logger.info("Splitting data by time...")
    split_date = df["order_created_at"].quantile(0.7)
    train = df[df["order_created_at"] < split_date]
    valid = df[df["order_created_at"] >= split_date]
    
    # Save features
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    
    train[feature_cols + ["actual_delivery_time_min"]].to_parquet(
        FEATURES_DIR / "train.parquet", index=False
    )
    valid[feature_cols + ["actual_delivery_time_min"]].to_parquet(
        FEATURES_DIR / "val.parquet", index=False
    )
    
    # Generate metrics
    metrics = {
        "initial_rows": initial_rows,
        "final_rows": len(df),
        "train_rows": len(train),
        "val_rows": len(valid),
        "features_created": len(feature_cols),
        "feature_memory_mb": df[feature_cols].memory_usage(deep=True).sum() / 1024 / 1024,
        "target_mean": df["actual_delivery_time_min"].mean(),
        "target_std": df["actual_delivery_time_min"].std()
    }
    
    # Save metrics
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(to_python_types(metrics), f, indent=2)
    
    logger.info("✅ Feature engineering complete!")
    logger.info(f"📊 Features: {len(feature_cols)} | Train: {len(train):,} | Val: {len(valid):,}")
    logger.info(f"📈 Metrics: {METRICS_PATH}")

if __name__ == "__main__":
    main()