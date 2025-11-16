import logging
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from typing import Tuple
from src.common.params import load_params

_PARAMS = load_params()


PROJECT_ROOT=Path(__file__).resolve().parents[2]
RAW_DATA_PATH=PROJECT_ROOT/"data"/"raw"/"swiggy.csv"
OUT_DIR=PROJECT_ROOT/"data"/"interim"
CSV_OUT=OUT_DIR/"restaurants_clean.csv"
PARQUET_OUT=OUT_DIR/"restaurants_clean.parquet"

logger = logging.getLogger(__name__)

# Get constants from params
RNG = np.random.default_rng(_PARAMS["seed"])
PRICE_MIN = _PARAMS["restaurants"]["price"]["min"]
PRICE_MAX = _PARAMS["restaurants"]["price"]["max"]
DELIVERY_FEE_BASE_MIN = _PARAMS["restaurants"]["delivery_fee"]["base_min"]
DELIVERY_FEE_BASE_MAX = _PARAMS["restaurants"]["delivery_fee"]["base_max"]
DELIVERY_FEE_VARIATION = _PARAMS["restaurants"]["delivery_fee"]["variation"]
GEO_JITTER_DEGREES = _PARAMS["restaurants"]["location_jitter"]


#Config
EXPECTED_COLS = {
    "id", "area", "city", "restaurant", "price",
    "avg_ratings", "total_ratings", "food_type", "address", "delivery_time"
}

# canonical city name for anchors and cosistent grouping
CITY_ALIASES = {
    "bangalore": "bangalore",
    "bengaluru": "bangalore",
    "blr": "bangalore",
    "b'lore": "bangalore",
    "mumbai": "mumbai",
    "bombay": "mumbai",
    "delhi": "delhi",
    "new delhi": "delhi",
    "hyderabad": "hyderabad",
    "pune": "pune",
    "kolkata": "kolkata",
    "calcutta": "kolkata",
    "chennai": "chennai",
    "madras": "chennai",
    "ahmedabad": "ahmedabad",
    "surat": "surat",
}

# Very Rough city anchors
CITY_ANCHORS = {
    "bangalore": (12.9716, 77.5946),
    "mumbai": (19.0760, 72.8777),
    "delhi": (28.6139, 77.2090),
    "hyderabad": (17.3850, 78.4867),
    "pune": (18.5204, 73.8567),
    "kolkata": (22.5726, 88.3639),
    "chennai": (13.0827, 80.2707),
    "ahmedabad": (23.0225, 72.5714),
    "surat": (21.1702, 72.8311),
}

# Helpers
def _normalize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    """ Normalize the columns to lowercase with underscore."""
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")

    )
    return df

def _canonical_city(s: pd.Series) -> pd.Series:
     """
    Map messy city strings to canonical keys used for anchors/grouping.
    Keeps original in 'city_raw'; returns normalized 'city'.
    """
     s_norm = s.astype(str).str.strip().str.lower()
     return s_norm.map(lambda x: CITY_ALIASES.get(x, x))

def _clean_price(series: pd.Series) -> pd.Series:
     """ Clean price column: coerce to numeric, trat <=0 as NaN."""
     s = pd.to_numeric(series, errors="coerce")
     #treat <=0 as missing; impute later; then cap
     s = s.mask(s <= 0, np.nan)
     return s

def _assign_price_band(price: pd.Series) -> pd.Series:
     """ Assign price band labels based on numeric ranges."""
     return pd.cut(
          price,
          bins=[0,250,500,800, np.inf],
          labels=["budget", "mid", "premium", "luxury"],
          include_lowest=True,

     )

def _clean_cuisines_to_list(series: pd.Series) -> pd.Series:
     """ Parse cuisines string into clean list."""
     def parse(x: str):
          if pd.isna(x):
               return []
          return [c.strip().lower() for c in str(x).split(",") if c.strip()]
     return series.apply(parse)
     
def _cuisines_list_to_str(series: pd.Series) -> pd.Series:
    """ Convert cuisines list to display string."""
    import ast

    def to_str(val):
        # Case 1: already a Python list
        if isinstance(val, list):
            return ", ".join([str(c).strip().title() for c in val if str(c).strip()])

        # Case 2: NaN
        if pd.isna(val):
            return ""

        # Case 3: string that might be a list or CSV
        s = str(val).strip()
        # try to parse "['north indian', 'chinese']"
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return ", ".join([str(c).strip().title() for c in parsed if str(c).strip()])
            except Exception:
                pass
        # fallback: treat as comma-separated
        parts = [p.strip().title() for p in s.split(",") if p.strip()]
        return ", ".join(parts)

    return series.apply(to_str)


def _hash_to_coord_vectorized(keys: np.ndarray, base_lats: np.ndarray, base_lons: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized version of coordinate generation.
    Generates deterministic jitter for each unique key.
    """
    def hash_single(key: str, base_lat: float, base_lon: float) -> Tuple[float, float]:
        h = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16)
        # Use different parts of hash for lat/lon to avoid correlation
        lat_jitter = ((h % 1000) / 1000 - 0.5) * GEO_JITTER_DEGREES
        lon_jitter = ((h // 1000 % 1000) / 1000 - 0.5) * GEO_JITTER_DEGREES
        return base_lat + lat_jitter, base_lon + lon_jitter
    
    vfunc = np.vectorize(hash_single)
    return vfunc(keys, base_lats, base_lons)

def add_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add latitude/longitude features with intra-area variance.
    FAILS if any city is missing from CITY_ANCHORS.
    """
    df = df.copy()
    
    # 🎯 FAIL FAST: Check for unknown cities
    unique_cities = set(df["city"].unique())
    known_cities = set(CITY_ANCHORS.keys())
    unknown_cities = unique_cities - known_cities
    
    if unknown_cities:
        raise ValueError(
            f"Found unknown cities not in CITY_ANCHORS: {sorted(unknown_cities)}\n"
            f"Please add them to CITY_ANCHORS dictionary with accurate (lat, lon)."
        )
    
    # Create unique location keys
    keys = (df["city"].astype(str) + "_" + 
            df["area"].astype(str) + "_" + 
            df["name"].astype(str)).str.lower()
    
    # Get base coordinates (now guaranteed to exist)
    bases = df["city"].map(CITY_ANCHORS)
    
    # Generate coordinates
    lats, lons = _hash_to_coord_vectorized(
        keys.values,
        bases.map(lambda x: x[0]).values,
        bases.map(lambda x: x[1]).values
    )
    
    df["lat"] = lats
    df["lon"] = lons
    
    # Validate
    assert df["lat"].between(-90, 90).all(), "Invalid latitude"
    assert df["lon"].between(-180, 180).all(), "Invalid longitude"
    
    return df


def _validate_schema(df_raw: pd.DataFrame) -> None:
     
     """ Validate input schema has required columns"""
     cols = set(df_raw.columns.str.strip().str.lower().str.replace(" ", "_"))
     missing = EXPECTED_COLS - cols
     if missing:
          raise KeyError(
               f"Input CSv missing required columns: {sorted(missing)}."
               f"Found: {sorted(list(cols))[:12]}..."

          )
     
def _validate_restaurant_ids(df: pd.DataFrame) -> None:

    """Ensure restaurant_id is unique before processing."""

    n_unique = df["restaurant_id"].nunique()
    n_total = len(df)
    if n_unique != n_total:
        duplicates = df["restaurant_id"].value_counts().head()
        raise ValueError(
            f"Found {n_total - n_unique} duplicate restaurant_id values. "
            f"Total rows: {n_total}, Unique IDs: {n_unique}. "
            f"Example duplicates:\n{duplicates}"
        )

# Main Trasformation

def clean_restaurants(input_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """ Main Cleaning pipeline for restaurant data
    Resets RNG for reproducibility on every call"""
    
    global RNG
    RNG = np.random.default_rng(42)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logger.info("Loading raw restaurants CSV from %s", input_path)
    df_raw = pd.read_csv(input_path, encoding="utf-8", engine="python")
    _validate_schema(df_raw)

    logger.info("Normalizing column names")
    df = _normalize_colnames(df_raw)

    # Rename to internal schema
    df = df.rename(columns={
        "id": "restaurant_id",
        "restaurant": "name",
        "avg_ratings": "avg_rating",
        "total_ratings": "num_ratings",
        "food_type": "cuisines_raw",
        "delivery_time": "base_delivery_time_min",
    })

    _validate_restaurant_ids(df)

    n0 = len(df)
    logger.info("Rows loaded: %d", n0)

    # Keep raw city; create canonical city
    df["city_raw"] = df["city"].astype(str)
    df["city"] = _canonical_city(df["city"])

    # Basic cleaning
    df["area"] = df["area"].astype(str).str.strip()
    df["name"] = df["name"].astype(str).str.strip()

    # Price cleaning + per-city imputation
    df["avg_menu_price"] = _clean_price(df["price"])
    df.drop(columns=["price"], inplace=True)

    # Log missing price by city.
    null_by_city = (
        df.assign(isnull=df["avg_menu_price"].isna())
          .groupby("city")["isnull"].mean()
          .sort_values(ascending=False)
    )
    top_cities_with_nulls = null_by_city[null_by_city > 0].head(5)
    if not top_cities_with_nulls.empty:
        logger.warning("Price missing ratio by city (top with nulls):\n%s",
                       top_cities_with_nulls.to_string())

    df["avg_menu_price"] = df.groupby("city")["avg_menu_price"].transform(
        lambda x: x.fillna(x.median())
    )

    # Cap extreme prices then band
    df["avg_menu_price"] = df["avg_menu_price"].clip(lower=PRICE_MIN, upper=PRICE_MAX)
    df["price_band"] = _assign_price_band(df["avg_menu_price"])

    # Cuisines: keep list (model) and pretty string (UI)
    cuisines_list = _clean_cuisines_to_list(df["cuisines_raw"])
    df["cuisines_str"] = _cuisines_list_to_str(cuisines_list)  # display string
    df.drop(columns=["cuisines_raw"], inplace=True)

    # Sanity filter on base delivery time
    before = len(df)
    df = df.dropna(subset=["base_delivery_time_min"])
    df = df[df["base_delivery_time_min"] > 0]
    dropped = before - len(df)
    drop_rate = dropped / max(before, 1)
    if drop_rate > 0.2:
        logger.warning("Dropped %d rows (%.1f%%) due to invalid base_delivery_time_min",
                       dropped, 100 * drop_rate)

    # Geo + synthetic delivery fee
    df = add_geo_features(df)

    # Warn about unknown city.
    unknown_city_ratio = 1.0 - df["city"].isin(CITY_ANCHORS.keys()).mean()
    if unknown_city_ratio > 0.2:
        logger.warning("Unknown city anchors for %.1f%% rows. Consider extending CITY_ANCHORS.",
                       100 * unknown_city_ratio)

    # Delivery Fees.
    df["delivery_fee"] = RNG.uniform(
        DELIVERY_FEE_BASE_MIN, 
        DELIVERY_FEE_BASE_MAX, 
        size=len(df)
    ) + RNG.normal(0, DELIVERY_FEE_VARIATION, size=len(df))
    df["delivery_fee"] = df["delivery_fee"].clip(15, 80).round(2)

    # Remove duplicate restaurants
    before = len(df)
    df = df.drop_duplicates(subset=["restaurant_id"]).reset_index(drop=True)
    dups = before - len(df)
    if dups:
        logger.info("Removed %d duplicate restaurant_id rows", dups)

    logger.info("Cleaning complete. Final rows: %d, columns: %d", len(df), df.shape[1])
    return df

# Validation Report Function
def generate_validation_report(df: pd.DataFrame) -> dict:
    """Generate a data quality report for the cleaned dataset."""
    report = {
        "total_restaurants": len(df),
        "unique_restaurant_ids": df["restaurant_id"].nunique(),
        "cities_covered": df["city"].nunique(),
        "avg_menu_price_mean": df["avg_menu_price"].mean(),
        "avg_menu_price_std": df["avg_menu_price"].std(),
        "missing_coordinates": df[["lat", "lon"]].isna().sum().to_dict(),
        "invalid_delivery_time": (df["base_delivery_time_min"] <= 0).sum(),
        "price_outliers": ((df["avg_menu_price"] < PRICE_MIN) | (df["avg_menu_price"] > PRICE_MAX)).sum(),
        "delivery_fee_stats": {
            "min": df["delivery_fee"].min(),
            "max": df["delivery_fee"].max(),
            "mean": df["delivery_fee"].mean(),
        }
    }
    
    # Pass/fail criteria
    report["passed"] = all([
        report["total_restaurants"] == report["unique_restaurant_ids"],
        report["missing_coordinates"]["lat"] == 0,
        report["missing_coordinates"]["lon"] == 0,
        report["invalid_delivery_time"] == 0,
        report["price_outliers"] == 0,
    ])
    
    return report

def main():
    """ Main Excecution function"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_clean = clean_restaurants()

    # Generate and log validation report
    report = generate_validation_report(df_clean)
    logger.info("Data Validation Report:\n%s", pd.Series(report).to_string())
    
    if not report["passed"]:
        logger.error("Data validation FAILED. Check issues above.")
        # Don't exit - let user decide if they want to proceed
    else:
        logger.info("Data validation PASSED.")

    # Save CSV

    df_clean.to_csv(CSV_OUT, index=False)
    logger.info("Saved CSV -> %s", CSV_OUT)

    # Parquet (fast). Requires pyarrow or fastparquet
    try:
        df_clean.to_parquet(PARQUET_OUT, index=False)
        logger.info("Saved Parquet -> %s", PARQUET_OUT)
    except Exception as e:
        logger.warning("Parquet save failed (install pyarrow?): %s", e)

if __name__ == "__main__":
    main()
