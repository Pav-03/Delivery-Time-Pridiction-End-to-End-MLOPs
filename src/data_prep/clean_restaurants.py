import logging
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd


PROJECT_ROOT=Path(__file__).resolve().parents[2]
RAW_DATA_PATH=PROJECT_ROOT/"data"/"raw"/"swiggy.csv"
OUT_DIR=PROJECT_ROOT/"data"/"interim"
CSV_OUT=OUT_DIR/"restaurants_clean.csv"
PARQUET_OUT=OUT_DIR/"restaurants_clean.parquet"

logger = logging.getLogger(__name__)

RNG = np.random.default_rng(42)

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

# Price Caps for tame outliers
PRICE_MIN = 50.0
PRICE_MAX = 3000.0

# Helpers
def _normalize_colnames(df: pd.DataFrame) -> pd.DataFrame:
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
     s = pd.to_numeric(series, errors="coerce")
     #treat <=0 as missing; impute later; then cap
     s = s.mask(s <= 0, np.nan)
     return s

def _assign_price_band(price: pd.Series) -> pd.Series:
     return pd.cut(
          price,
          bins=[0,250,500,800, np.inf],
          labels=["budget", "mid", "premium", "luxury"],
          include_lowest=True,

     )

def _clean_cuisines_to_list(series: pd.Series) -> pd.Series:
     def parse(x: str):
          if pd.isna(x):
               return []
          return [c.strip().lower() for c in str(x).split(",") if c.strip()]
     return series.apply(parse)
     
def _cuisines_list_to_str(series: pd.Series) -> pd.Series:
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


def _hash_to_coord(text: str, base_lat: float, base_lon: float) -> tuple[float, float]:
     h = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)
     lat_jitter = ((h % 1000) /1000 - 0.5)*0.06 #~+- 0.03
     lon_jitter = ((h % 1000) /1000 - 0.5)*0.06
     return base_lat + lat_jitter, base_lon + lon_jitter

def add_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lats, lons = [], []

    # sanity: make sure helper is a function
    assert callable(_hash_to_coord), (
        f"_hash_to_coord is not callable; got type={type(_hash_to_coord)}. "
        "Did a variable accidentally overwrite the function name?"
    )

    for i, row in df.iterrows():
        city = str(row["city"]).strip().lower()
        area = str(row["area"]).strip().lower()

        base = CITY_ANCHORS.get(city, (20.0, 78.0))
        if not (isinstance(base, (tuple, list)) and len(base) == 2):
            raise ValueError(
                f"CITY_ANCHORS[{city!r}] must be a (lat, lon) pair. Got: {base!r}"
            )

        latlon = _hash_to_coord(f"{city}_{area}", float(base[0]), float(base[1]))
        if not (isinstance(latlon, (tuple, list)) and len(latlon) == 2):
            raise ValueError(
                f"_hash_to_coord returned invalid value for city={city}, area={area}: {latlon!r}"
            )

        lats.append(float(latlon[0]))
        lons.append(float(latlon[1]))

    df["lat"] = lats
    df["lon"] = lons
    return df


def _validate_schema(df_raw: pd.DataFrame) -> pd.DataFrame:
     cols = set(df_raw.columns.str.strip().str.lower().str.replace(" ", "_"))
     missing = EXPECTED_COLS - cols
     if missing:
          raise KeyError(
               f"Input CSv missing required columns: {sorted(missing)}."
               f"Found: {sorted(list(cols))[:12]}..."

          )

# Main Trasformation

def clean_restaurants(input_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
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
    df["cuisines"] = cuisines_list                       # list-like for modeling
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

    unknown_city_ratio = 1.0 - df["city"].isin(CITY_ANCHORS.keys()).mean()
    if unknown_city_ratio > 0.2:
        logger.warning("Unknown city anchors for %.1f%% rows. Consider extending CITY_ANCHORS.",
                       100 * unknown_city_ratio)

    df["delivery_fee"] = (df["avg_menu_price"] * 0.05).clip(10, 60).round(0)

    # Remove duplicate restaurants
    before = len(df)
    df = df.drop_duplicates(subset=["restaurant_id"]).reset_index(drop=True)
    dups = before - len(df)
    if dups:
        logger.info("Removed %d duplicate restaurant_id rows", dups)

    logger.info("Cleaning complete. Final rows: %d, columns: %d", len(df), df.shape[1])
    return df

def main():
    # Configure logging only when running as script
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_clean = clean_restaurants()
    # CSV
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
