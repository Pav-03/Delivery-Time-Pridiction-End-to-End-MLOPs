import logging
import pandas as pd
import numpy as np
from pathlib import Path
import hashlib

from src.common.params import load_params
from src.common.utils import to_python_types


_PARAMS = load_params()

# Derived constant form params 

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESTAURANTS_PATH = PROJECT_ROOT / _PARAMS["paths"]["interim"] / f"{_PARAMS['restaurants']['output_name']}.csv"
OUT_DIR = PROJECT_ROOT /_PARAMS["paths"]["interim"]
CSV_OUT = OUT_DIR / f"{_PARAMS['users']['output_name']}.csv"
PARQUET_OUT = OUT_DIR / f"{_PARAMS['users']['output_name']}.parquet"

# Logic constants (not config)
USER_ID_PREFIX = "U"

logger = logging.getLogger(__name__)

# Vectorised coordinated generation (mirroring restaurants)

def _hash_to_coord_vec(keys: np.ndarray, base_lats: np.ndarray, base_lons: np.ndarray):
    """vectorised coordinate generation with deterministic jitter."""

    def hash_single(key: str, base_lat: float, base_lon: float):
        h = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16)
        lat_jitter = ((h%1000)/1000-0.5)*_PARAMS["restaurants"]["location_jitter"]
        lon_jitter = ((h%1000)/1000-0.5)*_PARAMS["restaurants"]["location_jitter"]

        return base_lat + lat_jitter, base_lon + lon_jitter
    
    vfunc = np.vectorize(hash_single)
    return vfunc(keys, base_lats, base_lons)

# Pre-compute locaion weights

def precompute_location_weights(restaurants: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:

    """
    Pre-Compute city and area weights for fast vectorised sampling.
    returns: (city_weights, area_weights_df)
    """
     
    # city weights propostional to restaurants density

    city_weights = restaurants["city"].value_counts(normalize=True)

    # Area weights: within each city, proportional to restaurant density

    area_weights = restaurants.groupby(["city", "area"])["restaurant_id"].count().rename("restaurants_count").reset_index()

    area_weights["weight"] = (
        area_weights.groupby("city")["restaurants_count"].transform(lambda x: x / x.sum())

    )

    return city_weights, area_weights


# VEctorised user generation (O(n) -> O(1))

def generate_users(restaurants_path: Path, n_users: int, max_fav_cuisines: int, seed: int) -> pd.DataFrame:
    """
    Generate synthetic users deterministically,
    uses vectorised operation for speed (10k users in <1 secons)
    """

    # use seed from param file for reproducibility

    rng = np.random.default_rng(seed)
    restaurants = pd.read_csv(restaurants_path)

    if "cuisines_str" not in restaurants.columns:
        raise KeyError("restaurants_clean.csv missing 'cuisines_str' column. run clean_restaurants.py first.")
    
    # Pre-Compute weights
    city_weights, area_weights = precompute_location_weights(restaurants)

    # Vectorized city selection

    cities = rng.choice(city_weights.index, size = n_users, p = city_weights.values)

    # Vectorised area selection 
    areas = []
    for city in cities:
        city_areas = area_weights[area_weights["city"] == city]
        area = rng.choice(city_areas["area"], p=city_areas["weight"])
        areas.append(area)

    # create user ids
    user_ids = [f"{USER_ID_PREFIX}{uid:05d}" for uid in range(1, n_users+1)]

    # Vectorized budget assignment based on local median prices
    budgets = []
    for i, (city, area) in enumerate(zip(cities, areas)):
        local_prices = restaurants[
            (restaurants["city"] == city) & (restaurants["area"] == area)
        ]["avg_menu_price"]
        
        if len(local_prices) == 0:
            budgets.append("mid")
        else:
            med = float(local_prices.median())
            if med <= 250:
                budgets.append(rng.choice(["budget", "mid"], p=[0.7, 0.3]))
            elif med <= 500:
                budgets.append(rng.choice(["budget", "mid", "premium"], p=[0.2, 0.6, 0.2]))
            else:
                budgets.append(rng.choice(["mid", "premium", "luxury"], p=[0.2, 0.5, 0.3]))
    
    # Vectorized favorite cuisines
    fav_cuisines_list = []
    fav_cuisines_str = []
    
    for i, (city, area) in enumerate(zip(cities, areas)):
        # Use cuisines_str and split
        local_cuisines = restaurants[
            (restaurants["city"] == city) & (restaurants["area"] == area)
        ]["cuisines_str"].dropna()
        
        # Split and flatten
        all_cuisines = []
        for c in local_cuisines:
            all_cuisines.extend([x.strip().lower() for x in c.split(",") if x.strip()])
        
        all_cuisines = sorted(set(all_cuisines))  # Unique
        
        if not all_cuisines:
            fav_list = []
        else:
            k = min(max_fav_cuisines, len(all_cuisines))
            fav_list = rng.choice(all_cuisines, size=k, replace=False).tolist()
        
        #fav_cuisines_list.append(fav_list)
        fav_cuisines_str.append(", ".join([c.title() for c in fav_list]))
    
    # Create DataFrame (vectorized)
    df = pd.DataFrame({
        "user_id": user_ids,
        "city": cities,
        "home_area": areas,
        "budget_segment": budgets,
        #"fav_cuisines_list": fav_cuisines_list,
        "fav_cuisines": fav_cuisines_str,
    })
    
    # Add home coordinates (deterministic hash)
    location_keys = np.array([f"{c}_{a}_{uid}" for c, a, uid in zip(cities, areas, user_ids)])
    city_coords = df["city"].map({
        "bangalore": (12.9716, 77.5946),
        "mumbai": (19.0760, 72.8777),
        "delhi": (28.6139, 77.2090),
        "hyderabad": (17.3850, 78.4867),
        "pune": (18.5204, 73.8567),
        "kolkata": (22.5726, 88.3639),
        "chennai": (13.0827, 80.2707),
        "ahmedabad": (23.0225, 72.5714),
        "surat": (21.1702, 72.8311),
    }).apply(lambda x: x if pd.notna(x) else (20.0, 78.0))
    
    home_lats, home_lons = _hash_to_coord_vec(
        location_keys,
        city_coords.map(lambda x: x[0]).values,
        city_coords.map(lambda x: x[1]).values
    )
    
    df["home_lat"] = home_lats
    df["home_lon"] = home_lons

    # VALIDATION: Ensure all coordinates are within bounds
    bounds = _PARAMS["restaurants"]["bounds"]
    if not df["home_lat"].between(bounds["lat_min"], bounds["lat_max"]).all():
        raise AssertionError(f"Users have latitudes outside bounds {bounds['lat_min']}-{bounds['lat_max']}")
    if not df["home_lon"].between(bounds["lon_min"], bounds["lon_max"]).all():
        raise AssertionError(f"Users have longitudes outside bounds {bounds['lon_min']}-{bounds['lon_max']}")
    
    
    return df

def main():
    """Run the user factory."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    
    # FIXURE #1: Fail-fast validation (prevents silent skips)
    if "bounds" not in _PARAMS["restaurants"]:
        logger.error("Missing 'bounds' in params.yaml! Required for geographic validation.")
        raise KeyError("Config must include restaurants.bounds: {lat_min, lat_max, lon_min, lon_max}")
    
    # FIXURE #2: Type checking (prevent typos like "8" instead of 8.0)
    bounds = _PARAMS["restaurants"]["bounds"]
    required_keys = ["lat_min", "lat_max", "lon_min", "lon_max"]
    for key in required_keys:
        if key not in bounds:
            raise KeyError(f"bounds missing key: {key}")
        if not isinstance(bounds[key], (int, float)):
            raise TypeError(f"bounds.{key} must be numeric, got {type(bounds[key])}")
    
    # Use params directly from _PARAMS
    n_users = _PARAMS["users"]["total"]
    max_cuisines = _PARAMS["users"]["max_fav_cuisines"]
    seed = _PARAMS["seed"]
    
    logger.info(f"🚀 Generating {n_users:,} users...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate users
    df = generate_users(RESTAURANTS_PATH, n_users, max_cuisines, seed)
    
    # **CRITICAL VALIDATION**
    assert df["user_id"].is_unique, "❌ Duplicate user IDs!"
    assert df["city"].notna().all(), "❌ Missing cities!"
    assert df["home_lat"].between(-90, 90).all(), "❌ Invalid latitude!"
    assert df["home_lon"].between(-180, 180).all(), "❌ Invalid longitude!"
    
    # FIXURE #3: Data validation with clear error messages
    assert df["home_lat"].between(bounds["lat_min"], bounds["lat_max"]).all(), \
        f"Users have latitudes outside bounds {bounds['lat_min']}-{bounds['lat_max']}"
    assert df["home_lon"].between(bounds["lon_min"], bounds["lon_max"]).all(), \
        f"Users have longitudes outside bounds {bounds['lon_min']}-{bounds['lon_max']}"
    
    logger.info("✅ All users within geographic bounds")
    
    # ==================== METRICS GENERATION ====================
    # Generate quality metrics for tracking
    metrics = {
        "total_users": len(df),
        "unique_user_ids": df["user_id"].nunique(),
        "cities_covered": df["city"].nunique(),
        "budget_segments": df["budget_segment"].value_counts().to_dict(),
        "avg_fav_cuisines": df["fav_cuisines"].str.split(",").apply(lambda x: len(x) if x != [''] else 0).mean(),
        "geographic_bounds_check": {
            "lat_min": float(df["home_lat"].min()),
            "lat_max": float(df["home_lat"].max()),
            "lon_min": float(df["home_lon"].min()),
            "lon_max": float(df["home_lon"].max()),
        },
        "missing_values": {
            "city": int(df["city"].isna().sum()),
            "home_area": int(df["home_area"].isna().sum()),
            "budget_segment": int(df["budget_segment"].isna().sum()),
            "fav_cuisines": int(df["fav_cuisines"].isna().sum()),
        }
    }

    # Write metrics to JSON file
    import json
    from src.common.utils import to_python_types
    
    metrics_dir = PROJECT_ROOT / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    with open(metrics_dir / "users.json", "w") as f:
        json.dump(to_python_types(metrics), f, indent=2)
    
    logger.info("📊 Metrics saved: %s", metrics_dir / "users.json")
    # ==================== END METRICS ====================
    
    # Log distributions
    logger.info("\n📊 User Distribution:\n%s", df["city"].value_counts().to_string())
    logger.info("\n💰 Budget Segments:\n%s", df["budget_segment"].value_counts().to_string())
    
    # Save
    logger.info(f"💾 Saving to {CSV_OUT} (shape: {df.shape})")
    df.to_csv(CSV_OUT, index=False)
    
    try:
        df.to_parquet(PARQUET_OUT, index=False)
        logger.info(f"💾 Parquet saved: {PARQUET_OUT}")
    except Exception as e:
        logger.warning(f"⚠️ Parquet failed: {e} (install pyarrow)")
    
    logger.info("✅ User generation complete!")

if __name__ == "__main__":
    main()



