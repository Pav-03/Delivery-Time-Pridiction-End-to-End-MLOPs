import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from src.data_prep import clean_restaurants as cr

# Fixture : Load data once, reuse across test for fast run
@pytest.fixture(scope="module")
def df_clean() -> pd.DataFrame:
    """Load cleaned restaurant data once for all tests."""
    csv_path = Path("data/interim/restaurants_clean.csv")
    if not csv_path.exists():
        pytest.fail(f"Cleaned data not found at {csv_path}. Run 'make clean_restaurants' first.")
    
    df = pd.read_csv(csv_path)

    # Parse data types explicitly (CSV loses type info)
    df["restaurant_id"] = df["restaurant_id"].astype(str)
    df["lat"] = df["lat"].astype(float)  # ✅ Ensure float
    df["lon"] = df["lon"].astype(float)
    df["base_delivery_time_min"] = df["base_delivery_time_min"].astype(float)  # ✅ Force float
    df["avg_menu_price"] = df["avg_menu_price"].astype(float)
    df["delivery_fee"] = df["delivery_fee"].astype(float)
    
    return df

# Unit Test : Schema and quality checks

def test_schema_columns(df_clean):
    """All required columns exist."""
    required_cols = {
        "restaurant_id", "name", "city", "city_raw", "area",
        "avg_rating", "num_ratings", "avg_menu_price", "price_band",
        "cuisines_str", "base_delivery_time_min", "lat", "lon", "delivery_fee"
    }
    missing = required_cols - set(df_clean.columns)
    assert not missing, f"Missing required columns: {missing}"

def test_schema_no_extra_columns(df_clean):
    """Ensure no unexpected columns (catch duplicates)."""
    expected = {
        "restaurant_id", "name", "city", "city_raw", "area", "address",
        "avg_rating", "num_ratings", "avg_menu_price", "price_band",
        "cuisines_str", "base_delivery_time_min", "lat", "lon", "delivery_fee"
    }
    extra = set(df_clean.columns) - expected
    assert not extra, f"Unexpected columns found: {extra}"

def test_restaurant_id_uniqueness(df_clean):
    """No duplicate restaurants (primary key integrity)."""
    n_dup = len(df_clean) - df_clean["restaurant_id"].nunique()
    assert n_dup == 0, f"Found {n_dup} duplicate restaurant_id values"

def test_no_missing_critical_data(df_clean):
    """No nulls in fields that would break ETA model."""
    critical_cols = ["restaurant_id", "city", "lat", "lon", "base_delivery_time_min", "delivery_fee"]
    missing = df_clean[critical_cols].isnull().sum()
    assert missing.sum() == 0, f"Missing values in critical columns:\n{missing[missing > 0]}"

def test_delivery_fee_not_perfectly_correlated_with_price(df_clean):
    """
    delivery_fee must NOT be a deterministic function of price.
    
    If correlation is too high (>0.95), it means our "fix" didn't work.
    This is a **data leakage** test - prevents model from cheating.
    """
    correlation = df_clean["delivery_fee"].corr(df_clean["avg_menu_price"])
    assert abs(correlation) < 0.95, (
        f"delivery_fee is too correlated with price (r={correlation:.3f}). "
        "This would cause data leakage in ETA model."
    )

def test_coordinates_are_realistic_for_india(df_clean):
    """Coordinates must be within India's geographic bounds."""
    INDIA_LAT_MIN, INDIA_LAT_MAX = 6.0, 38.0  # Kanyakumari to Kashmir
    INDIA_LON_MIN, INDIA_LON_MAX = 68.0, 97.0  # Pakistan border to Myanmar
    
    assert df_clean["lat"].between(INDIA_LAT_MIN, INDIA_LAT_MAX).all(), \
        f"Latitudes outside India: {df_clean['lat'].min():.2f}° to {df_clean['lat'].max():.2f}°"
    assert df_clean["lon"].between(INDIA_LON_MIN, INDIA_LON_MAX).all(), \
        f"Longitudes outside India: {df_clean['lon'].min():.2f}° to {df_clean['lon'].max():.2f}°"

def test_data_types(df_clean):
    """Columns have correct data types."""
    type_checks = {
        "restaurant_id": "object",
        "lat": "float64",
        "lon": "float64",
        "avg_menu_price": "float64",
        "delivery_fee": "float64",
        "base_delivery_time_min": "float64",
    }
    for col, expected_type in type_checks.items():
        actual_type = df_clean[col].dtype
        assert actual_type == expected_type, f"Column {col} has wrong type: {actual_type} != {expected_type}"

def test_city_is_canonical_and_lowercase(df_clean):
    """City values must be lowercase and from CITY_ANCHORS."""
    # Check lowercase
    assert df_clean["city"].str.islower().all(), "All city values must be lowercase"
    
    # Check against known cities
    unknown_cities = set(df_clean["city"]) - set(cr.CITY_ANCHORS.keys())

    # Allow up to 20% unknown (as per pipeline warning threshold)
    unknown_ratio = len(unknown_cities) / len(df_clean)
    assert unknown_ratio < 0.2, f"Too many unknown cities: {unknown_cities}"

def test_coordinates_have_intra_area_variance_warning(df_clean):
    """Log warning if too many areas have single coordinates."""
    coord_counts = df_clean.groupby(["city", "area"]).agg({
        "lat": "nunique",
        "lon": "nunique"
    })
    
    areas_with_single_coord = (coord_counts["lat"] == 1).sum()
    total_areas = len(coord_counts)
    ratio = areas_with_single_coord / total_areas
    
    # Warn but don't fail if data is naturally sparse
    if ratio > 0.4:
        pytest.skip(f"⚠️  WARNING: {ratio:.1%} of areas have only one restaurant. "
                   "This is okay if data is sparse, but verify jitter logic.")

def test_delivery_time_positive(df_clean):
    """Delivery times must be positive."""
    assert (df_clean["base_delivery_time_min"] > 0).all(), "All delivery times must be > 0"


def test_price_in_valid_range(df_clean):
    """Prices should be within configured bounds."""
    assert df_clean["avg_menu_price"].min() >= cr.PRICE_MIN, "Price below minimum threshold"
    assert df_clean["avg_menu_price"].max() <= cr.PRICE_MAX, "Price above maximum threshold"

def test_price_band_no_nulls(df_clean):
    """All restaurants must have a price band."""
    null_bands = df_clean["price_band"].isnull().sum()
    assert null_bands == 0, f"Found {null_bands} restaurants without price_band"


# Slow Test : Marked to skip during development iterations.
@pytest.mark.slow
def test_reproducibility():
    """Running twice with same seed should produce identical output."""
    df1 = cr.clean_restaurants()
    df2 = cr.clean_restaurants()
    
    # Check full DataFrame equality (including delivery_fee)
    pd.testing.assert_frame_equal(
        df1.sort_values("restaurant_id").reset_index(drop=True),
        df2.sort_values("restaurant_id").reset_index(drop=True),
        check_like=True  # Ignore column order differences
    )


@pytest.mark.slow
def test_pipeline_performance():
    """Pipeline should complete in reasonable time (<30s for 10k rows)."""
    import time
    
    start = time.time()
    df = cr.clean_restaurants()
    elapsed = time.time() - start
    
    assert elapsed < 30, f"Pipeline took {elapsed:.1f}s - too slow for production"
    assert len(df) > 0, "Pipeline produced no data"

# Warning Test:
def test_city_distribution_warning(df_clean):
    """Warn if data is heavily skewed toward one city."""
    city_counts = df_clean["city"].value_counts(normalize=True)
    max_city_pct = city_counts.max()
    
    if max_city_pct > 0.6:
        pytest.skip(f"WARNING: City '{city_counts.index[0]}' is {max_city_pct:.1%} of data - may cause bias")
    # If below threshold, test passes silently


def test_price_band_distribution_warning(df_clean):
    """Log warning if any price band is very sparse (<2%)."""
    band_counts = df_clean["price_band"].value_counts(normalize=True)
    
    for band, pct in band_counts.items():
        if pct < 0.02:
            pytest.skip(f"WARNING: Price band '{band}' is only {pct:.1%} of data - model may underfit")
    # All bands have decent representation    


