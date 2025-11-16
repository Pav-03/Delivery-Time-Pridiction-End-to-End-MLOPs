# ============================================================================
# 🧪 tests/test_generate_users.py - User Generator Test Suite
# ============================================================================
import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import sys

from src.data_prep import generate_users as gu

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "data_prep"))

import generate_users as gu

# ============================================================================
# 🎯 FIXTURE: Load generated data once
# ============================================================================
@pytest.fixture(scope="module")
def df_users() -> pd.DataFrame:
    """Load generated users (run 'make generate_users' first)."""
    csv_path = Path("data/interim/users.csv")
    if not csv_path.exists():
        pytest.fail(f"Users data not found at {csv_path}. Run 'make generate_users' first.")
    
    df = pd.read_csv(csv_path)
    
    # Parse types
    df["user_id"] = df["user_id"].astype(str)
    df["home_lat"] = pd.to_numeric(df["home_lat"])
    df["home_lon"] = pd.to_numeric(df["home_lon"])
    
    return df

# ============================================================================
# 🧪 UNIT TESTS: Schema & Data Quality
# ============================================================================
def test_schema_columns(df_users):
    """🎯 All required columns exist."""
    required = {
        "user_id", "city", "home_area", "budget_segment",
        "fav_cuisines_list", "fav_cuisines_str", "home_lat", "home_lon"
    }
    missing = required - set(df_users.columns)
    assert not missing, f"Missing columns: {missing}"

def test_user_id_uniqueness(df_users):
    """🎯 User IDs must be unique (primary key)."""
    n_dup = len(df_users) - df_users["user_id"].nunique()
    assert n_dup == 0, f"Found {n_dup} duplicate user IDs"

def test_no_missing_critical_data(df_users):
    """🎯 No nulls in critical fields."""
    critical = ["user_id", "city", "home_lat", "home_lon"]
    missing = df_users[critical].isnull().sum().sum()
    assert missing == 0, f"Found {missing} null values in critical columns"

def test_coordinates_are_realistic(df_users):
    """🎯 Coordinates must be valid geographic values."""
    assert df_users["home_lat"].between(-90, 90).all(), "Invalid latitude"
    assert df_users["home_lon"].between(-180, 180).all(), "Invalid longitude"

def test_coordinates_in_india_bounds(df_users):
    """🎯 Coordinates should be within India."""
    # Use bounds from params if available
    try:
        params = gu._PARAMS
        bounds = params["restaurants"]["bounds"]
        lat_min, lat_max = bounds["lat_min"], bounds["lat_max"]
        lon_min, lon_max = bounds["lon_min"], bounds["lon_max"]
    except KeyError:
        # Fallback to approximate India bounds
        lat_min, lat_max = 8.0, 37.0
        lon_min, lon_max = 68.0, 97.0
    
    assert df_users["home_lat"].between(lat_min, lat_max).all()
    assert df_users["home_lon"].between(lon_min, lon_max).all()

def test_budget_segment_valid(df_users):
    """🎯 Budget segments must be from allowed set."""
    allowed = {"budget", "mid", "premium", "luxury"}
    actual = set(df_users["budget_segment"].unique())
    assert actual.issubset(allowed), f"Invalid budget segments: {actual - allowed}"

def test_fav_cuisines_format(df_users):
    """🎯 Favorite cuisines should be properly formatted."""
    # No null strings
    null_cuisines = df_users["fav_cuisines_str"].isnull().sum()
    assert null_cuisines == 0, f"Found {null_cuisines} null cuisine strings"
    
    # Reasonable length
    max_len = df_users["fav_cuisines_str"].str.len().max()
    assert max_len < 200, f"Cuisine string too long: max {max_len} chars"

def test_city_distribution_balanced(df_users):
    """🎯 City distribution should match restaurant weights."""
    city_counts = df_users["city"].value_counts(normalize=True)
    max_city_pct = city_counts.max()
    
    # No single city > 60% (prevent dominance)
    assert max_city_pct < 0.6, f"City '{city_counts.index[0]}' is {max_city_pct:.1%} of users"

def test_budget_distribution_reasonable(df_users):
    """🎯 Budget segments should have reasonable distribution."""
    budget_counts = df_users["budget_segment"].value_counts(normalize=True)
    
    # Each segment should have at least 5% of users
    for budget, pct in budget_counts.items():
        assert pct > 0.05, f"Budget '{budget}' has only {pct:.1%} of users"

def test_data_volume_reasonable(df_users):
    """🎯 Should have enough users for training."""
    n_users = len(df_users)
    assert n_users > 1000, f"Only {n_users} users - insufficient for training"
    assert n_users < 1000000, f"Too many users ({n_users}) - may cause memory issues"

def test_max_fav_cuisines_respected(df_users):
    """🎯 Users shouldn't exceed max favorite cuisines."""
    max_fav = gu._PARAMS["users"]["max_fav_cuisines"]
    
    # Count cuisines in each list
    cuisine_counts = df_users["fav_cuisines_list"].apply(
        lambda x: len(eval(x)) if isinstance(x, str) else len(x)
    )
    
    assert cuisine_counts.max() <= max_fav, f"Some users have >{max_fav} cuisines"

def test_intra_area_variance(df_users):
    """🎯 Multiple users in same area should have different coordinates."""
    # Sample a popular area
    area_counts = df_users.groupby(["city", "home_area"]).size()
    if len(area_counts) > 0:
        popular_area = area_counts.idxmax()
        city, area = popular_area
        
        users_in_area = df_users[
            (df_users["city"] == city) & 
            (df_users["home_area"] == area)
        ]
        
        # If >1 user in area, coordinates should differ
        if len(users_in_area) > 1:
            unique_coords = users_in_area[["home_lat", "home_lon"]].drop_duplicates()
            assert len(unique_coords) > 1, "Users in same area have identical coordinates"

# ============================================================================
# 🔁 Reproducibility Test
# ============================================================================
@pytest.mark.slow
def test_reproducibility():
    """Running twice with same seed should produce identical coordinates."""
    df1 = gu.generate_users(gu.RESTAURANTS_PATH, 1000, 5, seed=42)
    df2 = gu.generate_users(gu.RESTAURANTS_PATH, 1000, 5, seed=42)
    
    # Coordinates must be identical (deterministic)
    pd.testing.assert_series_equal(df1["home_lat"], df2["home_lat"])
    pd.testing.assert_series_equal(df1["home_lon"], df2["home_lon"])
    
    # User IDs must be identical
    pd.testing.assert_series_equal(df1["user_id"], df2["user_id"])

# ============================================================================
# ⚡ Performance Test
# ============================================================================
@pytest.mark.slow
def test_performance():
    """Should generate 10k users in <2 seconds."""
    import time
    
    start = time.time()
    df = gu.generate_users(gu.RESTAURANTS_PATH, n_users=10000, max_fav_cuisines=5, seed=42)
    elapsed = time.time() - start
    
    assert elapsed < 2.0, f"Generation took {elapsed:.2f}s - too slow"
    assert len(df) == 10000, "Wrong number of users generated"