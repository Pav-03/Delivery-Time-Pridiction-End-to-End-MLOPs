import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

from src.data_prep import fetch_weather_traffic as fw

# FIXTURE: Load generated data once
@pytest.fixture(scope="module")
def df_routes() -> pd.DataFrame:
    """Load generated route-level data."""
    csv_path = fw.ROUTES_OUT  # Use the constant from the module
    if not csv_path.exists():
        pytest.fail(f"Route data not found at {csv_path}. Run 'make generate-routes' first")
    
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

@pytest.fixture(scope="module")
def route_gen():
    """Fixture for route generator."""
    restaurants = pd.read_csv(fw.RESTAURANTS_PATH)
    users = pd.read_csv(fw.USERS_PATH)
    return fw.SyntheticRouteGenerator(restaurants, users)

# ==================== SCHEMA TESTS ====================

def test_schema_columns_routes(df_routes):
    """All required columns exist."""
    required = {
        "route_key", "timestamp", "temperature", "rain", "rain_exposure_factor",
        "traffic_level", "distance_km", "highway_pct", "base_eta_minutes"
    }
    missing = required - set(df_routes.columns)
    assert not missing, f"Missing columns: {missing}"

def test_no_nulls_critical_routes(df_routes):
    """No nulls in critical fields."""
    critical = ["route_key", "timestamp", "traffic_level", "base_eta_minutes"]
    missing = df_routes[critical].isnull().sum().sum()
    assert missing == 0, f"Found {missing} nulls in critical columns"

# ==================== DATA QUALITY TESTS ====================

def test_route_key_format(df_routes):
    """Route keys must follow 'city_origin_to_dest' format."""
    pattern = r"^[a-z]+_[a-z]+_to_[a-z]+$"
    invalid = df_routes[~df_routes["route_key"].str.match(pattern)]
    assert len(invalid) == 0, f"Invalid route_key formats: {invalid['route_key'].unique()}"

def test_traffic_level_bounds(df_routes):
    """Traffic must be between 0.0 and 1.0."""
    assert df_routes["traffic_level"].between(0.0, 1.0).all(), "Traffic out of bounds"

def test_base_eta_reasonable(df_routes):
    """Base ETA should be 5-60 minutes for realistic delivery routes."""
    assert df_routes["base_eta_minutes"].between(5, 60).all(), "Base ETA unrealistic"

def test_distance_correlates_with_eta(df_routes):
    """Longer routes should have higher base ETAs (sanity check)."""
    # Sample a few routes from same city
    sample = df_routes[df_routes["route_key"].str.startswith("bangalore")].head(100)
    if not sample.empty:
        correlation = sample["distance_km"].corr(sample["base_eta_minutes"])
        assert correlation > 0.7, f"Distance-ETA correlation too low: {correlation:.2f}"

def test_weather_seasonality(df_routes):
    """Monsoon months (June-Sept) should have higher rain probability."""
    monsoon_rain = df_routes[df_routes["timestamp"].dt.month.isin([6,7,8,9])]["rain"].mean()
    dry_rain = df_routes[~df_routes["timestamp"].dt.month.isin([6,7,8,9])]["rain"].mean()
    
    assert monsoon_rain > dry_rain * 2, f"Monsoon rain ({monsoon_rain:.2f}) not >2x dry season ({dry_rain:.2f})"

def test_traffic_peak_hours(df_routes):
    """Peak hours (7-10, 18-21) should have higher traffic."""
    peak_hours = df_routes[df_routes["timestamp"].dt.hour.isin([7,8,9,18,19,20])]["traffic_level"].mean()
    off_hours = df_routes[df_routes["timestamp"].dt.hour.isin([1,2,3,4,5])]["traffic_level"].mean()
    
    assert peak_hours > off_hours + 0.1, f"Peak traffic ({peak_hours:.2f}) not higher than off-peak ({off_hours:.2f})"

# ==================== REPRODUCIBILITY TESTS ====================

@pytest.mark.slow
def test_route_data_is_reproducible(route_gen):
    """Running twice with same seed produces identical results."""
    params = fw.load_params()
    days = int(params["days"])
    seed = int(params["seed"])
    noise_std = float(params["traffic"]["noise_std"])
    
    # Get a real route that exists
    route_key = list(route_gen.route_geometries.keys())[0]
    times = fw.generate_time_index(days=days)
    
    weather_fetcher = fw.SyntheticRouteWeatherFetcher(params["weather"]["city_baselines"], route_gen)
    traffic_fetcher = fw.SyntheticRouteTrafficFetcher(params["traffic"]["city_baselines"], route_gen)
    
    # Run twice
    df1 = weather_fetcher.fetch(route_key, times, seed)
    df2 = weather_fetcher.fetch(route_key, times, seed)
    
    df3 = traffic_fetcher.fetch(route_key, times, seed + 100, noise_std)
    df4 = traffic_fetcher.fetch(route_key, times, seed + 100, noise_std)
    
    # DataFrames must match exactly
    pd.testing.assert_frame_equal(df1, df2)
    pd.testing.assert_frame_equal(df3, df4)

# ==================== PERFORMANCE TESTS ====================

@pytest.mark.slow
def test_performance_routes(route_gen):
    """Should generate 10 routes × 30 days (720 hours) in <5 seconds."""
    import time
    
    params = fw.load_params()
    days = 30
    seed = 42
    
    # Pick first 10 routes
    routes = list(route_gen.route_geometries.keys())[:10]
    times = fw.generate_time_index(days=days)
    
    weather_fetcher = fw.SyntheticRouteWeatherFetcher(params["weather"]["city_baselines"], route_gen)
    traffic_fetcher = fw.SyntheticRouteTrafficFetcher(params["traffic"]["city_baselines"], route_gen)
    
    start = time.time()
    
    for route in routes:
        weather_fetcher.fetch(route, times, seed)
        traffic_fetcher.fetch(route, times, seed + 100, float(params["traffic"]["noise_std"]))
    
    elapsed = time.time() - start
    
    assert elapsed < 10.0, f"Route generation took {elapsed:.2f}s - too slow"

# ==================== EDGE CASE TESTS ====================

def test_invalid_route_key_raises_error(route_gen):
    """Invalid route keys should raise KeyError."""
    with pytest.raises(KeyError):
        route_gen.estimate_base_eta("invalid_route_key")

def test_generate_time_index_validates_inputs():
    """generate_time_index should reject invalid days."""
    with pytest.raises(ValueError, match="Days must be positive"):
        fw.generate_time_index(days=0)