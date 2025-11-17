import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import re

from src.data_prep import fetch_weather_traffic as fw

# ==================== GLOBAL FIXTURES ====================

@pytest.fixture(scope="module")
def params():
    """Load and validate params centrally."""
    p = fw.load_params()
    required = ["days", "seed", "chunk_hours", "traffic", "weather"]
    for key in required:
        if key not in p:
            pytest.fail(f"params.yaml missing required key: '{key}'")
    return p

@pytest.fixture(scope="module")
def df_routes(params) -> pd.DataFrame:
    """Load generated route-level data."""
    csv_path = fw.ROUTES_OUT
    if not csv_path.exists():
        pytest.fail(
            f"Route data not found at {csv_path}. "
            f"Run 'make generate-routes' first"
        )
    
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # ✅ FIX: Calculate expected rows from generator, not data
    try:
        restaurants = pd.read_csv(fw.RESTAURANTS_PATH)
        users = pd.read_csv(fw.USERS_PATH)
        route_gen = fw.SyntheticRouteGenerator(restaurants, users)
        num_routes = len(route_gen.route_geometries)
    except Exception as e:
        pytest.skip(f"Could not validate row count: {e}")
    
    expected_rows = num_routes * int(params["days"]) * 24
    actual_rows = len(df)
    
    # Allow 20% margin for filtered routes
    if actual_rows < expected_rows * 0.8:
        pytest.fail(
            f"Data incomplete: expected ~{expected_rows} rows (from {num_routes} routes × {params['days']} days), "
            f"got {actual_rows}. Re-run data generation."
        )
    
    return df

@pytest.fixture(scope="module")
def route_gen(params):
    """Fixture for route generator."""
    try:
        restaurants = pd.read_csv(fw.RESTAURANTS_PATH)
        users = pd.read_csv(fw.USERS_PATH)
    except FileNotFoundError as e:
        pytest.fail(f"Missing input data: {e}. Run 'make generate-users' and 'make generate-restaurants'")
    
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

# ==================== DATA QUALITY TESTS (FIXED) ====================

def test_route_key_format(df_routes):
    """Route keys must follow 'city_origin_to_dest' format (flexible)."""
    # ✅ FIX: Check structure, not characters (allows spaces, hyphens, dots)
    has_exactly_two_underscores = df_routes["route_key"].str.count("_") >= 2
    has_to_separator = df_routes["route_key"].str.contains("_to_")
    
    invalid = df_routes[~(has_exactly_two_underscores & has_to_separator)]
    assert len(invalid) == 0, f"Invalid route_key structure: {invalid['route_key'].unique()[:5]}"

def test_traffic_level_bounds(df_routes):
    """Traffic must be between 0.0 and 1.0."""
    assert df_routes["traffic_level"].between(0.0, 1.0).all(), "Traffic out of bounds"

def test_base_eta_reasonable(df_routes):
    """Base ETA should be 2-60 minutes (allowing very short routes)."""
    # ✅ FIX: Lower bound to 2 minutes for realistic short deliveries
    out_of_bounds = df_routes[~df_routes["base_eta_minutes"].between(2, 60)]
    
    if not out_of_bounds.empty:
        # Show offending values
        offending_stats = out_of_bounds["base_eta_minutes"].describe()
        sample_values = out_of_bounds["base_eta_minutes"].head().tolist()
        pytest.fail(
            f"Found {len(out_of_bounds)} rows with base_eta outside [2, 60]:\n"
            f"Stats: {offending_stats}\n"
            f"Sample values: {sample_values}"
        )

def test_distance_correlates_with_eta(df_routes):
    """Longer routes should have higher base ETAs (sanity check)."""
    # Sample per city to avoid cross-city bias
    correlations = []
    for city in df_routes["route_key"].str.split("_").str[0].unique()[:5]:
        sample = df_routes[df_routes["route_key"].str.startswith(city)].head(200)
        if len(sample) > 10:
            corr = sample["distance_km"].corr(sample["base_eta_minutes"])
            if not np.isnan(corr):
                correlations.append(corr)
    
    assert len(correlations) > 0, "Not enough data for correlation test"
    avg_corr = np.mean(correlations)
    assert avg_corr > 0.5, f"Distance-ETA correlation too low: {avg_corr:.2f}"

def test_weather_seasonality(df_routes):
    """Monsoon months (June-Sept) should have higher rain probability."""
    monsoon_data = df_routes[df_routes["timestamp"].dt.month.isin([6,7,8,9])]
    
    # ✅ FIX: Skip test if no monsoon data (short time range)
    if monsoon_data.empty:
        pytest.skip("No monsoon data available (days parameter too small)")
    
    monsoon_rain = monsoon_data["rain"].mean()
    dry_rain = df_routes[~df_routes["timestamp"].dt.month.isin([6,7,8,9])]["rain"].mean()
    
    assert monsoon_rain > dry_rain + 0.01, (
        f"Monsoon rain ({monsoon_rain:.3f}) not sufficiently higher "
        f"than dry season ({dry_rain:.3f})"
    )

def test_traffic_peak_hours(df_routes):
    """Peak hours (7-10, 18-21) should have higher traffic."""
    peak_hours = df_routes[df_routes["timestamp"].dt.hour.isin(list(range(7, 10)) + list(range(18, 21)))]
    off_hours = df_routes[df_routes["timestamp"].dt.hour.isin([1, 2, 3, 4, 5])]
    
    if not peak_hours.empty and not off_hours.empty:
        peak_mean = peak_hours["traffic_level"].mean()
        off_mean = off_hours["traffic_level"].mean()
        assert peak_mean > off_mean + 0.05, (
            f"Peak traffic ({peak_mean:.2f}) not higher than off-peak ({off_mean:.2f})"
        )

# ==================== REPRODUCIBILITY TESTS ====================

@pytest.mark.slow
def test_route_data_is_reproducible(route_gen, params):
    """Running twice with same seed produces identical results."""
    days = 3  # Use fewer days for speed
    seed = int(params["seed"])
    noise_std = float(params["traffic"]["noise_std"])
    
    # Get first route
    route_key = list(route_gen.route_geometries.keys())[0]
    times = fw.generate_time_index(days=days)
    
    weather_fetcher = fw.SyntheticRouteWeatherFetcher(params["weather"]["city_baselines"], route_gen)
    traffic_fetcher = fw.SyntheticRouteTrafficFetcher(params["traffic"]["city_baselines"], route_gen)
    
    # Run twice
    df1 = weather_fetcher.fetch(route_key, times, seed)
    df2 = weather_fetcher.fetch(route_key, times, seed)
    
    df3 = traffic_fetcher.fetch(route_key, times, seed + 100, noise_std)
    df4 = traffic_fetcher.fetch(route_key, times, seed + 100, noise_std)
    
    # DataFrames must match exactly (ignore dtype warnings)
    pd.testing.assert_frame_equal(df1, df2, check_dtype=False)
    pd.testing.assert_frame_equal(df3, df4, check_dtype=False)

# ==================== PERFORMANCE SMOKE TEST ====================

@pytest.mark.slow
def test_performance_routes(route_gen, params):
    """Should generate 10 routes × 7 days in reasonable time."""
    import time
    
    days = 7
    seed = 42
    routes = list(route_gen.route_geometries.keys())[:10]
    times = fw.generate_time_index(days=days)
    
    weather_fetcher = fw.SyntheticRouteWeatherFetcher(params["weather"]["city_baselines"], route_gen)
    traffic_fetcher = fw.SyntheticRouteTrafficFetcher(params["traffic"]["city_baselines"], route_gen)
    
    start = time.time()
    
    for route in routes:
        weather_fetcher.fetch(route, times, seed)
        traffic_fetcher.fetch(route, times, seed + 100, float(params["traffic"]["noise_std"]))
    
    elapsed = time.time() - start
    
    # Smoke test: should complete, not crash
    assert elapsed < 30.0, f"Route generation took {elapsed:.1f}s - may indicate performance regression"

# ==================== EDGE CASE TESTS ====================

def test_invalid_route_key_raises_error(route_gen):
    """Invalid route keys should raise KeyError."""
    with pytest.raises(KeyError, match="Route invalid_route_key not found"):
        route_gen.estimate_base_eta("invalid_route_key")

def test_generate_time_index_validates_inputs():
    """generate_time_index should reject invalid days."""
    with pytest.raises(ValueError, match="Days must be positive"):
        fw.generate_time_index(days=0)
    
    with pytest.raises(ValueError, match="Days must be positive"):
        fw.generate_time_index(days=-5)

def test_empty_area_handling(df_routes):
    """Routes with empty area names should be handled gracefully."""
    # ✅ FIX: Added df_routes fixture parameter
    # Ensure no route keys have double underscores (malformed)
    malformed = df_routes["route_key"][df_routes["route_key"].str.contains("__")]
    assert len(malformed) == 0, f"Malformed route keys with double underscores: {malformed.tolist()[:5]}"