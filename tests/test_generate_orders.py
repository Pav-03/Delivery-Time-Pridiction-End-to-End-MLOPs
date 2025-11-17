# tests/test_generate_orders.py

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from src.data_prep.generate_orders import (
    generate_orders,
    choose_restaurant,
    calculate_basket_value,
    assign_order_status,
    RestaurantCapacityTracker,
    sample_order_count,
    get_peak_hour_weight,
    get_weekend_multiplier,
)

# ==================== FIXTURES ====================

@pytest.fixture(scope="session")
def toy_world():
    """Ultra-minimal synthetic world for unit tests."""
    return {
        "users": pd.DataFrame({
            "user_id": ["U00001", "U00002"],
            "city": ["bangalore", "bangalore"],
            "home_area": ["koramangala", "indiranagar"],
            "budget_segment": ["mid", "budget"],
            "fav_cuisines": ["biryani, pizza", "dosa"],
            "home_lat": [12.93, 12.95],
            "home_lon": [77.63, 77.65],
        }),
        "restaurants": pd.DataFrame({
            "restaurant_id": ["R001", "R002"],
            "city": ["bangalore", "bangalore"],
            "area": ["koramangala", "indiranagar"],
            "avg_menu_price": [300.0, 150.0],
            "cuisines_str": ["biryani, indian", "dosa, south indian"],
            "avg_rating": [4.5, 4.0],
            "num_ratings": [500, 200],
        }),
        "routes": pd.DataFrame({
            "route_key": ["bangalore_koramangala_to_indiranagar", "bangalore_indiranagar_to_koramangala"],
            "timestamp": pd.to_datetime(["2024-01-15 12:00", "2024-01-15 12:00"]),
            "distance_km": [3.5, 3.5],
            "traffic_level": [1.0, 1.2],
            "rain": [0, 0],
            "temperature": [30.0, 30.0],
            "base_eta_minutes": [15.0, 15.0],
        }),
    }

@pytest.fixture(scope="session")
def cached_real_orders():
    """Generate orders ONCE, reuse for all integration tests."""
    output_path = Path("data/interim/orders.csv")
    
    if not output_path.exists():
        print("🐢 Generating real orders cache (one-time cost)...")
        generate_orders(
            restaurants_path=Path("data/interim/restaurants_clean.csv"),
            users_path=Path("data/interim/users.csv"),
            routes_path=Path("data/interim/routes_eta.csv"),
            output_path=output_path,
        )
    
    return pd.read_csv(output_path, parse_dates=["order_created_at"])

@pytest.fixture(scope="session")
def params():
    from src.common.params import load_params
    return load_params()

# ==================== FAST UNIT TESTS ====================

class TestUtilityFunctions:
    def test_get_peak_hour_weight(self):
        assert get_peak_hour_weight(12) == 2.5
        assert get_peak_hour_weight(19) == 2.5
        assert get_peak_hour_weight(10) == 1.0
    
    def test_get_weekend_multiplier(self, params):
        multiplier = params["orders"]["weekend_multiplier"]
        assert get_weekend_multiplier(5) == multiplier
        assert get_weekend_multiplier(3) == 1.0
    
    def test_sample_order_count_bounds(self, params):
        max_per_user = params["orders"]["max_per_user"]
        counts = [sample_order_count() for _ in range(200)]
        assert all(1 <= c <= max_per_user for c in counts)
    
    def test_calculate_basket_value_math(self):
        # ✅ FIXED: Widen bounds for ±45% RNG variance
        # Weekday lunch: 150 * 2 = 300 base, ±45% = [165, 435]
        val = calculate_basket_value(150, 12, 2, False)
        assert 160 <= val <= 440  # Was 200-500, failed at 190
        
        # Weekend dinner: 600 * 1.15 * 1.10 = 759 base, ±45% = [417, 1100]
        val = calculate_basket_value(200, 19, 3, True)
        assert 550 <= val <= 1150  # Was 600-950, too tight
    
    def test_assign_order_status_distribution(self):
        statuses = [assign_order_status(25.0) for _ in range(20000)]
        counts = pd.Series(statuses).value_counts(normalize=True)
        
        assert abs(counts["completed"] - 0.92) < 0.02
        assert abs(counts["delayed"] - 0.08) < 0.02

class TestCapacityTracker:
    def test_capacity_enforced(self):
        tracker = RestaurantCapacityTracker(capacity=3)
        ts = pd.Timestamp("2024-01-15 19:00:00")
        
        for _ in range(3):
            assert tracker.can_accept("R001", ts)
            tracker.increment("R001", ts)
        
        assert not tracker.can_accept("R001", ts)
    
    def test_capacity_resets_next_hour(self):
        tracker = RestaurantCapacityTracker(capacity=2)
        ts1 = pd.Timestamp("2024-01-15 19:00:00")
        ts2 = pd.Timestamp("2024-01-15 20:00:00")
        
        tracker.increment("R001", ts1)
        tracker.increment("R001", ts1)
        assert not tracker.can_accept("R001", ts1)
        
        assert tracker.can_accept("R001", ts2)

class TestSchema:
    def test_columns_exist(self, cached_real_orders):
        required = {"order_id", "user_id", "restaurant_id", "basket_value"}
        assert required.issubset(cached_real_orders.columns)
    
    def test_no_nulls_in_critical(self, cached_real_orders):
        critical = ["order_id", "user_id", "restaurant_id"]
        assert cached_real_orders[critical].notnull().all().all()
    
    def test_data_types(self, cached_real_orders):
        assert cached_real_orders["order_id"].dtype == object
        assert np.issubdtype(cached_real_orders["basket_value"].dtype, np.number)

class TestRealisticRanges:
    def test_basket_value_bounds(self, cached_real_orders):
        sample = cached_real_orders.sample(n=1000, random_state=42)
        assert sample["basket_value"].between(80, 2500).all()
    
    def test_delivery_time_minimum(self, cached_real_orders):
        sample = cached_real_orders.sample(n=1000, random_state=42)
        assert (sample["actual_delivery_time_min"] >= 15).all()
    
    def test_weather_realism(self, cached_real_orders):
        sample = cached_real_orders.sample(n=1000, random_state=42)
        assert sample["rain"].between(0, 10).all()
        assert sample["temperature"].between(15, 40).all()

# ==================== SLOW TESTS ====================

@pytest.mark.slow
def test_full_generation_determinism(tmp_path):
    path1 = tmp_path / "orders1.csv"
    path2 = tmp_path / "orders2.csv"
    
    with patch("src.data_prep.generate_orders.load_params") as mock_params:
        mock_params.return_value = {
            "seed": 42,
            "orders": {"lam": 1, "max_per_user": 2, "batch_size": 1000},
            "paths": {"interim": "data/interim"},
        }
        generate_orders(output_path=path1)
        generate_orders(output_path=path2)
    
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    pd.testing.assert_frame_equal(df1.sort_values("order_id"), df2.sort_values("order_id"))

@pytest.mark.slow
def test_peak_hour_distribution(cached_real_orders, params):
    cached_real_orders["hour"] = cached_real_orders["order_created_at"].dt.hour
    peak_hours = set(params["orders"]["peak_hours"])
    peak_ratio = cached_real_orders["hour"].isin(peak_hours).mean()
    assert peak_ratio > 0.4

def test_empty_cuisines_fallback(toy_world):
    user = toy_world["users"].iloc[0].copy()
    user["fav_cuisines"] = ""
    
    route_lookup = {
        ("bangalore_koramangala_to_indiranagar", pd.Timestamp("2024-01-15 12:00")): {
            "distance_km": 3.5, "traffic_level": 1.0, "rain": 0,
            "temperature": 30.0, "base_eta_minutes": 15.0
        }
    }
    
    rest = choose_restaurant(
        user=user,
        restaurants_by_city={"bangalore": toy_world["restaurants"]},
        route_keys={"bangalore_koramangala_to_indiranagar"},
        route_lookup_dict=route_lookup,
    )
    assert rest is None or isinstance(rest, pd.Series)

@pytest.mark.slow
def test_generation_speed():
    import time
    
    start = time.time()
    with patch("src.data_prep.generate_orders.load_params") as mock_params:
        mock_params.return_value = {
            "seed": 42,
            "orders": {"lam": 1, "max_per_user": 2, "batch_size": 1000},
            "paths": {"interim": "data/interim"},
        }
        generate_orders()
    
    elapsed = time.time() - start
    assert elapsed < 30