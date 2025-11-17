# tests/test_generate_orders.py

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

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

@pytest.fixture
def sample_params():
    """Load test parameters."""
    from src.common.params import load_params
    return load_params()

@pytest.fixture
def temp_output_dir():
    """Create temporary output directory for tests."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def generated_orders_sample(sample_params, temp_output_dir):
    """Generate a small sample of orders for testing."""
    output_path = temp_output_dir / "test_orders.csv"
    
    # Override params for faster testing
    sample_params["users"]["total"] = 100  # Small sample
    sample_params["orders"]["lam"] = 2      # Low order rate
    
    orders = generate_orders(
        restaurants_path=Path("data/interim/restaurants_clean.csv"),
        users_path=Path("data/interim/users.csv"),
        routes_path=Path("data/interim/routes_eta.csv"),
        output_path=output_path,
    )
    return orders

# ==================== SCHEMA VALIDATION TESTS ====================

class TestOrderSchema:
    """Validate the output schema and data types."""
    
    def test_required_columns_exist(self, generated_orders_sample):
        """All required columns must be present."""
        required_cols = {
            "order_id", "user_id", "restaurant_id", "route_key",
            "order_created_at", "basket_value", "num_items",
            "distance_km", "rain", "temperature", "traffic_level",
            "base_eta_minutes", "actual_delivery_time_min", "order_status"
        }
        missing = required_cols - set(generated_orders_sample.columns)
        assert not missing, f"Missing columns: {missing}"
    
    def test_no_duplicate_order_ids(self, generated_orders_sample):
        """Order IDs must be unique."""
        assert generated_orders_sample["order_id"].is_unique, "Duplicate order IDs found!"
    
    def test_no_nulls_in_critical_columns(self, generated_orders_sample):
        """Critical columns must not have nulls."""
        critical_cols = ["order_id", "user_id", "restaurant_id", "order_created_at"]
        null_counts = generated_orders_sample[critical_cols].isnull().sum()
        assert null_counts.sum() == 0, f"Nulls found: {null_counts[null_counts > 0]}"
    
    def test_data_types(self, generated_orders_sample):
        """Verify correct data types."""
        assert generated_orders_sample["order_id"].dtype == object
        assert generated_orders_sample["user_id"].dtype == object
        assert generated_orders_sample["restaurant_id"].dtype == object
        assert pd.api.types.is_datetime64_any_dtype(generated_orders_sample["order_created_at"])
        assert np.issubdtype(generated_orders_sample["basket_value"].dtype, np.number)
        assert np.issubdtype(generated_orders_sample["distance_km"].dtype, np.number)

# ==================== DATA INTEGRITY TESTS ====================

class TestDataIntegrity:
    """Validate data integrity and realistic ranges."""
    
    def test_basket_value_range(self, generated_orders_sample):
        """Basket value must be within realistic bounds."""
        assert generated_orders_sample["basket_value"].between(80, 2500).all(), \
            "Basket value out of realistic range!"
    
    def test_delivery_time_minimum(self, generated_orders_sample):
        """Delivery times must be at least 15 minutes."""
        assert (generated_orders_sample["actual_delivery_time_min"] >= 15).all(), \
            "Delivery time below minimum threshold!"
    
    def test_distance_positive(self, generated_orders_sample):
        """Distance must be positive."""
        assert (generated_orders_sample["distance_km"] > 0).all(), \
            "Distance must be > 0 km!"
    
    def test_weather_bounds(self, generated_orders_sample):
        """Weather values must be realistic."""
        # Rain: 0-10mm (from params)
        assert generated_orders_sample["rain"].between(0, 10).all(), "Rain out of bounds!"
        # Temperature: 15-40°C (from params)
        assert generated_orders_sample["temperature"].between(15, 40).all(), "Temperature out of bounds!"
        # Traffic: 0.5-2.0x (from params)
        assert generated_orders_sample["traffic_level"].between(0.5, 2.0).all(), "Traffic out of bounds!"

# ==================== BUSINESS LOGIC TESTS ====================

class TestBusinessLogic:
    """Validate that business rules are correctly encoded."""
    
    @pytest.mark.slow
    def test_peak_hour_distribution(self, sample_params):
        """Peak hours should have ~2.5x more orders."""
        # Generate a larger sample for statistical significance
        output_path = Path(tempfile.mkdtemp()) / "peak_test.csv"
        orders = generate_orders(
            restaurants_path=Path("data/interim/restaurants_clean.csv"),
            users_path=Path("data/interim/users.csv"),
            routes_path=Path("data/interim/routes_eta.csv"),
            output_path=output_path,
        )
        
        orders["hour"] = pd.to_datetime(orders["order_created_at"]).dt.hour
        total_orders = len(orders)
        
        # Count orders in peak vs non-peak hours
        peak_hours = set(sample_params["orders"]["peak_hours"])
        peak_orders = orders[orders["hour"].isin(peak_hours)]
        peak_ratio = len(peak_orders) / total_orders
        
        # Expect peak ratio > 50% due to 2.5x weighting
        assert peak_ratio > 0.4, f"Peak hour orders too low: {peak_ratio:.2%}"
        shutil.rmtree(output_path.parent)
    
    def test_weekend_multiplier(self, sample_params):
        """Weekends should have higher basket values."""
        output_path = Path(tempfile.mkdtemp()) / "weekend_test.csv"
        orders = generate_orders(
            restaurants_path=Path("data/interim/restaurants_clean.csv"),
            users_path=Path("data/interim/users.csv"),
            routes_path=Path("data/interim/routes_eta.csv"),
            output_path=output_path,
        )
        
        orders["weekday"] = pd.to_datetime(orders["order_created_at"]).dt.weekday
        weekend_multiplier = sample_params["orders"]["weekend_multiplier"]
        
        weekend_basket = orders[orders["weekday"] >= 5]["basket_value"].mean()
        weekday_basket = orders[orders["weekday"] < 5]["basket_value"].mean()
        
        # Weekend baskets should be higher
        assert weekend_basket > weekday_basket * 1.05, \
            "Weekend basket multiplier not working!"
        shutil.rmtree(output_path.parent)
    
    def test_order_status_distribution(self, generated_orders_sample):
        """Order status should follow ~92% completed, 5% delayed, 3% cancelled."""
        status_counts = generated_orders_sample["order_status"].value_counts(normalize=True)
        
        assert abs(status_counts.get("completed", 0) - 0.92) < 0.05, \
            f"Completed rate off: {status_counts.get('completed', 0):.2%}"
        assert abs(status_counts.get("delayed", 0) - 0.05) < 0.03, \
            f"Delayed rate off: {status_counts.get('delayed', 0):.2%}"
        assert status_counts.get("cancelled", 0) < 0.05, \
            f"Cancellation rate too high: {status_counts.get('cancelled', 0):.2%}"

# ==================== DETERMINISM TESTS ====================

@pytest.mark.slow
class TestDeterminism:
    """Ensure reproducibility with same seed."""
    
    def test_deterministic_output(self, sample_params, temp_output_dir):
        """Same seed should produce identical orders."""
        # Run once
        output1 = temp_output_dir / "orders1.csv"
        orders1 = generate_orders(
            restaurants_path=Path("data/interim/restaurants_clean.csv"),
            users_path=Path("data/interim/users.csv"),
            routes_path=Path("data/interim/routes_eta.csv"),
            output_path=output1,
        )
        
        # Run twice (same seed from params)
        output2 = temp_output_dir / "orders2.csv"
        orders2 = generate_orders(
            restaurants_path=Path("data/interim/restaurants_clean.csv"),
            users_path=Path("data/interim/users.csv"),
            routes_path=Path("data/interim/routes_eta.csv"),
            output_path=output2,
        )
        
        # Should be identical (except possibly order_id if timestamp-based)
        pd.testing.assert_frame_equal(
            orders1.sort_values("order_id").reset_index(drop=True),
            orders2.sort_values("order_id").reset_index(drop=True),
            check_like=True,
        )

# ==================== UNIT TESTS ====================

class TestUtilityFunctions:
    """Test individual utility functions."""
    
    def test_get_peak_hour_weight(self):
        """Peak hours should return 2.5, others 1.0."""
        assert get_peak_hour_weight(12) == 2.5  # Lunch peak
        assert get_peak_hour_weight(19) == 2.5  # Dinner peak
        assert get_peak_hour_weight(10) == 1.0  # Non-peak
    
    def test_get_weekend_multiplier(self, sample_params):
        """Weekends should return multiplier, weekdays 1.0."""
        multiplier = sample_params["orders"]["weekend_multiplier"]
        assert get_weekend_multiplier(5) == multiplier  # Saturday
        assert get_weekend_multiplier(6) == multiplier  # Sunday
        assert get_weekend_multiplier(3) == 1.0         # Thursday
    
    def test_sample_order_count_distribution(self):
        """Order count should be Poisson distributed."""
        counts = [sample_order_count() for _ in range(1000)]
        
        # All counts should be positive
        assert all(c >= 1 for c in counts), "Order count can't be zero!"
        
        # Most users should have few orders (Poisson skew)
        median_count = np.median(counts)
        assert median_count < 10, f"Median too high: {median_count}"
    
    def test_calculate_basket_value_bounds(self):
        """Basket value respects min/max bounds."""
        # Test extreme cases
        min_val = calculate_basket_value(80, 10, 1, False)  # Budget restaurant, 1 item
        max_val = calculate_basket_value(2500, 20, 12, True)  # Luxury, 12 items, weekend
        
        assert 80 <= min_val <= 2500, f"Min bounds violated: {min_val}"
        assert 80 <= max_val <= 2500, f"Max bounds violated: {max_val}"
    
    def test_assign_order_status_distribution(self):
        """Status assignment follows probability distribution."""
        statuses = [assign_order_status(25.0) for _ in range(10000)]
        status_counts = pd.Series(statuses).value_counts(normalize=True)
        
        assert status_counts["completed"] > 0.90, "Completed rate too low"
        assert status_counts["completed"] < 0.94, "Completed rate too high"
        assert status_counts["delayed"] > 0.03, "Delayed rate too low"
        assert status_counts.get("cancelled", 0) < 0.05, "Cancel rate too high"

class TestCapacityTracker:
    """Test the RestaurantCapacityTracker class."""
    
    def test_capacity_limit_enforced(self):
        """Tracker should enforce hourly capacity."""
        tracker = RestaurantCapacityTracker(capacity=3)
        ts = pd.Timestamp("2025-01-17 19:00:00")
        
        # Should accept first 3 orders
        assert tracker.can_accept("R001", ts) is True
        tracker.increment("R001", ts)
        assert tracker.can_accept("R001", ts) is True
        tracker.increment("R001", ts)
        assert tracker.can_accept("R001", ts) is True
        tracker.increment("R001", ts)
        
        # Should reject the 4th
        assert tracker.can_accept("R001", ts) is False
    
    def test_capacity_resets_next_hour(self):
        """Capacity should reset after floor-hour boundary."""
        tracker = RestaurantCapacityTracker(capacity=2)
        ts1 = pd.Timestamp("2025-01-17 19:00:00")
        ts2 = pd.Timestamp("2025-01-17 20:00:00")
        
        # Fill up 7 PM slot
        tracker.increment("R001", ts1)
        tracker.increment("R001", ts1)
        assert tracker.can_accept("R001", ts1) is False
        
        # 8 PM should be free
        assert tracker.can_accept("R001", ts2) is True

# ==================== EDGE CASE TESTS ====================

class TestEdgeCases:
    """Test edge cases and failure modes."""
    
    def test_user_with_no_fav_cuisines(self, sample_params, temp_output_dir):
        """Users with empty fav_cuisines should still generate orders."""
        # Modify a user to have empty cuisines
        users = pd.read_csv("data/interim/users.csv")
        users.loc[0, "fav_cuisines"] = ""
        test_users_path = temp_output_dir / "test_users_no_cuisine.csv"
        users.to_csv(test_users_path, index=False)
        
        output_path = temp_output_dir / "orders_empty_cuisine.csv"
        orders = generate_orders(
            restaurants_path=Path("data/interim/restaurants_clean.csv"),
            users_path=test_users_path,
            routes_path=Path("data/interim/routes_eta.csv"),
            output_path=output_path,
        )
        
        # Should still generate orders
        assert len(orders) > 0, "No orders generated for user without cuisines!"
    
    def test_restaurant_with_no_valid_routes(self, sample_params, temp_output_dir):
        """Should gracefully handle users in areas with no routes."""
        # Create a fake user in a non-existent area
        users = pd.read_csv("data/interim/users.csv")
        users.loc[0, "home_area"] = "FAKE_AREA_999"
        test_users_path = temp_output_dir / "test_users_fake_area.csv"
        users.to_csv(test_users_path, index=False)
        
        output_path = temp_output_dir / "orders_fake_route.csv"
        orders = generate_orders(
            restaurants_path=Path("data/interim/restaurants_clean.csv"),
            users_path=test_users_path,
            routes_path=Path("data/interim/routes_eta.csv"),
            output_path=output_path,
        )
        
        # Should complete without crashing, even if no orders for that user
        assert isinstance(orders, pd.DataFrame), "Should return DataFrame even if empty"
    
    def test_batch_write_functionality(self, temp_output_dir):
        """Batch writing should create file without memory bloat."""
        output_path = temp_output_dir / "batch_test.csv"
        
        # Generate small batch
        orders = generate_orders(
            restaurants_path=Path("data/interim/restaurants_clean.csv"),
            users_path=Path("data/interim/users.csv"),
            routes_path=Path("data/interim/routes_eta.csv"),
            output_path=output_path,
        )
        
        # File should exist and be non-empty
        assert output_path.exists(), "Output file not created!"
        assert output_path.stat().st_size > 0, "Output file is empty!"
        
        # Should be able to read it back
        df_back = pd.read_csv(output_path)
        assert len(df_back) == len(orders), "Batch write corrupted data!"

# ==================== PERFORMANCE TESTS ====================

@pytest.mark.slow
class TestPerformance:
    """Test performance characteristics."""
    
    def test_generation_speed(self, sample_params):
        """Should generate orders at reasonable speed."""
        import time
        
        start = time.time()
        
        # Generate small sample
        output_path = Path(tempfile.mkdtemp()) / "perf_test.csv"
        sample_params["users"]["total"] = 50  # Tiny sample
        generate_orders(
            restaurants_path=Path("data/interim/restaurants_clean.csv"),
            users_path=Path("data/interim/users.csv"),
            routes_path=Path("data/interim/routes_eta.csv"),
            output_path=output_path,
        )
        
        elapsed = time.time() - start
        
        # Should be fast for 50 users (under 10 seconds)
        assert elapsed < 10, f"Too slow: {elapsed:.2f}s for 50 users"
        shutil.rmtree(output_path.parent)

# ==================== INTEGRATION TEST ====================

@pytest.mark.slow
def test_full_pipeline_integration(sample_params):
    """Test that entire pipeline works end-to-end."""
    # This assumes data files exist from previous steps
    assert Path("data/interim/restaurants_clean.csv").exists(), "Missing restaurants!"
    assert Path("data/interim/users.csv").exists(), "Missing users!"
    assert Path("data/interim/routes_eta.csv").exists(), "Missing routes!"
    
    # Generate orders
    output_path = Path(tempfile.mkdtemp()) / "integration_test.csv"
    orders = generate_orders(
        restaurants_path=Path("data/interim/restaurants_clean.csv"),
        users_path=Path("data/interim/users.csv"),
        routes_path=Path("data/interim/routes_eta.csv"),
        output_path=output_path,
    )
    
    # Should generate reasonable number of orders
    n_users = pd.read_csv("data/interim/users.csv").shape[0]
    expected_min = n_users * 1  # At least 1 order per user
    expected_max = n_users * sample_params["orders"]["max_per_user"]
    
    assert expected_min <= len(orders) <= expected_max, \
        f"Order count unrealistic: {len(orders)} for {n_users} users"
    
    shutil.rmtree(output_path.parent)