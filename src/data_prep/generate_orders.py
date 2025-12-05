

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import gc
import warnings
warnings.filterwarnings("ignore")
import json

from src.common.params import load_params
from src.common.utils import to_python_types

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESTAURANTS_PATH = PROJECT_ROOT / "data" / "interim" / "restaurants_clean.csv"
USERS_PATH = PROJECT_ROOT / "data" / "interim" / "users.csv"
ROUTES_PATH = PROJECT_ROOT / "data" / "interim" / "routes_eta.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "interim" / "orders.csv"

# ✅ NEW: Use params for realistic behavior
params = load_params()
RNG = np.random.default_rng(params["seed"])


# ==================== UTILITY FUNCTIONS ====================

def get_peak_hour_weight(hour: int) -> float:
    """Boost orders during lunch/dinner peaks."""
    if hour in params["orders"]["peak_hours"]:
        return 2.5
    return 1.0


def get_weekend_multiplier(weekday: int) -> float:
    """Boost orders on weekends."""
    return params["orders"]["weekend_multiplier"] if weekday >= 5 else 1.0


def sample_order_count() -> int:
    """Skewed: many light users, few heavy."""
    lam = params["orders"]["lam"]
    n = RNG.poisson(lam=lam)
    return int(np.clip(n, 1, params["orders"]["max_per_user"]))


# ==================== RESTAURANT SELECTION ====================

def choose_restaurant(
    user: pd.Series, 
    restaurants_by_city: dict,  # ✅ OPTIMIZED: dict instead of DataFrame
    route_keys: set,
    route_lookup_dict: dict     # ✅ OPTIMIZED: dict for O(1) lookups
) -> pd.Series:
    """Pick restaurant weighted by cuisine, rating, route existence."""
    city = user["city"]
    user_area = user["home_area"]
    fav_cuisines = str(user["fav_cuisines"]).split(",") if user["fav_cuisines"] else []

    # ✅ FAST: O(1) dictionary lookup
    candidates = restaurants_by_city.get(city)
    if candidates is None or candidates.empty:
        return None

    # ✅ FAST: Vectorized route_key creation (no .apply())
    route_prefix = f"{city}_"
    route_suffix = f"_to_{user_area}"
    candidates = candidates.copy()
    candidates["route_key"] = route_prefix + candidates["area"] + route_suffix
    
    # Filter for valid routes
    valid_mask = candidates["route_key"].isin(route_keys)
    candidates = candidates[valid_mask]
    
    if candidates.empty:
        return None

    # Weighting logic (same as before)
    weights = candidates["num_ratings"].clip(lower=10) ** 0.6
    weights *= (candidates["avg_rating"].fillna(3.5) / 5.0) ** 1.2
    
    if fav_cuisines:
        has_match = candidates["cuisines"].apply(
            lambda cs: any(c.strip() in cs for c in fav_cuisines)
        )
        weights *= np.where(has_match, 1.4, 1.0)

    same_area = candidates["area"] == user_area
    weights *= np.where(same_area, 1.1, 1.0)

    probs = (weights / weights.sum()).to_numpy()
    idx = RNG.choice(candidates.index, p=probs)
    return candidates.loc[idx]


# ==================== ORDER VALUE & STATUS ====================

def calculate_basket_value(
    base_price: float, 
    hour: int, 
    num_items: int,
    is_weekend: bool
) -> float:
    """Time-based pricing premium."""
    value = base_price * num_items
    
    # Peak hour premium (dinner)
    if 19 <= hour <= 22:
        value *= 1.15
    
    # Weekend premium
    if is_weekend:
        value *= 1.10
    
    # Random variance ±30%
    variance = RNG.normal(loc=0, scale=0.15)
    value *= (1 + variance)
    
    return float(np.clip(value, 80, 2500))


def assign_order_status(delivery_time: float) -> str:
    """92% completed, 5% delayed, 3% cancelled."""
    p = RNG.random()
    if p < 0.92:
        return "completed"
    elif p < 0.97:
        return "delayed"
    else:
        # Cancellation more likely if ETA > 30 min
        return "cancelled" if delivery_time > 30 else "delayed"


# ==================== CAPACITY MANAGEMENT ====================

class RestaurantCapacityTracker:
    """Enforce max orders/hour per restaurant."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.orders_per_hour: Dict[Tuple[str, pd.Timestamp], int] = {}
    
    def can_accept(self, restaurant_id: str, timestamp: pd.Timestamp) -> bool:
        """Check if restaurant has capacity at this hour."""
        key = (restaurant_id, timestamp.floor("h"))
        current = self.orders_per_hour.get(key, 0)
        return current < self.capacity
    
    def increment(self, restaurant_id: str, timestamp: pd.Timestamp):
        """Record an order for capacity tracking."""
        key = (restaurant_id, timestamp.floor("h"))
        self.orders_per_hour[key] = self.orders_per_hour.get(key, 0) + 1


# ==================== MAIN GENERATION ====================

def generate_orders(
    restaurants_path=RESTAURANTS_PATH,
    users_path=USERS_PATH,
    routes_path=ROUTES_PATH,
    output_path=OUTPUT_PATH,
) -> pd.DataFrame:
    """Generate realistic food delivery orders."""
    
    print("📦 Loading data...")
    restaurants = pd.read_csv(restaurants_path)
    users = pd.read_csv(users_path)
    routes = pd.read_csv(routes_path, parse_dates=["timestamp"])
    
    # ✅ OPTIMIZATION #1: Pre-index restaurants by city
    restaurants_by_city = {
        city: group.copy() for city, group in restaurants.groupby("city")
    }
    print(f"📊 Pre-indexed {len(restaurants_by_city)} cities")
    
    # ✅ OPTIMIZATION #2: Build O(1) route lookup dictionary
    route_lookup = routes.groupby(["route_key", "timestamp"]).first().reset_index()
    route_lookup_dict = {
        (row["route_key"], pd.Timestamp(row["timestamp"])): row 
        for _, row in route_lookup.iterrows()
    }
    print(f"📊 Built route lookup dict with {len(route_lookup_dict)} keys")
    
    # ✅ OPTIMIZATION #3: Pre-compute route_key templates
    for city, group in restaurants_by_city.items():
        group["route_key_prefix"] = f"{city}_" + group["area"] + "_to_"
    
    # Regular setup
    route_keys = set(routes["route_key"].unique())
    all_times = routes["timestamp"].unique()
    time_weights = np.array([
        get_peak_hour_weight(pd.Timestamp(t).hour) * 
        get_weekend_multiplier(pd.Timestamp(t).weekday())
        for t in all_times
    ])
    time_probs = time_weights / time_weights.sum()
    
    capacity_tracker = RestaurantCapacityTracker(
        params["orders"]["restaurant_capacity_per_hour"]
    )
    
    print("🍔 Generating orders...")
    orders = []
    batch_size = params["orders"]["batch_size"]
    order_id = 1
    
    for i, (_, user) in enumerate(users.iterrows()):
        if (i + 1) % 100 == 0:
            print(f"  → Users processed: {i+1}/{len(users)}")
        
        n_orders = sample_order_count()
        
        for _ in range(n_orders):
            # Pick weighted timestamp
            ts = pd.Timestamp(RNG.choice(all_times, p=time_probs))
            
            # Pick restaurant with optimized function
            rest = choose_restaurant(user, restaurants_by_city, route_keys, route_lookup_dict)
            if rest is None:
                continue
            
            # Check restaurant capacity
            if not capacity_tracker.can_accept(rest["restaurant_id"], ts):
                continue
            
            # Lookup route data with O(1) dictionary
            route_key = f"{rest['city']}_{rest['area']}_to_{user['home_area']}"
            route_row = route_lookup_dict.get((route_key, ts))
            
            if route_row is None:
                continue
            
            # Extract context from dictionary (faster than .iloc[0])
            distance_km = float(route_row["distance_km"])
            traffic_level = float(route_row["traffic_level"])
            rain = int(route_row["rain"])
            temp = float(route_row["temperature"])
            base_eta = float(route_row["base_eta_minutes"])
            
            # Calculate order details
            num_items = int(np.clip(RNG.poisson(lam=3), 1, 12))
            basket_value = calculate_basket_value(
                rest["avg_menu_price"], ts.hour, num_items, ts.weekday() >= 5
            )
            
            # Add realistic noise to ETA
            eta_noise = RNG.normal(0, 5)  # ±5 min random
            actual_delivery_time = base_eta + eta_noise
            
            # Assign status
            status = assign_order_status(actual_delivery_time)
            
            orders.append({
                "order_id": f"O{order_id:07d}",
                "user_id": user["user_id"],
                "restaurant_id": rest["restaurant_id"],
                "route_key": route_key,
                "order_created_at": ts,
                "basket_value": round(basket_value, 2),
                "num_items": num_items,
                "distance_km": round(distance_km, 2),
                "rain": rain,
                "temperature": temp,
                "traffic_level": traffic_level,
                "base_eta_minutes": round(base_eta, 1),
                "actual_delivery_time_min": max(15.0, round(actual_delivery_time, 1)),
                "order_status": status,
            })
            
            capacity_tracker.increment(rest["restaurant_id"], ts)
            
            # Batch write to avoid memory blowup
            if len(orders) >= batch_size:
                pd.DataFrame(orders).to_csv(output_path, mode="a", header=not output_path.exists(), index=False)
                orders = []
                gc.collect()
            
            order_id += 1
    
    # Write final batch
    if orders:
        pd.DataFrame(orders).to_csv(output_path, mode="a", header=not output_path.exists(), index=False)
    
    print(f"✅ Saved orders to {output_path}")

    
    
    # Load and return sample
    return pd.read_csv(output_path)


def main():
    """CLI entry point."""
    params = load_params()
    output_path = Path(params["paths"]["interim"]) / "orders.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Delete old file if exists
    if output_path.exists():
        output_path.unlink()
    
    orders = generate_orders(output_path=output_path)

    # ==================== METRICS GENERATION ====================
    # Generate business and data quality metrics
    status_dist = orders["order_status"].value_counts(normalize=True)

    metrics = {
        "total_orders": len(orders),
        "unique_users": orders["user_id"].nunique(),
        "unique_restaurants": orders["restaurant_id"].nunique(),
        "unique_routes": orders["route_key"].nunique(),
        "avg_basket_value": orders["basket_value"].mean(),
        "completion_rate": float(status_dist.get("completed", 0)),
        "delayed_rate": float(status_dist.get("delayed", 0)),
        "cancelled_rate": float(status_dist.get("cancelled", 0)),
        "avg_delivery_time": orders["actual_delivery_time_min"].mean(),
        "avg_base_eta": orders["base_eta_minutes"].mean(),
        "eta_vs_actual_gap": (orders["actual_delivery_time_min"] - orders["base_eta_minutes"]).mean(),
        "missing_values": {
            "basket_value": int(orders["basket_value"].isna().sum()),
            "delivery_time": int(orders["actual_delivery_time_min"].isna().sum()),
            "route_key": int(orders["route_key"].isna().sum()),
        },
        "delivery_time_range": {
            "min": float(orders["actual_delivery_time_min"].min()),
            "max": float(orders["actual_delivery_time_min"].max()),
        },
        "date_range": {
            "start": str(orders["order_created_at"].min()),
            "end": str(orders["order_created_at"].max()),
        }
    }

    # Write metrics to JSON file
    metrics_dir = PROJECT_ROOT / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    with open(metrics_dir / "orders.json", "w") as f:
        json.dump(to_python_types(metrics), f, indent=2)

    print(f"📊 Metrics saved: {metrics_dir / 'orders.json'}")
    # ==================== END METRICS ====================
    
    print(f"[generate_orders] Final shape: {orders.shape}")


if __name__ == "__main__":
    main()