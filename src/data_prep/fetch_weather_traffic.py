import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Protocol
import numpy as np
import pandas as pd
import requests
import itertools
import json

from src.common.params import load_params
from src.common.utils import to_python_types

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESTAURANTS_PATH = PROJECT_ROOT / "data" / "interim" / "restaurants_clean.csv"
USERS_PATH = PROJECT_ROOT / "data" / "interim" / "users.csv"
ROUTES_OUT = PROJECT_ROOT / "data" / "interim" / "routes_eta.csv"

logger = logging.getLogger(__name__)

# ==================== UTILITY FUNCTIONS (Add this if missing) =====
def generate_time_index(days: int, end_date: datetime = None) -> pd.DatetimeIndex:
    if days <= 0:
        raise ValueError(f"Days must be positive, got {days}")
    end = end_date or datetime(2024, 3, 31, 23, 0)
    start = end - timedelta(days=days - 1)
    return pd.date_range(start=start, end=end, freq="1h")  # Use lowercase 'h'

# ==================== TYPE CONTRACTS ====================
class RouteFetcher(Protocol):
    def fetch(self, route_id: str, times: pd.DatetimeIndex, seed: int) -> pd.DataFrame:
        ...

# ==================== ROUTE GEOMETRY ENGINE ====================
class SyntheticRouteGenerator:
    """Generate realistic route-level data by modeling road networks."""
    
    def __init__(self, restaurant_data: pd.DataFrame, user_data: pd.DataFrame):
        self.restaurants = restaurant_data
        self.users = user_data
        self.route_geometries = self._build_route_networks()
    
    def _build_route_networks(self) -> Dict[str, Dict]:
        """Generate realistic route networks with capping."""
        route_geoms = {}
        MAX_ROUTES_PER_CITY = 200  # Cap for speed
        
        for city in self.restaurants["city"].unique():
            city_restaurants = self.restaurants[self.restaurants["city"] == city]
            city_users = self.users[self.users["city"] == city]
            
            restaurant_areas = city_restaurants["area"].unique()
            user_areas = city_users["home_area"].unique()
            all_areas = sorted(set(restaurant_areas) | set(user_areas))
            
            area_index = {area: idx for idx, area in enumerate(all_areas)}
            
            # Generate and filter pairs
            pairs = []
            for r_area in restaurant_areas:
                for u_area in user_areas:
                    if r_area == u_area: 
                        continue
                    distance = abs(area_index.get(r_area, 0) - area_index.get(u_area, 0)) * 2
                    if distance <= 15:  # Max 15km
                        pairs.append((r_area, u_area, distance))
            
            # Sort and cap
            pairs.sort(key=lambda x: x[2])
            selected_pairs = pairs[:MAX_ROUTES_PER_CITY]
            
            logger.info("🗺️ City %s: %d routes (from %d possible)", 
                        city, len(selected_pairs), len(pairs))
            
            # Build geometries
            for area1, area2, distance in selected_pairs:
                route_key = f"{city}_{area1}_to_{area2}"
                base_distance = distance + np.random.uniform(1, 3)
                highway_pct = np.random.uniform(0.2, 0.5)
                arterial_pct = np.random.uniform(0.3, 0.6)
                residential_pct = 1.0 - highway_pct - arterial_pct
                
                route_geoms[route_key] = {
                    "distance_km": round(base_distance, 2),
                    "highway_pct": round(highway_pct, 2),
                    "arterial_pct": round(arterial_pct, 2),
                    "residential_pct": round(residential_pct, 2),
                }
        
        logger.info("✅ Total routes: %d", len(route_geoms))
        return route_geoms
    
    def estimate_base_eta(self, route_key: str) -> float:
        """Estimate base travel time without traffic/weather."""
        if route_key not in self.route_geometries:
            raise KeyError(f"Route {route_key} not found")
        
        geom = self.route_geometries[route_key]
        avg_speed = (geom["highway_pct"] * 50 + 
                     geom["arterial_pct"] * 30 + 
                     geom["residential_pct"] * 20)
        return geom["distance_km"] / avg_speed * 60

# ==================== ROUTE-LEVEL WEATHER FETCHER ====================
class SyntheticRouteWeatherFetcher:
    def __init__(self, city_baselines: Dict[str, float], route_generator: SyntheticRouteGenerator):
        self.city_baselines = city_baselines
        self.route_gen = route_generator
    
    def fetch(self, route_key: str, times: pd.DatetimeIndex, seed: int) -> pd.DataFrame:
        
        parts = route_key.split("_")
        city =parts[0]
        origin = parts[1]
        dest = parts[-1]
        geom = self.route_gen.route_geometries[route_key]
        distance_km = geom["distance_km"]
        
        rng = np.random.default_rng(seed)
        base_temp = self.city_baselines.get(city.lower(), 26)
        
        hours = times.hour.values
        months = times.month.values
        
        diurnal_effect = np.where((hours >= 12) & (hours <= 16), 2.0, -1.0)
        noise = rng.normal(0, 1.5, size=len(times))
        temperatures = base_temp + diurnal_effect + noise
        
        rain_probs = np.where(np.isin(months, [6, 7, 8, 9]), 0.15, 0.05)
        
        rain_exposure_factor = distance_km / 5.0
        residential_penalty = geom["residential_pct"] * 0.3
        route_rain_probs = rain_probs * (1 + rain_exposure_factor + residential_penalty)
        route_rain_probs = np.clip(route_rain_probs, 0.0, 0.8)
        rain = rng.binomial(1, route_rain_probs)
        
        return pd.DataFrame({
            "route_key": route_key,
            "timestamp": times,
            "temperature": np.round(temperatures, 1),
            "rain": rain.astype(int),
            "rain_exposure_factor": np.round(rain_exposure_factor, 2)
        })

# ==================== ROUTE-LEVEL TRAFFIC FETCHER ====================
class SyntheticRouteTrafficFetcher:
    def __init__(self, city_baselines: Dict[str, float], route_generator: SyntheticRouteGenerator):
        self.city_baselines = city_baselines
        self.route_gen = route_generator
    
    def fetch(self, route_key: str, times: pd.DatetimeIndex, seed: int, noise_std: float) -> pd.DataFrame:
        
        parts = route_key.split("_")
        city =parts[0]
        origin = parts[1]
        dest = parts[-1]
        geom = self.route_gen.route_geometries[route_key]
        distance_km = geom["distance_km"]
        highway_pct = geom["highway_pct"]
        
        rng = np.random.default_rng(seed)
        base_level = self.city_baselines.get(city.lower(), 0.5)
        
        hours = times.hour.values
        weekdays = times.weekday.values
        
        is_peak = ((hours >= 7) & (hours <= 10)) | ((hours >= 18) & (hours <= 21))
        is_weekend = weekdays >= 5
        
        peak_effect = highway_pct * 0.35
        distance_effect = (distance_km / 10.0) * 0.1
        
        levels = np.full(len(times), base_level)
        levels[is_peak] += peak_effect
        levels[is_weekend] -= 0.1
        levels += distance_effect
        
        noise = rng.normal(0, noise_std, size=len(times))
        levels += noise
        levels = np.clip(levels, 0.05, 1.0)
        
        return pd.DataFrame({
            "route_key": route_key,
            "timestamp": times,
            "traffic_level": levels.astype(float),
            "distance_km": distance_km,
            "highway_pct": highway_pct
        })
    
# ==================== MAIN ORCHESTRATOR ====================
def main():
    """Orchestrate route-level weather & traffic generation."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    
    params = load_params()
    
    # Validate params
    required_keys = ["days", "seed", "traffic", "weather"]
    for key in required_keys:
        if key not in params:
            raise KeyError(f"params.yaml missing required key: '{key}'")
    
    days = int(params["days"])
    seed = int(params["seed"])
    noise_std = float(params.get("traffic", {}).get("noise_std", 0.1))
    chunk_hours = int(params.get("chunk_hours", 24))
    # Load city baselines
    weather_baselines = params.get("weather", {}).get("city_baselines", {})
    traffic_baselines = params.get("traffic", {}).get("city_baselines", {})
    
    if not weather_baselines or not traffic_baselines:
        logger.warning("City baselines missing in params.yaml. Using defaults.")
        weather_baselines = {
            "bangalore": 26, "mumbai": 29, "delhi": 24, "hyderabad": 28,
            "pune": 24, "kolkata": 28, "chennai": 30, "ahmedabad": 27, "surat": 28,
        }
        traffic_baselines = {
            "bangalore": 0.7, "mumbai": 0.7, "delhi": 0.65, "hyderabad": 0.55,
            "pune": 0.5, "kolkata": 0.6, "chennai": 0.55, "ahmedabad": 0.4, "surat": 0.4,
        }
    
    # Validate data files exist
    if not RESTAURANTS_PATH.exists():
        raise FileNotFoundError(f"Restaurant data not found at {RESTAURANTS_PATH}")
    if not USERS_PATH.exists():
        raise FileNotFoundError(f"User data not found at {USERS_PATH}")
    
    restaurants = pd.read_csv(RESTAURANTS_PATH)
    users = pd.read_csv(USERS_PATH)
    
    # Build route networks
    route_gen = SyntheticRouteGenerator(restaurants, users)
    
    # Generate time index
    times = generate_time_index(days=days)
    
    # Initialize fetchers
    weather_fetcher = SyntheticRouteWeatherFetcher(weather_baselines, route_gen)
    traffic_fetcher = SyntheticRouteTrafficFetcher(traffic_baselines, route_gen)
    
    # Get all routes
    all_routes = list(route_gen.route_geometries.keys())
    logger.info("📍 Generating data for %d pre-capped routes", len(all_routes))
    
    # ==================== INCREMENTAL FILE WRITING (Zero Memory Overhead) ====================
    ROUTES_OUT.parent.mkdir(parents=True, exist_ok=True)
    
    # Write headers first
    with open(ROUTES_OUT, "w") as f:
        f.write("route_key,timestamp,temperature,rain,rain_exposure_factor,traffic_level,distance_km,highway_pct,base_eta_minutes\n")
    
    # Process route-by-route with time-axis chunking
    for i, route_key in enumerate(all_routes):
        if i % 50 == 0:  # Progress every 50 routes
            logger.info(f"🔄 Progress: {i}/{len(all_routes)} routes ({i/len(all_routes)*100:.1f}%)")
        
        # Chunk the time index to keep memory usage low
        for start_idx in range(0, len(times), chunk_hours):
            chunk_times = times[start_idx:start_idx + chunk_hours]
            
            # Generate data for this single route chunk
            weather = weather_fetcher.fetch(route_key, chunk_times, seed + i)
            traffic = traffic_fetcher.fetch(route_key, chunk_times, seed + 1000 + i, noise_std)
            
            # Merge (small DataFrame)
            route_data = weather.merge(traffic, on=["route_key", "timestamp"], how="inner")
            route_data["base_eta_minutes"] = route_gen.estimate_base_eta(route_key)
            
            # Validation (per-route-chunk, not global)
            if route_data.isnull().any().any():
                logger.warning(f"Null values in route {route_key} chunk {start_idx}, skipping")
                continue
            if not route_data["traffic_level"].between(0.0, 1.0).all():
                logger.warning(f"Traffic out of bounds in route {route_key} chunk {start_idx}, skipping")
                continue
            
            # Append to disk immediately (never hold full route in memory)
            route_data.to_csv(ROUTES_OUT, mode="a", header=False, index=False)
    logger.info("✅ Route-level ETA data saved: %s", ROUTES_OUT)

    metrics = {
    "total_routes": len(route_gen.route_geometries),
    "total_timestamps": len(times),
    "avg_base_eta": df_sample["base_eta_minutes"].mean(),
    "avg_traffic_level": df_sample["traffic_level"].mean(),
    "rain_probability": df_sample["rain"].mean(),
    }

    metrics_dir = PROJECT_ROOT / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "routes_eta.json", "w") as f:
        json.dump(to_python_types(metrics), f, indent=2)
    logger.info("📊 Metrics saved: %s", metrics_dir / "routes_eta.json")
    
    # Load a sample to show
    df_sample = pd.read_csv(ROUTES_OUT, nrows=5)
    logger.info("\n📊 Sample routes:\n%s", df_sample.to_string())

# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    main()