# tests/test_clean_restaurants.py (add checks)
import pandas as pd

REQUIRED_COLS = {
    "restaurant_id","name","city","city_raw","area",
    "avg_rating","num_ratings","avg_menu_price","price_band",
    "cuisines","cuisines_str",
    "base_delivery_time_min","lat","lon","delivery_fee"
}

def test_city_is_canonical():
    df = pd.read_csv("data/interim/restaurants_clean.csv")
    # city must be lowercase canonical keys
    assert df["city"].str.lower().equals(df["city"]), "city should be lowercase canonical"
