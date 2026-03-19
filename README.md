<div align="center">
 
# 🍔 Delivery ETA — Synthetic Data Pipeline
 
### Build realistic food delivery data from scratch. The foundation you need before touching any ML.
 
[![Python](https://img.shields.io/badge/Python-85.1%25-3776AB?logo=python&logoColor=white)](https://python.org)
[![DVC](https://img.shields.io/badge/Pipeline-DVC-945DD6?logo=dvc&logoColor=white)](https://dvc.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
 
</div>
 
---
 
## What Is This?
 
This repo is a **synthetic data generation pipeline** for food delivery ETA prediction.
 
No model training. No deployment. No dashboards. Just the data — done right.
 
I've built end-to-end MLOps projects before. The model part is straightforward — plug in XGBoost, tune hyperparameters, deploy an API, done. What actually matters and what most people skip is **getting the data flow right**. Bad data in, bad model out. No amount of MLOps tooling fixes garbage inputs.
 
So this repo focuses entirely on that foundation:
- Cleaning real restaurant listings (Swiggy)
- Generating realistic users, weather, traffic, and orders
- Engineering features that actually make sense for ETA prediction
- Making the whole thing reproducible with DVC
 
If you want to build an ETA prediction platform on top of this — the data is ready. Plug in your model, serve it, monitor it. The hard part (data) is done.
 
---
 
## The Pipeline
 
Five stages. Each one feeds into the next. All tracked by DVC.
 
```
swiggy.csv (raw restaurant data)
     │
     ▼
 clean_restaurants  →  restaurants_clean.csv
     │                  (geo features, delivery fees, price cleanup)
     ▼
 generate_users     →  users.csv
     │                  (50K users, cuisine preferences, city locations)
     ▼
 fetch_weather_traffic → routes_eta.csv
     │                   (7 days × hourly weather & traffic per city)
     ▼
 prepare_orders     →  orders.csv
     │                  (500K+ orders with peak hours, cancellations,
     │                   weekend patterns, restaurant capacity limits)
     ▼
 build_features     →  train.parquet + val.parquet
                        (70/30 split, rolling windows, entity-level stats)
```
 
Every stage outputs metrics to `metrics/` so you can sanity-check the data at each step.
 
---
 
## What Makes the Synthetic Data Realistic?
 
This isn't random noise. The data follows actual delivery patterns:
 
**Orders behave like real orders** — Peak hours (lunch 12-1pm, dinner 6-10pm) get a 2.5× order boost. Weekends see 40% more volume. Restaurants have capacity limits (5 orders/hour). 8% of orders get cancelled. Basket values vary ±30%.
 
**Cities have personality** — Bangalore and Mumbai get high congestion (0.7), Surat and Ahmedabad are calmer (0.4). Weather baselines match reality — Chennai at 30°C, Delhi at 24°C with seasonal swings.
 
**Users aren't just random points** — 50K users with favorite cuisines, geo-locations within city bounds, and ordering patterns that follow a Poisson distribution (avg 8 orders/month, capped at 40).
 
All of this is configurable from one file — `params.yaml`. Change a number, run `dvc repro`, everything regenerates.
 
---
 
## Quick Start
 
```bash
git clone https://github.com/Pav-03/Delivery-Time-Pridiction-End-to-End-MLOPs.git
cd Delivery-Time-Pridiction-End-to-End-MLOPs
 
pip install -r requirements.txt
pip install -e .
 
# Run full pipeline
dvc repro
 
# Or run a single stage
dvc repro clean_restaurants
dvc repro generate_users
dvc repro build_features
```
 
Output lands in:
- `data/interim/` — intermediate CSVs and Parquets
- `data/processed/` — final `train.parquet` and `val.parquet`
- `metrics/` — JSON metrics per stage
 
---
 
## Project Structure
 
```
├── data/
│   ├── raw/                 # Swiggy restaurant listings
│   ├── interim/             # Cleaned restaurants, users, routes, orders
│   └── processed/           # train.parquet + val.parquet (model-ready)
│
├── src/
│   ├── data_prep/           # One script per pipeline stage
│   │   ├── clean_restaurants.py
│   │   ├── generate_users.py
│   │   ├── fetch_weather_traffic.py
│   │   ├── generate_orders.py
│   │   └── feature_eta.py
│   └── common/              # Shared params loader & utilities
│
├── metrics/                 # Stage-level metrics (JSON)
├── notebooks/               # Exploration notebooks
├── tests/                   # Pytest suite
│
├── dvc.yaml                 # Pipeline definition
├── dvc.lock                 # Reproducibility lock
├── params.yaml              # Single source of truth for all parameters
└── setup.py                 # Package install
```
 
---
 
## Key Parameters
 
Everything lives in `params.yaml`. Tweak and regenerate:
 
| Parameter | Value | What It Does |
|-----------|-------|-------------|
| `seed` | 42 | Reproducibility |
| `users.total` | 50,000 | Synthetic user count |
| `days` | 7 | Weather/traffic simulation window |
| `orders.lam` | 8 | Avg orders per user/month |
| `orders.peak_hours` | 12-13, 18-22 | Lunch & dinner rush |
| `orders.cancellation_rate` | 0.08 | 8% cancellations |
| `orders.weekend_multiplier` | 1.4 | Weekend volume boost |
| `features.rolling_window_hours` | 3 | Feature window size |
 
---
 
## What You Can Build On Top of This
 
The data pipeline is the foundation. If someone wants to extend it, here's what naturally comes next:
 
- **Model training** — XGBoost / LightGBM on `train.parquet`, validate on `val.parquet`
- **Experiment tracking** — Plug in MLflow or W&B
- **Serving** — FastAPI endpoint for ETA inference
- **Real data swap** — Replace synthetic generators with real API feeds (Google Maps traffic, actual order logs) — the feature engineering layer stays the same
- **Monitoring** — Evidently for data drift detection
 
The point is: the pipeline structure doesn't change when you move from synthetic to real data. That's why getting it right matters.
 
---
 
## License
 
MIT — see [LICENSE](LICENSE).
 
---
 
<div align="center">
 
*Good ML starts with good data. This repo gets the data right.*
 
</div>
 


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
