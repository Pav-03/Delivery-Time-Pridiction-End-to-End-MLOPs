.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = Delivery-Time-Pridiction-End-to-End-MLOPs
PYTHON_INTERPRETER = python3
PYTHON := $(PYTHON_INTERPRETER)  # ✅ FIX: Define PYTHON based on PYTHON_INTERPRETER
PYTEST := pytest                  # ✅ FIX: Explicitly define pytest

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# PATH DEFINITIONS                                                              #
#################################################################################

INTERIM_DIR := data/interim
FEATURE_DIR := data/features
RESTAURANTS_PATH := $(PROJECT_DIR)/data/interim/restaurants_clean.csv
USERS_PATH := $(PROJECT_DIR)/data/interim/users.csv
ROUTES_OUT := $(PROJECT_DIR)/data/interim/routes_eta.csv
ORDERS_OUT := $(PROJECT_DIR)/data/interim/orders.csv

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Make Dataset
data: requirements
	$(PYTHON_INTERPRETER) src/data/make_dataset.py data/raw data/processed

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src

## Upload Data to S3
sync_data_to_s3:
ifeq (default,$(PROFILE))
	aws s3 sync data/ s3://$(BUCKET)/data/
else
	aws s3 sync data/ s3://$(BUCKET)/data/ --profile $(PROFILE)
endif

## Download Data from S3
sync_data_from_s3:
ifeq (default,$(PROFILE))
	aws s3 sync s3://$(BUCKET)/data/ data/
else
	aws s3 sync s3://$(BUCKET)/data/ data/ --profile $(PROFILE)
endif

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already installed.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

# --- DATA PIPELINE TARGETS ---

.PHONY: clean_restaurants generate_users generate-routes generate-orders clean-orders data

## Clean and process raw restaurant data
clean_restaurants: requirements
	$(PYTHON_INTERPRETER) -m src.data_prep.clean_restaurants

## 👥 Generate synthetic users
generate_users: clean_restaurants
	@echo "👥 Generating synthetic users..."
	$(PYTHON_INTERPRETER) -m src.data_prep.generate_users

## 🌤️ Generate route-level weather & traffic data
generate-routes: $(USERS_PATH) $(RESTAURANTS_PATH)
	@echo "🌤️ Generating route-level weather & traffic..."
	$(PYTHON_INTERPRETER) -m src.data_prep.fetch_weather_traffic

## 🛒 Generate synthetic orders
generate-orders: $(USERS_PATH) $(RESTAURANTS_PATH) $(ROUTES_OUT)
	@echo "🛒 Generating realistic orders..."
	@echo "   Using Python: $(PYTHON)"
	@$(PYTHON) -m src.data_prep.generate_orders

## 🗑️ Clean generated orders
clean-orders:
	@rm -f $(ORDERS_OUT)
	@echo "🗑️ Cleaned orders data"

## ✅ Run full data pipeline
data: clean_restaurants generate_users generate-routes generate-orders
	@echo "✅ Full data pipeline complete!"

#################################################################################
# TEST TARGETS                                                                  #
#################################################################################

.PHONY: test test-fast test-coverage test-clean test-users test-routes test-orders test-all

## Run ALL tests
test:
	@echo "🧪 Running ALL tests..."
	$(PYTEST) tests/ -v

## Run only fast tests (skip slow)
test-fast:
	@echo "🚀 Running FAST tests (skip slow)..."
	$(PYTEST) tests/ -v -m "not slow"

## Run tests with coverage report
test-coverage:
	@echo "📊 Running tests with coverage..."
	$(PYTEST) tests/ --cov=src --cov-report=html --cov-report=term-missing
	@echo "📈 Open htmlcov/index.html to view coverage"

## Clean test artifacts
test-clean:
	@echo "🧹 Cleaning test artifacts..."
	rm -rf .pytest_cache/ htmlcov/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

## Test user generator
test-users:
	@echo "👥 Testing user generator..."
	$(PYTEST) tests/test_generate_users.py -v

## Test route generator
test-routes:
	@echo "🧪 Testing route data generator..."
	$(PYTEST) tests/test_fetch_weather_traffic.py -v

## Test order generator
test-orders:
	@echo "🧪 Testing orders..."
	$(PYTEST) tests/test_generate_orders.py -v -m "not slow"

## Run full test suite (alias)
test-all: test

#################################################################################
# SELF-DOCUMENTING HELP                                                         #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html >
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')

	@echo "Available commands:"
	@echo "  make clean_restaurants   Clean and process raw restaurant data"
	@echo "  make generate_users      Generate synthetic users"
	@echo "  make generate-routes     Generate route-level weather & traffic"
	@echo "  make generate-orders     Generate synthetic orders"
	@echo "  make clean-orders        Clean generated orders"
	@echo "  make data                Run full data pipeline"
	@echo "  make test                Run all tests"
	@echo "  make test-fast           Run fast tests only"
	@echo "  make test-orders         Test order generation"
	@echo "  make requirements        Install Python dependencies"
	@echo "  make clean               Clean Python cache files"
	@echo "  make lint                Lint code with flake8"
	@echo "  make help                Show this help message"