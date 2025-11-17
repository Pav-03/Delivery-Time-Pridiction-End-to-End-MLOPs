.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = Delivery-Time-Pridiction-End-to-End-MLOPs
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

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



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
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
	@echo "  make test-fast          Run fast tests (development)"
	@echo "  make test               Run all tests (pre-commit)"
	@echo "  make test-coverage      Run tests with coverage report"
	@echo ""
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

# Where our interim/feature data lives; adjust if you changed paths
INTERIM_DIR := data/interim
FEATURE_DIR := data/features

# Run only the restaurant cleaning step
.PHONY: clean_restaurants
clean_restaurants: requirements
	$(PYTHON_INTERPRETER) -m src.data_prep.clean_restaurants

## 👥 Generate synthetic users (NEW TARGET)
.PHONY: generate_users
generate_users: clean_restaurants
	@echo "👥 Generating synthetic users..."
	$(PYTHON_INTERPRETER) -m src.data_prep.generate_users

## 🔄 Run full data pipeline
.PHONY: data
data: clean_restaurants generate_users
	@echo "✅ Full data pipeline complete!"


# ============================================================================
# 🧪 TESTING TARGETS
# ============================================================================

## Run ALL tests (including slow integration tests)
.PHONY: test
test:
	@echo "🧪 Running ALL tests..."
	pytest tests/ -v

## Run only fast tests (skip slow)
.PHONY: test-fast
test-fast:
	@echo "🚀 Running FAST tests (skip slow)..."
	pytest tests/ -v -m "not slow"

## Run tests with coverage report
.PHONY: test-coverage
test-coverage:
	@echo "📊 Running tests with coverage..."
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing
	@echo "📈 Open htmlcov/index.html to view coverage"

## Clean test cache and coverage files
.PHONY: test-clean
test-clean:
	@echo "🧹 Cleaning test artifacts..."
	rm -rf .pytest_cache/ htmlcov/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

## 👥 Run ONLY user generation tests (NEW TARGET)
.PHONY: test-users
test-users:
	@echo "👥 Testing user generator..."
	pytest tests/test_generate_users.py -v

.PHONY: generate-routes
generate-routes:  ## Generate route-level weather & traffic data
	@echo "🌤️ Generating route-level weather & traffic..."
	@python src/data_prep/fetch_weather_traffic.py

.PHONY: test-routes
test-routes:  ## Test route data generation
	@echo "🧪 Testing route data generator..."
	@pytest tests/test_fetch_weather_traffic.py -v

.PHONY: test-all
test-all:  ## Run all tests (restaurants, users, routes)
	@echo "🧪 Running full test suite..."
	@pytest tests/ -v

# Add route dependencies to existing targets
data: restaurants_clean.csv users.csv weather-traffic.csv

weather-traffic.csv: $(USERS_PATH) $(RESTAURANTS_PATH)
	$(MAKE) generate-routes