# Environment Setup
setup-venv:
	python3 -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

setup-conda:
	@echo "Setting up Conda environment..."
	conda create -n legion python=3.11 -y
	@echo "Installing dependencies with Poetry in Conda environment..."
	conda run -n legion POETRY_VIRTUALENVS_CREATE=false poetry install

# Install requirements with pip or poetry
install:
	@if [ "$(POETRY)" = "true" ]; then \
		echo "Using Poetry to install dependencies..."; \
		POETRY_VIRTUALENVS_CREATE=false poetry install; \
	else \
		echo "Using pip to install dependencies..."; \
		pip install -r requirements.txt; \
	fi

# Install pre-commit hooks
pre-commit:
	@if [ "$(POETRY)" = "true" ]; then \
		POETRY_VIRTUALENVS_CREATE=false poetry run pre-commit install; \
	else \
		pre-commit install; \
	fi

# Full Setup: Environment + Dependencies
setup:
	@if [ "$(ENV)" = "conda" ]; then \
		echo "Setting up environment with Conda..."; \
		make setup-conda POETRY=$(POETRY); \
	elif [ "$(ENV)" = "venv" ]; then \
		echo "Setting up environment with venv..."; \
		make setup-venv POETRY=$(POETRY); \
	else \
		echo "Invalid or missing ENV. Use ENV=conda or ENV=venv."; \
		exit 1; \
	fi
	@echo "Installing dependencies..."
	@make install POETRY=$(POETRY)
	@make pre-commit POETRY=$(POETRY)
	@echo "You will still need to activate the environment with 'conda activate legion'."

# Linting
lint:
	@if [ "$(POETRY)" = "true" ]; then \
		poetry run python scripts/lint.py; \
	else \
		python scripts/lint.py; \
	fi

# Testing
test:
	@if [ "$(POETRY)" = "true" ]; then \
		poetry run pytest -v; \
	else \
		pytest -v; \
	fi
