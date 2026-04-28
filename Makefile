# Makefile — Nakaseke NCD-AI project automation
# Run: make <target>

PYTHON   := python
APP_PORT := 5000
IMG      := nakaseke-ncd-ai

.PHONY: install train run test docker-build docker-run mlflow clean help

## install   : Install all Python dependencies
install:
	pip install -r requirements.txt

## train     : Run the full ML pipeline (load -> clean -> features -> train -> evaluate)
train:
	$(PYTHON) train.py

## run       : Start the Flask web application locally (http://localhost:5000)
run:
	$(PYTHON) -m flask --app app/app.py run --host=0.0.0.0 --port=$(APP_PORT) --debug

## test      : Run the test suite
test:
	$(PYTHON) -m pytest tests/ -v --tb=short

## docker-build : Build the Docker image
docker-build:
	docker build -t $(IMG):latest .

## docker-run   : Run the Docker container
docker-run:
	docker run -p $(APP_PORT):5000 --rm $(IMG):latest

## docker-compose : Start the full stack (app + MLflow)
compose-up:
	docker compose up --build

compose-down:
	docker compose down

## mlflow    : Open the MLflow tracking UI in your browser (must run 'train' first)
mlflow:
	mlflow ui --backend-store-uri mlruns --host 0.0.0.0 --port 5001

## clean     : Remove generated artefacts
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache training.log

help:
	@grep -E '^## ' Makefile | sed 's/## /  /'
