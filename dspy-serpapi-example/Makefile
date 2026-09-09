.PHONY: install mlflow run agent clear-traces help

MLFLOW_TRACKING_URI ?= http://127.0.0.1:5000
MLFLOW_EXPERIMENT ?= dspy-serpapi-demo

install:
	uv sync

mlflow:
	uv run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts

run:
	MLFLOW_DISABLE_AGENT_HINT=1 MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) MLFLOW_EXPERIMENT=$(MLFLOW_EXPERIMENT) uv run python main.py

agent:
	MLFLOW_DISABLE_AGENT_HINT=1 MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) MLFLOW_EXPERIMENT=$(MLFLOW_EXPERIMENT) uv run python main.py --agent

clear-traces:
	@MLFLOW_DISABLE_AGENT_HINT=1 MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) uv run python -c 'import time, mlflow; from mlflow import MlflowClient; e=mlflow.get_experiment_by_name("$(MLFLOW_EXPERIMENT)"); n=MlflowClient().delete_traces(e.experiment_id, max_timestamp_millis=int(time.time()*1000)) if e else 0; print(f"Deleted {n} traces from $(MLFLOW_EXPERIMENT)")'

help:
	@echo "make install  Install the sample dependencies"
	@echo "make mlflow   Start the MLflow UI/tracking server on http://127.0.0.1:5000"
	@echo "make run      Run the direct traced SerpApi search"
	@echo "make agent    Run the traced Gemini + DSPy ReAct demo"
	@echo "make clear-traces  Delete all traces in the configured MLflow experiment"
