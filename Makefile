.PHONY: setup scan retrain dashboard health discord test lint clean sync-models-down sync-models-up

PYTHON := .venv/bin/python
PYTEST := .venv/bin/pytest

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt
	@echo "Done. Activate with: source .venv/bin/activate"

scan:
	PYTHONPATH=src $(PYTHON) scripts/orchestrator.py scan

retrain:
	PYTHONPATH=src $(PYTHON) scripts/orchestrator.py retrain

dashboard:
	PYTHONPATH=src $(PYTHON) scripts/dashboard.py

health:
	PYTHONPATH=src $(PYTHON) scripts/orchestrator.py health

discord:
	PYTHONPATH=src $(PYTHON) scripts/run_discord.py

test:
	PYTHONPATH=src $(PYTEST) -v

lint:
	PYTHONPATH=src .venv/bin/ruff check src/ tests/

sync-models-down:
	@echo "Pulling models from Railway → local models/"
	railway run --service rally-bot -- tar cf - -C /app/models . | tar xf - -C models/

sync-models-up:
	@echo "Pushing local models/ → Railway"
	tar cf - -C models . | railway run --service rally-bot -- tar xf - -C /app/models/

clean:
	rm -rf __pycache__ src/**/__pycache__ tests/__pycache__
	rm -rf .pytest_cache
	rm -rf *.egg-info src/*.egg-info
