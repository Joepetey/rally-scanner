.PHONY: setup scan retrain dashboard health discord test lint clean sync-models-down sync-models-up

PYTHON := .venv/bin/python
PYTEST := .venv/bin/pytest

setup:
	python3 -m venv .venv
	.venv/bin/pip install -e ".[dev,discord]"
	@echo "Done. Activate with: source .venv/bin/activate"

scan:
	$(PYTHON) scripts/orchestrator.py scan

retrain:
	$(PYTHON) scripts/orchestrator.py retrain

dashboard:
	$(PYTHON) scripts/dashboard.py

health:
	$(PYTHON) scripts/orchestrator.py health

discord:
	$(PYTHON) scripts/run_discord.py

test:
	$(PYTEST) -v

lint:
	.venv/bin/ruff check src/ tests/

sync-models-down:
	@echo "Pulling models from Railway → local models/"
	railway run --service rally-bot -- tar cf - -C /app/models . | tar xf - -C models/

sync-models-up:
	@echo "Pushing local models/ → Railway"
	tar cf - -C models . | railway run --service rally-bot -- tar xf - -C /app/models/

clean:
	rm -rf __pycache__ src/rally/__pycache__ tests/__pycache__
	rm -rf .pytest_cache src/rally/.pytest_cache
	rm -rf *.egg-info src/*.egg-info
