.PHONY: setup scan retrain dashboard health discord test lint clean

PYTHON := .venv/bin/python
PYTEST := .venv/bin/pytest

setup:
	python -m venv .venv
	.venv/bin/pip install -e ".[dev]"
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

clean:
	rm -rf __pycache__ src/rally/__pycache__ tests/__pycache__
	rm -rf .pytest_cache src/rally/.pytest_cache
	rm -rf *.egg-info src/*.egg-info
