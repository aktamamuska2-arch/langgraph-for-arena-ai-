.PHONY: help install test lint format build mc clean

PY ?= python3
OUT ?= artifacts/mc

help:
	@echo "MARITIME SENTINEL - make targets"
	@echo "  install    install runtime + dev dependencies"
	@echo "  test       run the pytest suite"
	@echo "  lint       ruff check"
	@echo "  format     ruff format"
	@echo "  build      run the AM-SES self-healing build graph end-to-end"
	@echo "  mc         headless Monte Carlo sweep -> $(OUT)"
	@echo "  clean      remove generated artifacts and caches"

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest tests -q

lint:
	$(PY) -m ruff check .

format:
	$(PY) -m ruff format .

build:
	$(PY) scripts/run_workflow.py

mc:
	$(PY) scripts/run_montecarlo.py --out $(OUT)

clean:
	rm -rf artifacts __pycache__ .pytest_cache .ruff_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
