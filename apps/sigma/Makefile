# SIGMA operational targets. Keep this file dependency-free: everything runs on
# the venv created by scripts/bootstrap.sh.
VENV ?= /home/user/sigma-venv
PY := $(VENV)/bin/python
HOST ?= 0.0.0.0
PORT ?= 8000

.PHONY: all bootstrap test audit prune serve cli clean

all: test audit

bootstrap:
	bash scripts/bootstrap.sh

test:
	$(PY) -m unittest discover -s tests -t . -v

audit:
	$(PY) scripts/audit_size.py

prune:
	$(PY) scripts/prune_venv.py $(VENV)

serve:
	$(PY) -m sigma.cli serve --host $(HOST) --port $(PORT)

cli:
	@echo "contoh: $(PY) -m sigma.cli run --question 'risiko portofolio?' --seed 7"

clean:
	rm -rf var
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
