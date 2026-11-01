.PHONY: install workflow faults serve test lint atlas clean

install:
	pip install -r requirements.txt

workflow:
	python scripts/run_workflow.py

faults:
	python scripts/fault_matrix.py

serve:
	python scripts/serve.py

test:
	python -m pytest tests -q

lint:
	ruff check .
	ruff format --check .

atlas:
	python scripts/render_atlas.py

clean:
	rm -rf artifacts/*.png artifacts/*.json artifacts/fault_matrix
