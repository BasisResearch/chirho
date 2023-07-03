lint: FORCE
	flake8
	black --check .
	isort --check .
	python scripts/update_headers.py --check

format:
	black .
	isort .

tests: lint FORCE
	pytest -v tests

FORCE: