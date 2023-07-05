lint: FORCE
	scripts/lint.sh

format:
	black .
	isort .

tests: lint FORCE
	pytest -v tests

FORCE: