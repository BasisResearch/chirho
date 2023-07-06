lint: FORCE
	./scripts/clean.sh

format:
	black .
	isort .

tests: lint FORCE
	pytest -v tests

FORCE: