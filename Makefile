lint: FORCE
	./scripts/lint.sh

format:
	./scripts/clean.sh

tests: lint FORCE
	pytest -v tests

FORCE: