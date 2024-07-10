lint: FORCE
	./scripts/lint.sh

lint-notebooks:
	./scripts/lint_notebooks.sh

format:
	./scripts/clean.sh

format-notebooks:
	./scripts/clean_notebooks.sh

tests: lint FORCE
	pytest -v tests

FORCE: