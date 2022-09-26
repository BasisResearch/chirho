#!/bin/bash
set -euxo pipefail

poetry run cruft check
poetry run mypy --ignore-missing-imports causal_pyro/
poetry run isort --check --diff causal_pyro/ tests/
poetry run black --check causal_pyro/ tests/
poetry run flake8 causal_pyro/ tests/
poetry run safety check -i 39462 -i 40291
poetry run bandit -r causal_pyro/
