#!/bin/bash
set -euxo pipefail

mypy --ignore-missing-imports causal_pyro/
isort --check --diff causal_pyro/ tests/
black --check causal_pyro/ tests/
flake8 causal_pyro/ tests/