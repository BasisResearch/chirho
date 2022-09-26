#!/bin/bash
set -euxo pipefail

./scripts/lint.sh
pytest -s --cov=causal_pyro/ --cov=tests --cov-report=term-missing ${@-} --cov-report html
