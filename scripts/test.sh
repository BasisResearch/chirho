#!/bin/bash
set -euxo pipefail

./scripts/lint.sh
pytest -s --cov=chirho/ --cov=tests --cov-report=term-missing ${@-} --cov-report html
