#!/bin/bash
set -euxo pipefail

mypy --ignore-missing-imports chirho/
isort --check --profile black --diff chirho/ tests/
black --check chirho/ tests/
flake8 chirho/ tests/