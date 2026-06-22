#!/bin/bash
set -euxo pipefail

ruff check chirho/ tests/
ruff format --diff chirho/ tests/
mypy --ignore-missing-imports chirho/
