#!/bin/bash
set -euxo pipefail

poetry run isort causal_pyro/ tests/
poetry run black causal_pyro/ tests/
