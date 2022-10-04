#!/bin/bash
set -euxo pipefail

isort --profile black causal_pyro/ tests/
black causal_pyro/ tests/
