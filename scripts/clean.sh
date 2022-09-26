#!/bin/bash
set -euxo pipefail

isort causal_pyro/ tests/
black causal_pyro/ tests/
