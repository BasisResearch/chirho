#!/bin/bash
set -euxo pipefail

isort --profile black chirho/ tests/
black chirho/ tests/
