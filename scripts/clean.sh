#!/bin/bash
set -euxo pipefail

SRC="chirho/ tests/"

ruff check --fix $SRC
ruff format $SRC
