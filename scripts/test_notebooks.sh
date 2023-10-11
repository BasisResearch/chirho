#!/bin/bash

INCLUDED_NOTEBOOKS="docs/source/tutorial_i.ipynb"

CI=1 pytest --nbval-lax --dist loadscope -n auto $INCLUDED_NOTEBOOKS
