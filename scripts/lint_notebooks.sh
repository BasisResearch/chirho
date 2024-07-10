#!/bin/bash

INCLUDED_NOTEBOOKS="docs/source/tutorial_i.ipynb" # docs/source/backdoor.ipynb docs/source/dr_learner.ipynb docs/source/mediation.ipynb docs/source/sdid.ipynb docs/source/slc.ipynb docs/source/dynamical_intro.ipynb docs/source/actual_causality.ipynb"

export ISORT_PROFILE=black

nbqa mypy --ignore-missing-imports $INCLUDED_NOTEBOOKS
nbqa isort --check --diff $INCLUDED_NOTEBOOKS
nbqa black --check $INCLUDED_NOTEBOOKS
nbqa flake8 $INCLUDED_NOTEBOOKS