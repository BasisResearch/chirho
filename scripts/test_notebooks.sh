#!/bin/bash

INCLUDED_NOTEBOOKS="docs/source/tutorial_i.ipynb docs/source/backdoor.ipynb docs/source/cevae.ipynb docs/source/dr_learner.ipynb"

CI=1 pytest --nbval-lax --dist loadscope -n auto $INCLUDED_NOTEBOOKS
