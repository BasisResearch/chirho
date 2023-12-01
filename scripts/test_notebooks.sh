#!/bin/bash

INCLUDED_NOTEBOOKS="docs/source/tutorial_i.ipynb docs/source/backdoor.ipynb docs/source/dr_learner.ipynb docs/source/mediation.ipynb docs/source/sdid.ipynb docs/source/slc.ipynb"

CI=1 pytest --nbval-lax --dist loadscope -n auto $INCLUDED_NOTEBOOKS
