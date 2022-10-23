---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Example: Bayesian estimation with structured latent confounders

```{code-cell} ipython3
from typing import Dict, List, Optional, Tuple, Union, TypeVar

import torch
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample, PyroParam
from pyro.contrib.autoname import scope
from pyro.poutine import condition, reparam

import causal_pyro
from causal_pyro.query.do_messenger import do
from causal_pyro.counterfactual.handlers import Factual, MultiWorldCounterfactual, TwinWorldCounterfactual
```

## Background: Hierarchically structured confounding

In other examples, we have demonstrated how probabilistic
programs can be used to model causal relationships between attributes of
individual entities. However, it is often useful to model relationships
between multiple kinds of entities explicitly.

For example, a student's
educational outcome may depend on her own attributes, as well as the
attributes of her school. In this hierarchical setting, where multiple
students belong to the same school, we can often estimate causal effects
even if these potentially confounding school-level attributes are
latent.

Hierarchical structure is a common motif in social science and
econometric applications of causal inference; appearing in
multi-level-models {cite:p}`gelman2006data`, difference-in-difference
designs {cite:p}`shadish2002experimental`, and within-subjects
designs {cite:p}`loftus1994using`, all of which are out of scope for
graph-based identification methods. Nonetheless, even flexible Gaussian
process versions of these kinds of causal designs can be implemented in
a causal probabilistic programming language {cite:p}`witty_2021`. 

+++

## Model: GP-SLC

Moving beyond
simple linear models, recent work has introduced *Gaussian Processes
with Structured Latent Confounders* (GP-SLC) {cite:p}`witty2020`, using
flexible Gaussian process priors for causal inference in hierarchical
settings. The following generative program is a slightly simplified
variant of GP-SLC.

```{code-cell} ipython3
# Not working
# def slc_cbn(theta_X, theta_T, theta_Y, N_objects=3, N_instances=4):
#     with pyro.plate("objects", N_objects, dim=-2) as objects:
#         U = pyro.sample("U", dist.Normal(0, 1))
#         with pyro.plate("instances", N_instances, dim=-1) as instances:
#             X = pyro.sample("X", dist.Normal(*f_X(U, theta_X), theta_X))
#             T = pyro.sample("T", dist.Normal(*f_T(U, X, theta_T)))
#             Y = pyro.sample("Y", dist.Normal(*f_Y(U, X, T, theta_Y)))
#             return Y

# pyro.render_model(slc_cbn, model_args=(...))
```

## Query: individual treatment effects (ITE)
Following the same informal script as
in the previous examples gives an expanded generative program defining a
joint distribution over object-level latent confounders $U$ and observed
instance-level covariates $X$, treatment $T$, and outcomes $Y$, thereby
inducing a distribution on the individual treatment effects for each
instance.

```{code-cell} ipython3
# Not working
# @TwinWorldCounterfactual(dim=-3)
# @pyro.infer.reparam(config=pyro.infer.reparam.AutoReparam())
# def slc_surrogate_scm(N_objects, N_instances):
#     theta_X = pyro.sample("theta_X", ...)
#     theta_T = pyro.sample("theta_T", ...)
#     theta_Y = pyro.sample("theta_Y", ...)
#     Y = slc_cbn(theta_X, theta_T, theta_Y, N_objects=N_objects, N_instances=N_instances)
#     return Y

# pyro.render_model(slc_surrogate_scm, model_args=())
```

This causal model allows estimation of *individual treatment effects*
$ITE^{(o,i)} = f_y(Y^{(o,i)}_{do(T=1)}) - f_y(Y^{(o,i)}_{do(T=0)})$,
e.g. the increase in a particular student's educational outcome with or
without a particular intervention.

```{code-cell} ipython3
# Not working
# def slc_surrogate_ite(x_obs, t_obs, y_obs):
#     with do(slc_surrogate_scm, {"T": 1. - t_obs}), \
#             pyro.condition(data={"X": x_obs, "Y": y_obs, "T": t_obs}):
#         Ys = slc_surrogate_scm(y_obs.shape[0], y_obs.shape[1])
#         ITE = Ys[1, :, :] - Ys[0, :, :]
#         return ITE

# pyro.render_model(slc_surrogate_ite, model_args=(x_obs, t_obs, y_obs))
```

Note that here we are able to estimate the individual
treatment effect because we assumed that exogenous noise is additive.
Here, the hierarchical structure is compactly expressed as a pair of nested `pyro.plate` statements
over objects $o$ and instances $i$.

+++

## References

```{bibliography}
:filter: docname in docnames
```
