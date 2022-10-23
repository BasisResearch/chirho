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

# Example: the causal effect variational autoencoder

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

## Background: Proxy variables and latent confounders

The backdoor adjustment example assumed that it was always possible to measure all
potential confounders $X$, but when this is not the case, additional
assumptions are necessary to perform causal inference. This example,
derived from {cite:p}`louizos2017causal`, considers a setting where parametric
assumptions are necessary for a causal model to be fully identifiable
from observed data.

Suppose we observe a population of individuals with features $X_i$
undergo treatment $t_i \in \{0, 1\}$ with outcome $y_i$. The treatment
variable might represent a medication or an educational strategy, for
example, for populations of patients or students, respectively.

The task
is to estimate the *conditional average treatment effect*: for a new
individual with features $X_*$, what difference in outcome $y_*$ should
we expect if we assign treatment $t_* = 1$ vs. $t_* = 0$? One cannot
simply estimate the conditional probabilities
$p(y_* \mid X = X_*, t = 0)$ and $p(y_* \mid X = X_*, t = 1)$, because
there may be hidden confounders: latent factors $z$ that induce
non-causal correlations between $t$ and $y$ even controlling for the
observed covariates $X$. 

For example, a student's socio-economic status
might influence both their outcome $y$ and the educational strategy $t$
they are exposed to, and the observed covariates $X$ may not fully
characterize the student's SES. As a result, conditioning on $t$ may
alter the distribution over SES, changing the reported outcome.


+++

## Model: neural surrogate causal Bayesian network

Our model captures the intuition that our three observed variables, $X$,
$t$, and $y$, may be correlated, thanks to unobserved confounders $z$.
Here, $f$, $g$, and $h$ are neural networks parameterized by different
parts of the parameter set $\theta$. The parameters of our model can be fit
using standard techniques in Pyro (e.g., stochastic variational
inference).

```{code-cell} ipython3
class ProxyConfounderModel(PyroModule):
    def __init__(self, f_X, f_T, f_Y):
        super().__init__()
        self.f_X = f_X
        self.f_T = f_T
        self.f_Y = f_Y

    def forward(self):
        Z = pyro.sample("Z", dist.Normal(0, 1).expand([10]).to_event(1))
        X = pyro.sample("X", dist.Normal(*self.f_X(Z)))
        T = pyro.sample("T", dist.Bernoulli(logits=self.f_T(Z)))
        Y = pyro.sample("Y", dist.Normal(*self.f_Y(T, Z)))
        return Y
```

## Query: conditional average treatment effect (CATE)

We can now set up a larger model in which the *conditional average
treatment effect* (CATE) we want to estimate is a random variable.

```{code-cell} ipython3
class CEVAE_CATE(PyroModule):
    def __init__(self, individual_model: ProxyConfounderModel):
        super().__init__()
        self.individual_model = individual_model

    def forward(self, x_obs, t_obs, y_obs, x_pred):

        with condition(data={"X": x_obs, "T": t_obs, "Y": y_obs}), \
                pyro.plate("observations", size=x_obs.shape[0], dim=-1):

            Y_obs = self.individual_model()

        with scope(prefix="intervened"), \
                do(actions={"T": 1. - t_obs}), \
                condition(data={"X": x_pred}):
            Ys_pred = self.individual_model()
            return Ys_pred
```

The CATE is the expected return value of this new model, conditioning on
the observed covariates $X$. Any inference method available in Pyro
could be used to estimate it, including amortized variational inference
{cite:p}`kingma2013auto` as in the original paper {cite:p}`louizos2017causal`.

+++

# References

```{bibliography}
:filter: docname in docnames
```

```{code-cell} ipython3

```
