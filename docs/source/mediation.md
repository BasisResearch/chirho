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

# Example: Mediation analysis and direct effects

```{code-cell} ipython3
from typing import Dict, List, Optional, Tuple, Union, TypeVar, Callable

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

## Background: nested counterfactuals

[Direct and indirect effects](https://ftp.cs.ucla.edu/pub/stat_ser/R273-U.pdf)

[Nested counterfactuals](https://proceedings.neurips.cc/paper/2021/hash/36bedb6eb7152f39b16328448942822b-Abstract.html)

+++

## Model: Pearl's mediation example

```{code-cell} ipython3
class MediationSCM(PyroModule):

    @staticmethod
    def f_W(U2, U3, e_W):
        return U2 + U3 + e_W

    @staticmethod
    def f_X(U1, U3, U4, e_X):
        return U1 + U3 + U4 + e_X

    @staticmethod
    def f_Z(U4, X, W, e_X):
        return U4 + X + W + e_X

    @staticmethod
    def f_Y(X, Z, U1, U2, e_Y):
        return X + Z + U1 + U2 + e_Y

    def forward(self):
        U1 = pyro.sample("U1", dist.Normal(0, 1))
        U2 = pyro.sample("U2", dist.Normal(0, 1))
        U3 = pyro.sample("U3", dist.Normal(0, 1))
        U4 = pyro.sample("U4", dist.Normal(0, 1))

        e_W = pyro.sample("e_W", dist.Normal(0, 1))
        W = pyro.deterministic("W", self.f_W(U2, U3, e_W), event_dim=0)

        e_X = pyro.sample("e_X", dist.Normal(0, 1))
        X = pyro.deterministic("X", self.f_X(U1, U3, U4, e_X), event_dim=0)

        e_Z = pyro.sample("e_Z", dist.Normal(0, 1))
        Z = pyro.deterministic("Z", self.f_Z(U4, X, W, e_Z), event_dim=0)

        e_Y = pyro.sample("e_Y", dist.Normal(0, 1))
        Y = pyro.deterministic("Y", self.f_Y(X, Z, U1, U2, e_Y), event_dim=0)
        return Y
```

## Query: natural direct effect (NDE)

+++

The natural direct effect

```{code-cell} ipython3
# natural direct effect: DE{x,x'}(Y) = E[ Y(X=x', Z(X=x)) - E[Y(X=x)] ]
def direct_effect(model: MediationSCM, x, x_prime, w_obs, x_obs, z_obs, y_obs) -> Callable:
    return do(actions={"X": x})(
        do(actions={"X": x_prime})(
            do(actions={"Z": lambda Z: Z})(
                condition(data={"W": w_obs, "X": x_obs, "Z": z_obs, "Y": y_obs})(
                    MultiWorldCounterfactual(-2)(
                        pyro.plate("data", size=y_obs.shape[-1], dim=-1)(
                            model))))))
```

The indirect effect is actually the same query operation under `MultiWorldCounterfactual`:

```{code-cell} ipython3

# indirect effect: IE{x,x'}(Y) = E[ Y(X=x, Z(X=x')) - E[Y(X=x)] ]
def indirect_effect(model: MediationSCM, x, x_prime, w_obs, x_obs, z_obs, y_obs) -> Callable:
    return do(actions={"X": x})(
        do(actions={"X": x_prime})(
            do(actions={"Z": lambda Z: Z})(
                condition(data={"W": w_obs, "X": x_obs, "Z": z_obs, "Y": y_obs})(
                    MultiWorldCounterfactual(-2)(
                        pyro.plate("data", size=y_obs.shape[-1], dim=-1)(
                            model))))))
```
