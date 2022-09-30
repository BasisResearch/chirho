from typing import Callable, Optional, TypeVar, Union

import pyro
import pyro.distributions as dist

from causal_pyro.primitives import intervene, Intervention
from causal_pyro.counterfactual.handlers import BaseCounterfactual, Factual, TwinWorldCounterfactual
from causal_pyro.query.do_messenger import DoMessenger, do


T = TypeVar("T")


def f_W(U2, U3, e_W):
    return U2 + U3 + e_W

def f_X(U1: T, U3: T, U4: T, e_X: T) -> T: 
    return U1 + U3 + U4 + e_X

def f_Z(U4, X, W, e_X):
    return U4 + X + W + e_X

def f_Y(X, Z, U1, U2, e_Y):
    return X + Z + U1 + U2 + e_Y

# X.shape() # (2,) 
# X = intervene(X, (x, x_prime))
# X.shape() # (2, 3)


def model():
    U1 = pyro.sample("U1", dist.Normal(0, 1))
    U2 = pyro.sample("U2", dist.Normal(0, 1))
    U3 = pyro.sample("U3", dist.Normal(0, 1))
    U4 = pyro.sample("U4", dist.Normal(0, 1))

    e_W = pyro.sample("e_W", dist.Normal(0, 1))
    W = pyro.deterministic("W", f_W(U2, U3, e_W), event_dim=0)

    e_X = pyro.sample("e_X", dist.Normal(0, 1))
    X = pyro.deterministic("X", f_X(U1, U3, U4, e_X), event_dim=0) # = [0.1, x, x_prime]

    e_Z = pyro.sample("e_Z", dist.Normal(0, 1))
    Z = pyro.deterministic("Z", f_Z(U4, X, W, e_Z), event_dim=0)  # [Z, Z(X=x')]

    e_Y = pyro.sample("e_Y", dist.Normal(0, 1))
    Y = pyro.deterministic("Y", f_Y(X, Z, U1, U2, e_Y), event_dim=0)   # [Y, Y(X=x), Y(X=x', Z(X=x))]
    return Z, Y


x = 1.0
x_prime = 2.0

conditioned_model = pyro.condition(model, {"W":1., "X":0.1, "Z":2., "Y":1.1})
# conditioned_model = ConditionMessenger({...})(model)
intervened_model = DoMessenger({"X": x})(conditioned_model)
intervened_model_2 = do(conditioned_model, {"X": x})
# intervened_model = do(conditioned_model, {"X": (x, x_prime), "Z": lambda obs: obs})

test = intervened_model()
test2 = TwinWorldCounterfactual(-1)(intervened_model)()

print(test)
print(test2)


print()


# def bayesian_ite():

#     theta = pyro.sample("theta", ...)

#     with pyro.plate("data", N, dim=-1):
#         Y = intervened_model(theta)
#         return Y[..., 1, 2, :] - Y[..., 0, 1, :]


# def bayesian_ate():
#     ites = bayesian_ite()
#     return ites.sum(dim=-1) / ites.shape[-1]


# def bayesian_cate():
#     theta = pyro.sample("theta", ...)

#     with pyro.plate("data", N, dim=-1), condition(...):
#         Y_obs = model(theta)

#     with pyro.condition({"W": w_obs}), do({"X": (x, x_prime), "Z": lambda obs: obs}):
#         Y_pred = model(theta)
#         return Y_pred


# def propositional_intervene(obs: T, prop: Callable[[T], float], name: str) -> T:
#     act = pyro.sample(name, ImproperUniform(...))
#     pyro.factor(name + "_weight", prop(act))
#     return intervene(obs, act)


# def path_specific_intervene(
#     mechanism: Callable[..., T],
#     cf_mechanism: Callable[..., T]
# ) -> Callable[..., T]:

#     def intervened_mechanism(*args):
#         return intervene(mechanism(*args), cf_mechanism(*args))

#     return intervened_mechanism