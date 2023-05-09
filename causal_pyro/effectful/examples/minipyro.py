from typing import Any, Callable, Container, ContextManager, Generic, List, NamedTuple, Optional, Type, TypeVar, Union

import collections
import contextlib
import torch

from torch.distributions import Distribution
from torch.distributions.constraints import Constraint

from ..ops.terms import Environment, Operation, Term, define
from ..ops.forms import Return
from ..ops.interpretations import Interpretation, cont
from ..ops.interpretations import reflect


S, T = TypeVar("S"), TypeVar("T")



@define(Operation)
def sample(
    name: str,
    distribution: Distribution,
    obs: Optional[T] = None
) -> T:
    ...


@define(Operation)
def param(
    name: str,
    init_value: Optional[T] = None,
    constraint=torch.distributions.constraints.real,
    event_dim: Optional[int] = None,
) -> T:
    ...


ParamStore = collections.OrderedDict[str, tuple[torch.Tensor, Constraint, Optional[int]]]


class DefaultModel(Interpretation[ParamStore]):
    state: ParamStore


@DefaultModel.define(sample)
def default_sample(
    param_store: ParamStore,
    ctx: Environment[T],
    result: Optional[T],
    name: str,
    distribution: Distribution,
    obs: Optional[T] = None
) -> T:
    return distribution.sample()


@DefaultModel.define(param)
def default_param(
    param_store: ParamStore,
    ctx: Environment[T],
    result: Optional[T],
    name: str,
    init_value: Optional[T] = None,
    constraint=torch.distributions.constraints.real,
    event_dim: Optional[int] = None,
) -> T:
    if name in param_store:
        return param_store[name][0]
    if init_value is None:
        raise ValueError(f"Missing init_value for parameter {name}")
    param_store[name] = (init_value, constraint, event_dim)
    return init_value


class Plate(NamedTuple):
    name: str
    size: int
    dim: Optional[int]


@define(Operation)
def enter_plate(p: ContextManager[Plate]) -> Plate:
    ...


@define(Operation)
def exit_plate(p: Plate):
    ...


@define(Operation)
@contextlib.contextmanager
def plate(name: str, size: int, dim: Optional[int] = None) -> ContextManager[Plate]:
    p = Plate(name, size, dim)
    try:
        yield enter_plate(p)
    finally:
        exit_plate(p)


class TraceNode(NamedTuple):
    name: str
    value: torch.Tensor


class SampleTraceNode(TraceNode):
    name: str
    distribution: Distribution
    value: torch.Tensor
    is_observed: bool


class ParamTraceNode(TraceNode):
    name: str
    constraint: Constraint
    value: torch.Tensor
    event_dim: Optional[int]


Trace = collections.OrderedDict[str, TraceNode]


class trace(Interpretation[Trace]):
    state: Trace


@trace.define(sample)
def trace_sample(
    state: Trace,
    ctx: Environment[T],
    result: Optional[T],
    name: str,
    distribution: Distribution,
    obs: Optional[T] = None
) -> T:
    result = cont(ctx, result)
    trace[name] = SampleTraceNode(name, result, distribution, obs is not None)
    return result


@trace.define(param)
def trace_param(
    tr: Trace,
    ctx: Environment[T],
    result: Optional[T],
    name: str,
    init_value: Optional[T] = None,
    constraint=torch.distributions.constraints.real,
    event_dim: Optional[int] = None,
) -> T:
    ctx, result = cont(ctx, result)
    tr[name] = TraceNode(name, result, constraint, event_dim)
    return result


class replay(Interpretation[Trace]):
    state: Trace


@replay.define(sample)
def replay_sample(
    tr: Trace,
    ctx: Environment[T],
    result: Optional[T],
    name: str,
    distribution: Distribution,
    obs: Optional[T] = None
) -> T:
    if result is None and name in tr:
        result = tr[name].value
    return cont(ctx, result)


Observations = collections.OrderedDict[str, torch.Tensor]


class condition(Interpretation[Observations]):
    state: Observations


@condition.define(sample)
def condition_sample(
    state: Observations,
    ctx: Environment[T],
    result: Optional[T],
    name: str,
    distribution: Distribution,
    obs: Optional[T] = None
) -> T:
    try:
        result = state[name]
    except KeyError:
        pass
    return cont(ctx, result)


class block(Interpretation[Container[str]]):
    state: Container[str]


@block.define(sample)
def block_sample(
    blocked: Container[str],
    ctx: Environment[T],
    result: Optional[T],
    name: str,
    distribution: Distribution,
    obs: Optional[T] = None
) -> T:
    if name in blocked:
        return reflect(ctx, result)
    return cont(ctx, result)


def elbo(pyro_model: Callable[..., T], guide: Callable[..., T], *args: Any, **kwargs: Any) -> T:

    runtime = DefaultModel()

    with handle(trace(Trace()), runtime=runtime) as guide_trace:
        guide(*args, **kwargs)

    with handle(replay(guide_trace), runtime=runtime), \
            handle(trace(Trace()), runtime=runtime) as model_trace:
        pyro_model(*args, **kwargs)

    elbo = 0.0
    for name, node in model_trace.items():
        if isinstance(node, SampleTraceNode):
            elbo = elbo + node.distribution.log_prob(node.value).sum()

    for name, node in guide_trace.items():
        if isinstance(node, SampleTraceNode):
            elbo = elbo - node.distribution.log_prob(node.value).sum()

    return -elbo
