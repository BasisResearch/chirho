from typing import Any, Callable, Container, ContextManager, Generic, Hashable, List, NamedTuple, Optional, Type, TypeVar, Union

import collections
import contextlib
import torch

from torch.distributions import Distribution
from torch.distributions.constraints import Constraint
from causal_pyro.effectful.ops.interpretation import StatefulInterpretation, register

from causal_pyro.effectful.ops.operation import Operation, define
from causal_pyro.effectful.ops.handler import fwd, handler, reflect, product, prompt_calls


S, T = TypeVar("S"), TypeVar("T")

Environment = dict[Hashable, T]  # TODO define and import from terms


@define(Operation)
def sample(
    name: str,
    distribution: Distribution,
    obs: Optional[T] = None
) -> T:
    raise NotImplementedError


@define(Operation)
def param(
    name: str,
    init_value: Optional[T] = None,
    constraint=torch.distributions.constraints.real,
    event_dim: Optional[int] = None,
) -> T:
    raise NotImplementedError


ParamStore = collections.OrderedDict[str, tuple[torch.Tensor, Constraint, Optional[int]]]


class DefaultInterpretation(Generic[T], StatefulInterpretation[ParamStore, T]):
    state: ParamStore


@register(sample, DefaultInterpretation)
def default_sample(
    param_store: ParamStore,
    # ctx: Environment[T],
    result: Optional[T],
    name: str,
    distribution: Distribution,
    obs: Optional[T] = None
) -> T:
    return distribution.sample()


@register(sample, DefaultInterpretation)
def default_param(
    param_store: ParamStore,
    # ctx: Environment[T],
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


class PlateData(NamedTuple):
    name: str
    size: int
    dim: Optional[int]


@define(Operation)
def enter_plate(p: PlateData) -> PlateData:
    return p


@define(Operation)
def exit_plate(p: PlateData):
    pass


@define(Operation)
@contextlib.contextmanager
def plate(name: str, size: int, dim: Optional[int] = None):
    p = PlateData(name, size, dim)
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


class trace(Generic[T], StatefulInterpretation[Trace, T]):
    state: Trace


@register(sample, trace)
def trace_sample(
    tr: Trace,
    # ctx: Environment[T],
    result: Optional[T],
    name: str,
    distribution: Distribution,
    obs: Optional[T] = None
) -> T:
    result = fwd(result)
    tr[name] = SampleTraceNode(name, result, distribution, obs is not None)
    return result


@register(param, trace)
def trace_param(
    tr: Trace,
    # ctx: Environment[T],
    result: Optional[T],
    name: str,
    init_value: Optional[T] = None,
    constraint=torch.distributions.constraints.real,
    event_dim: Optional[int] = None,
) -> T:
    result = fwd(result)
    tr[name] = TraceNode(name, result, constraint, event_dim)
    return result


class replay(Generic[T], StatefulInterpretation[Trace, T]):
    state: Trace


@register(sample, replay)
def replay_sample(
    tr: Trace,
    # ctx: Environment[T],
    result: Optional[T],
    name: str,
    distribution: Distribution,
    obs: Optional[T] = None
) -> T:
    if result is None and name in tr:
        result = tr[name].value
    return fwd(result)


Observations = collections.OrderedDict[str, torch.Tensor]


class condition(Generic[T], StatefulInterpretation[Observations, T]):
    state: Observations


@register(sample, condition)
def condition_sample(
    state: Observations,
    # ctx: Environment[T],
    result: Optional[T],
    name: str,
    distribution: Distribution,
    obs: Optional[T] = None
) -> T:
    try:
        result = state[name]
    except KeyError:
        pass
    return fwd(result)


class block(Generic[T], StatefulInterpretation[Container[str], T]):
    state: Container[str]


@register(sample, block)
def block_sample(
    blocked: Container[str],
    # ctx: Environment[T],
    result: Optional[T],
    name: str,
    *args
) -> T:
    return reflect(result) if name in blocked else fwd(result)


@register(param, block)
def block_param(
    blocked: Container[str],
    # ctx: Environment[T],
    result: Optional[T],
    name: str,
    *args
) -> T:
    return reflect(result) if name in blocked else fwd(result)


def trace_elbo(param_store: ParamStore, pyro_model: Callable[..., T], guide: Callable[..., T], *args, **kwargs) -> torch.Tensor:

    with handler(product(DefaultInterpretation(param_store), prompt_calls(reflect, sample, param))):

        with handler(trace(Trace())) as guide_trace:
            guide(*args, **kwargs)

        with handler(replay(guide_trace)), handler(trace(Trace())) as model_trace:
            pyro_model(*args, **kwargs)

        elbo = 0.0
        for name, node in model_trace.items():
            if isinstance(node, SampleTraceNode):
                elbo = elbo + node.distribution.log_prob(node.value).sum()

        for name, node in guide_trace.items():
            if isinstance(node, SampleTraceNode):
                elbo = elbo - node.distribution.log_prob(node.value).sum()

        return -elbo


###############################################################################

@define(Operation)
def intervene(obs: T, act: Optional[T] = None) -> T:
    return act if act is not None else obs


class MultiWorldCounterfactual(Generic[T], StatefulInterpretation[List[PlateData], T]):
    state: List[PlateData]


@register(intervene, MultiWorldCounterfactual)
def multi_world_counterfactual_intervene(
    cf_plates: List[PlateData],
    result: Optional[T],
    obs: T,
    act: Optional[T] = None
) -> T:
    act_result = fwd(result)
    new_plate = PlateData("__intervention__", 2, -5)
    cf_plates.append(new_plate)
    return scatter(obs, act_result, dim=new_plate.dim)


@register(sample, MultiWorldCounterfactual)
def multi_world_counterfactual_sample(
    cf_plates: List[PlateData],
    result: Optional[T],
    name: str,
    dist: Distribution,
    obs: Optional[T] = None,
) -> T:
    with contextlib.ExitStack() as stack:
        for p in cf_plates:
            if p.name in indices_of(dist):
                stack.enter_context(handler(plate(p.name, p.size, p.dim)))
        return fwd(result)
