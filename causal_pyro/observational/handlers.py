import functools
import operator
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    Protocol,
    Set,
    TypedDict,
    TypeVar,
    Union,
)

import pyro
import pyro.distributions.constraints as constraints
import torch

from causal_pyro.indexed.ops import IndexSet, gather, indexset_as_mask
from causal_pyro.indexed.handlers import add_indices


T = TypeVar("T")

Kernel = Callable[[T, T], torch.Tensor]


class TorchKernel(torch.nn.Module):
    support: constraints.Constraint

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class SoftEqKernel(TorchKernel):
    """
    Kernel that returns a Bernoulli log-probability of equality.
    """

    support: constraints.Constraint = constraints.boolean
    alpha: torch.Tensor

    def __init__(self, alpha: Union[float, torch.Tensor], *, event_dim: int = 0):
        super().__init__()
        self.register_buffer("alpha", torch.as_tensor(alpha))
        if event_dim > 0:
            self.support = constraints.independent(constraints.boolean, event_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        eq = (x == y).to(dtype=self.alpha.dtype)
        return (
            pyro.distributions.Bernoulli(probs=self.alpha)
            .expand([1] * self.support.event_dim)
            .to_event(self.support.event_dim)
            .log_prob(eq)
        )


class RBFKernel(TorchKernel):
    """
    Kernel that returns a Normal log-probability of distance.
    """

    support: constraints.Constraint = constraints.real
    scale: torch.Tensor

    def __init__(self, scale: Union[float, torch.Tensor], *, event_dim: int = 0):
        super().__init__()
        self.register_buffer("scale", torch.as_tensor(scale))
        if event_dim > 0:
            self.support = constraints.independent(constraints.real, event_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (
            pyro.distributions.Normal(loc=0.0, scale=self.scale)
            .expand([1] * self.support.event_dim)
            .to_event(self.support.event_dim)
            .log_prob(x - y)
        )


class _MaskedDelta(Protocol):
    base_dist: pyro.distributions.Delta
    event_dim: int
    _mask: Union[bool, torch.Tensor]


class _DeterministicReparamMessage(TypedDict):
    name: str
    fn: _MaskedDelta
    value: torch.Tensor
    is_observed: Literal[True]


class KernelSoftConditionReparam(pyro.infer.reparam.reparam.Reparam):
    """
    Reparametrizer that allows approximate soft conditioning on a :func:`pyro.deterministic`
    site using a kernel function that compares the observed and computed values,
    as in approximate Bayesian computation methods from classical statistics.

    This may be useful for estimating counterfactuals in Pyro programs
    corresponding to structural causal models with exogenous noise variables.

    The kernel function should return a score corresponding to the
    log-probability of the observed value given the computed value,
    which is then added to the model's unnormalized log-joint probability
    using :func:`pyro.factor`  :

        :math:`\\log p(v' | v) \\approx K(v, v')`

    The score tensor returned by the kernel function must have shape equal
    or broadcastable to the ``batch_shape`` of the site.

    .. note::
        Kernel functions must be positive-definite and symmetric.
        For example, :class:`~RBFKernel` returns a Normal log-probability
        of the distance between the observed and computed values.
    """

    def __init__(self, kernel: Kernel[torch.Tensor]):
        self.kernel = kernel
        super().__init__()

    def apply(
        self, msg: _DeterministicReparamMessage
    ) -> pyro.infer.reparam.reparam.ReparamResult:
        name = msg["name"]
        event_dim = msg["fn"].event_dim
        observed_value = msg["value"]
        computed_value = msg["fn"].base_dist.v

        if observed_value is not computed_value:  # fast path for trivial case
            approx_log_prob = self.kernel(computed_value, observed_value)
            pyro.factor(f"{name}_approx_log_prob", approx_log_prob)

        new_fn = pyro.distributions.Delta(observed_value, event_dim=event_dim).mask(
            False
        )
        return {"fn": new_fn, "value": observed_value, "is_observed": True}


class AutoSoftConditioning(pyro.infer.reparam.strategies.Strategy):
    """
    Automatic reparametrization strategy that allows approximate soft conditioning
    on ``pyro.deterministic`` sites in a Pyro model.

    This may be useful for estimating counterfactuals in Pyro programs corresponding
    to structural causal models with exogenous noise variables.

    This strategy uses :class:`~KernelSoftConditionReparam` to approximate
    the log-probability of the observed value given the computed value
    at each :func:`pyro.deterministic` site whose observed value is different
    from its computed value.

    .. note::
        Implementation details are subject to change.
        Currently uses a few pre-defined kernels such as :class:`~SoftEqKernel`
        and :class:`~RBFKernel` which are chosen for each site based on
        the site's ``event_dim`` and ``support``.
    """

    def __init__(self, *, scale: float = 1.0, alpha: float = 1.0):
        self.alpha = alpha
        self.scale = scale
        super().__init__()

    @staticmethod
    def site_is_deterministic(msg: pyro.infer.reparam.reparam.ReparamMessage) -> bool:
        return (
            msg["is_observed"]
            and isinstance(msg["fn"], pyro.distributions.MaskedDistribution)
            and isinstance(msg["fn"].base_dist, pyro.distributions.Delta)
        )

    def configure(
        self, msg: pyro.infer.reparam.reparam.ReparamMessage
    ) -> Optional[pyro.infer.reparam.reparam.Reparam]:
        if not self.site_is_deterministic(msg) or msg["value"] is msg["fn"].base_dist.v:
            return None

        if msg["fn"].base_dist.v.is_floating_point():
            scale = self.scale * functools.reduce(
                operator.mul, msg["fn"].event_shape, 1.0
            )
            return KernelSoftConditionReparam(
                RBFKernel(scale=scale, event_dim=len(msg["fn"].event_shape))
            )

        if msg["fn"].base_dist.v.dtype in (
            torch.bool,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            alpha = self.alpha * functools.reduce(
                operator.mul, msg["fn"].event_shape, 1.0
            )
            return KernelSoftConditionReparam(
                SoftEqKernel(alpha=alpha, event_dim=len(msg["fn"].event_shape))
            )

        raise NotImplementedError(
            f"Could not reparameterize deterministic site {msg['name']}"
        )


class CutModule(pyro.poutine.messenger.Messenger):
    """
    Converts a Pyro model into a module using the "cut" operation
    """

    vars: Set[str]

    def __init__(self, vars: Set[str]):
        self.vars = vars
        super().__init__()

    def _pyro_sample(self, msg: dict[str, Any]) -> None:
        # There are 4 cases to consider for a sample site:
        # 1. The site appears in self.vars and is observed
        # 2. The site appears in self.vars and is not observed
        # 3. The site does not appear in self.vars and is observed
        # 4. The site does not appear in self.vars and is not observed
        if msg["name"] not in self.vars:
            if msg["is_observed"]:
                # use mask to remove the contribution of this observed site to the model log-joint
                msg["mask"] = (
                    msg["mask"] if msg["mask"] is not None else True
                ) & torch.tensor(False, dtype=torch.bool).expand(msg["fn"].batch_shape)
            else:
                pass

        # For sites that do not appear in module, rename them to avoid naming conflict
        if msg["name"] not in self.vars:
            msg["name"] = f"{msg['name']}_nuisance"


class CutComplementModule(pyro.poutine.messenger.Messenger):
    vars: Set[str]

    def __init__(self, vars: Set[str]):
        self.vars = vars
        super().__init__()

    def _pyro_sample(self, msg: dict[str, Any]) -> None:
        # There are 4 cases to consider for a sample site:
        # 1. The site appears in self.vars and is observed
        # 2. The site appears in self.vars and is not observed
        # 3. The site does not appear in self.vars and is observed
        # 4. The site does not appear in self.vars and is not observed
        if msg["name"] in self.vars:
            # use mask to remove the contribution of this observed site to the model log-joint
            msg["mask"] = (
                msg["mask"] if msg["mask"] is not None else True
            ) & torch.tensor(False, dtype=torch.bool).expand(msg["fn"].batch_shape)

    def _pyro_post_sample(self, msg: dict[str, Any]) -> None:
        if msg["name"] in self.vars:
            assert msg["is_observed"]


def cut(
    model: Optional[Callable] = None, *, vars: Set[str] = set()
) -> tuple[Callable, Callable]:
    if model is None:
        return functools.partial(cut, vars=vars)

    return CutModule(vars)(model), CutComplementModule(vars)(model)


class IndexCutModule(
    pyro.poutine.messenger.Messenger
):  # TODO subclass DependentMaskMessenger
    """
    Represent module and complement in a single Pyro model using plates
    """

    vars: Set[str]
    name: str

    def __init__(self, vars: Set[str], *, name: str = "__cut_plate"):
        self.vars = vars
        self.name = name
        super().__init__()

    def __enter__(self):
        add_indices(IndexSet(**{self.name: {0, 1}}))
        return super().__enter__()

    def _pyro_sample(self, msg: dict[str, Any]) -> None:
        if pyro.poutine.util.site_is_subsample(msg):
            return
        # There are 4 cases to consider for a sample site:
        # 1. The site appears in self.vars and is observed
        # 2. The site appears in self.vars and is not observed
        # 3. The site does not appear in self.vars and is observed
        # 4. The site does not appear in self.vars and is not observed

        # TODO inherit this logic from indexed.handlers.DependentMaskMessenger
        # use mask to remove the contribution of this observed site to the model log-joint
        cut_index = IndexSet(**{self.name: {0 if msg["name"] in self.vars else 1}})
        mask = indexset_as_mask(cut_index) | msg["is_observed"]  # TODO device
        msg["mask"] = mask if msg["mask"] is None else msg["mask"] & mask

        # expand distribution to make sure two copies of a variable are sampled
        msg["fn"] = msg["fn"].expand(
            torch.broadcast_shapes(msg["fn"].batch_shape, mask.shape)
        )

    def _pyro_post_sample(self, msg: dict[str, Any]) -> None:
        if pyro.poutine.util.site_is_subsample(msg):
            return

        if (not msg["is_observed"]) and (msg["name"] in self.vars):
            # TODO: enforce this constraint exactly
            value_one = gather(
                msg["value"],
                IndexSet(**{self.name: {0}}),
                event_dim=msg["fn"].event_dim,
            )
            value_two = gather(
                msg["value"],
                IndexSet(**{self.name: {1}}),
                event_dim=msg["fn"].event_dim,
            )
            eq_constraint = (
                -torch.abs(value_one.detach() - value_two)
                .expand(msg["value"].shape)
                .reshape(msg["fn"].batch_shape + (-1,))
                .mean(-1)
            )

            cut_index = IndexSet(**{self.name: {0 if msg["name"] in self.vars else 1}})
            with pyro.poutine.mask(mask=indexset_as_mask(cut_index)):  # TODO device
                pyro.factor(f"{msg['name']}_equality_contraint", eq_constraint)
