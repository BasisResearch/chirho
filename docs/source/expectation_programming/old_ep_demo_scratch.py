import pyro
import torch
from pyro.infer.autoguide import AutoMultivariateNormal
import pyro.distributions as dist
import matplotlib.pyplot as plt
import seaborn as sns
import inspect
from torch import tensor as tt

from collections import OrderedDict

from typing import (
    Callable,
    TypeVar,
    Optional,
    Dict,
    List,
    Generic,
    Union
)

# from typing_extensions import Unpack

import numpy as np
import functools

from pyro.poutine.replay_messenger import ReplayMessenger

pyro.settings.set(module_local_params=True)


# FIXME Use the actual type from pyro. Gotta find ref. Just using this to force type checking.
class TraceType:
    @property
    def nodes(self) -> Dict[str, Dict[str, torch.Tensor]]:
        raise NotImplementedError()

    def log_prob_sum(self) -> torch.Tensor:
        raise NotImplementedError()


KWType = OrderedDict[str, torch.Tensor]
KWTypeNNParams = OrderedDict[str, torch.nn.Parameter]
KWTypeWithTrace = OrderedDict[str, Union[torch.Tensor, TraceType]]

# Model should involv epyro primitives.
ModelType = Callable[[], KWType]

# This should not involve pyro primitives
# This tuple return is primarily to support gradients of expectations.
ExpectigrandType = Callable[[KWType], KWType]

# This function takes both decision parameters and stochastics, and returns a tuple of things.
# This tuple return is primarily to support gradients of expectations.
DecisionExpectigrandType = Callable[[KWTypeNNParams, KWType], KWType]


# noinspection PyUnusedLocal
@pyro.poutine.runtime.effectful(type="expectation")
def expectation(
        f: ExpectigrandType,
        p: ModelType) -> KWType:
    raise NotImplementedError("Requires an expectation handler to function.")


@pyro.poutine.runtime.effectful(type="build_expectigrand_gradient")
def build_expectigrand_gradient(
        dparams: KWTypeNNParams,
        f: DecisionExpectigrandType
) -> ExpectigrandType:

    # Making this an effectful operation allows constraints to 1) add their gradients to the decision
    #  parameters and 2) add any auxiliary parameters to dparams that are required to represent themselves
    #  as an unconstrained problem.

    def df_dd(stochastics: KWType) -> KWType:

        y: KWType = f(dparams, stochastics)

        if len(y) != 1:
            # TODO eventually support multiple outputs? Will probably want to do this
            #  once constraints get properly involved.
            raise ValueError("Decision function must return a single tensor value.")

        ddparams = torch.autograd.grad(
            outputs=list(y.values()),
            inputs=list(dparams.values()),
            create_graph=True)

        return OrderedDict(zip((_gradify_dparam_name(k) for k in dparams.keys()), ddparams))

    return df_dd


def _gradify_dparam_name(name: str) -> str:
    return f"df_d{name}"


# noinspection PyUnusedLocal
@pyro.poutine.runtime.effectful(type="optimize_decision")
def optimize_decision(
        f: DecisionExpectigrandType,
        p: ModelType,
        terminal_condition: Callable[[KWTypeNNParams, int], bool],
        # Note the difference between this signature and the one in optimize_proposal. TODO unify.
        adjust_grads: Optional[Callable[[KWType], KWType]] = None,
        callback: Optional[Callable[[], None]] = None
) -> KWType:
    raise NotImplementedError("Requires both an expectation and optimizer handler.")


def msg_args_kwargs_to_kwargs(msg):

    ba = inspect.signature(msg["fn"]).bind(*msg["args"], **msg["kwargs"])
    ba.apply_defaults()

    return ba.arguments


class ExpectationHandler(pyro.poutine.messenger.Messenger):

    def _pyro_expectation(self, msg) -> None:
        if msg["done"]:
            raise ValueError("You may be operating in a context with more than one expectation handler. In these"
                             " cases, you must explicitly specify which expectation handler to use by using the OOP"
                             " style call with the desired handler (e.g. `ExpectationHandler(...).expectation(...)`).")

    def expectation(self, *args, **kwargs):
        # Calling this method blocks all other expectation handlers and uses only this one.
        with self:
            with pyro.poutine.messenger.block_messengers(lambda m: isinstance(m, ExpectationHandler) and m is not self):
                return expectation(*args, **kwargs)

    def optimize_proposal(self, *args, **kwargs):
        # Calling this method blocks all other expectation handlers and uses only this one.
        with self:
            with pyro.poutine.messenger.block_messengers(lambda m: isinstance(m, ExpectationHandler) and m is not self):
                return optimize_proposal(*args, **kwargs)


# Keyword arguments From Trace
def kft(trace: TraceType) -> KWType:
    # Copy the ordereddict.
    new_trace = OrderedDict(trace.nodes.items())
    # Remove the _INPUT and _RETURN nodes.
    del new_trace["_INPUT"]
    del new_trace["_RETURN"]
    return OrderedDict(zip(new_trace.keys(), [v["value"] for v in new_trace.values()]))


DT = TypeVar("DT")
DTr = TypeVar("DTr")


class OppableDictParent:  # FIXME 28dj10dlk
    pass


# TODO inherit from dict to just extend relevant methods, but couldn't figure out proper
#  inheritance of generics.
class OppableDict(OppableDictParent, Generic[DT]):
    """
    Helper class for executing operations on all the values of a dictionary.
    Heavily typed to make sure everything is lining up.
    """

    def __init__(self, d: Optional[OrderedDict[str, DT]] = None):
        if d is None:
            d: OrderedDict[str, DT] = OrderedDict()

        self._d = d

    def __getitem__(self, item: str) -> DT:
        return self._d[item]

    def __contains__(self, item: str) -> bool:
        return item in self._d

    def __setitem__(self, key: str, value: DT) -> None:
        self._d[key] = value

    def items(self):
        return self._d.items()

    def op(self, f: Callable[[DT], DTr]) -> 'OppableDict[DTr]':
        ret = OppableDict()

        for k, v in self.items():
            ret[k] = f(v)

        return ret

    def op_other(
            self,
            f: Callable[[DT, ...], DTr],
            *others: 'OppableDict[DT]') -> 'OppableDict[DTr]':
        ret = OppableDict()

        for k, v in self.items():
            ret[k] = f(v, *tuple(o[k] for o in others))

        return ret

    @functools.singledispatchmethod
    def __sub__(self, other):
        raise NotImplementedError()

    @__sub__.register(Union[float, torch.Tensor])
    def _(self, other: Union[float, torch.Tensor]) -> "OppableDict":
        return self.op(lambda v: v - other)

    # FIXME 28dj10dlk Once https://github.com/python/cpython/issues/86153 drops.

    # @__add__.register(OppableDict)  # FIXME 28dj10dlk
    @__sub__.register(OppableDictParent)
    # def _(self, other: "OppableDict") -> "OppableDict":  # FIXME 28dj10dlk desired
    def _(self, other: OppableDictParent) -> "OppableDict":  # FIXME 28dj10dlk
        assert isinstance(other, OppableDict)  # FIXME 28dj10dlk runtime check instead.
        return self.op_other(lambda v, o: v - o, other)

    @functools.singledispatchmethod
    def __add__(self, other):
        raise NotImplementedError()

    @__add__.register(Union[float, torch.Tensor])
    def _(self, other: Union[float, torch.Tensor]) -> "OppableDict":
        return self.op(lambda v: v + other)

    # FIXME 28dj10dlk
    @__add__.register(OppableDictParent)
    def _(self, other: OppableDictParent) -> "OppableDict":
        assert isinstance(other, OppableDict)
        return self.op_other(lambda v, o: v + o, other)

    @functools.singledispatchmethod
    def __truediv__(self, other):
        raise NotImplementedError()

    @__truediv__.register(Union[float, torch.Tensor])
    def _(self, other: Union[float, torch.Tensor]) -> "OppableDict":
        return self.op(lambda v: v / other)

    # @__truediv__.register(OppableDict)  # FIXME 28dj10dlk
    @__truediv__.register(OppableDictParent)
    # def _(self, other: "OppableDict") -> "OppableDict":  # FIXME 28dj10dlk desired
    def _(self, other: OppableDictParent) -> "OppableDict":  # FIXME 28dj10dlk
        assert isinstance(other, OppableDict)  # FIXME 28dj10dlk runtime check instead.
        return self.op_other(lambda v, o: v / o, other)

    @functools.singledispatchmethod
    def __mul__(self, other):
        raise NotImplementedError()

    @__mul__.register(Union[float, torch.Tensor])
    def _(self, other: Union[float, torch.Tensor]) -> "OppableDict":
        return self.op(lambda v: v * other)

    # FIXME 28dj10dlk desired
    @__mul__.register(OppableDictParent)
    def _(self, other: OppableDictParent) -> "OppableDict":
        assert isinstance(other, OppableDict)
        return self.op_other(lambda v, o: v * o, other)

    @property
    def wrapped(self) -> OrderedDict[str, DT]:
        return self._d


class DictOLists(OppableDict[List[torch.Tensor]]):
    def append(self, value: Dict[str, torch.Tensor]) -> None:

        for k, v in value.items():

            if k not in self:
                self[k] = []

            self[k].append(v)


class MonteCarloExpectation(ExpectationHandler):
    # Adapted from Rafal's "query library" code.

    def __init__(self, num_samples: int):
        self.num_samples = num_samples

        super().__init__()

    def _pyro_expectation(self, msg) -> None:
        super()._pyro_expectation(msg)

        kwargs = msg_args_kwargs_to_kwargs(msg)
        p: ModelType = kwargs["p"]
        f: ExpectigrandType = kwargs["f"]

        fvs = DictOLists()

        for i in range(self.num_samples):
            trace = pyro.poutine.trace(p).get_trace()
            ret = trace.nodes["_RETURN"]["value"]  # type: KWType
            fv = f(ret)
            fvs.append(fv)

        msg_value = fvs.op(lambda v: torch.sum(torch.tensor(v)) / self.num_samples)

        msg["value"] = msg_value
        msg["done"] = True


class SNISExpectation(ExpectationHandler):

    def __init__(self, q: ModelType, num_samples: int):
        self.q = q
        self.num_samples = num_samples

        super().__init__()

    def _pyro_expectation(self, msg) -> None:
        super()._pyro_expectation(msg)

        kwargs = msg_args_kwargs_to_kwargs(msg)
        q = self.q
        p = kwargs["p"]  # type: ModelType
        f = kwargs["f"]  # type: ExpectigrandType

        fvs = DictOLists()
        plps = []
        qlps = []

        for i in range(self.num_samples):
            # Sample stochastics from the proposal distribution.
            q_trace = pyro.poutine.trace(q).get_trace()
            qlp = q_trace.log_prob_sum()

            # Trace the full model with the proposed stochastics.
            with ReplayMessenger(trace=q_trace):
                trace = pyro.poutine.trace(lambda: f(p())).get_trace()

                # Record the return value and the log probability with respect to the model.
                fv = trace.nodes["_RETURN"]["value"]  # type: KWType
                plp = trace.log_prob_sum()

            fvs.append(fv)
            plps.append(plp)
            qlps.append(qlp)

        plps = torch.tensor(plps)
        qlps = torch.tensor(qlps)

        unw = plps - qlps  # unnormalized weights
        w = torch.exp(unw - torch.logsumexp(unw, dim=0))  # normalized weights

        msg_value = fvs.op(lambda v: torch.sum(torch.tensor(v) * w))

        msg["value"] = msg_value
        msg["done"] = True


# noinspection PyUnusedLocal
@pyro.poutine.runtime.effectful(type="optimize_proposal")
def optimize_proposal(p: ModelType, f: ExpectigrandType, n_steps=1, lr=0.01,
                      # Note the difference in signature here vs in the optimize_decision. TODO unify.
                      adjust_grads_: Optional[Callable[[torch.nn.Parameter, ...], None]] = None,
                      callback: Optional[Callable[[], None]] = None):
    raise NotImplementedError()


class LazySVIStuff:
    elbo: pyro.infer.Trace_ELBO = None
    optim = None


class TABIExpectation(ExpectationHandler):
    QTRACE_KEY = "TABIExpectation_q_trace_for_log_prob_and_replay"
    FAC_KEY = "TABIExpectation_expectation_targeting_log_factor"

    def __init__(self, q_plus: ModelType, q_den: ModelType, num_samples: int,
                 grad_clip: Optional[Callable] = None,
                 q_minus: Optional[ModelType] = None):
        self.q_plus = q_plus
        self.q_minus = q_minus
        self.q_den = q_den
        self.num_samples = num_samples
        self.grad_clip = grad_clip

        self._lazy_q_plus_svi = LazySVIStuff()
        self._lazy_q_minus_svi = LazySVIStuff()
        self._lazy_q_den_svi = LazySVIStuff()

        super().__init__()

    def __enter__(self):
        self._lazy_q_plus_svi = LazySVIStuff()
        self._lazy_q_minus_svi = LazySVIStuff()
        self._lazy_q_den_svi = LazySVIStuff()

        return super().__enter__()

    def _optimize_proposal_part(self, n_steps, elbo, optim, adjust_grads_=None,
                                callback: Optional[Callable[[], None]] = None):
        for step in range(0, n_steps + 1):
            for param in elbo.parameters():
                param.grad = None

            optim.zero_grad()
            loss = elbo()
            loss.backward()

            if adjust_grads_ is not None:
                adjust_grads_(*tuple(elbo.parameters()))

            if callback is not None:
                callback()

            optim.step()

    def get_part(self, sign: float, f: ExpectigrandType, p: ModelType):
        def factor_augmented_p():
            stochastics: KWType = p()
            fv: KWType = f(stochastics)
            fv_od = OppableDict(fv)

            # FIXME HACK 1e-6 is a hack to avoid log(0), make passable argument?
            facval = fv_od.op(lambda v: torch.log(1e-6 + torch.relu(sign * v)))

            for k, v in facval.items():
                aug_k = f'{self.FAC_KEY}_{sign}_{k}'
                fac = pyro.factor(aug_k, v)

                assert aug_k not in stochastics
                stochastics[aug_k] = fac

            return stochastics
        return factor_augmented_p

    @staticmethod
    def get_svi_stuff(p, q, svi_stuff, lr):
        if svi_stuff.elbo is None:
            svi_stuff.elbo = pyro.infer.Trace_ELBO()(p, q)
            svi_stuff.elbo()
        if svi_stuff.optim is None:
            svi_stuff.optim = torch.optim.ASGD(svi_stuff.elbo.parameters(), lr=lr)
        return svi_stuff

    def _pyro_optimize_proposal(self, msg) -> None:
        kwargs = msg_args_kwargs_to_kwargs(msg)
        p: ModelType = kwargs["p"]
        f: ExpectigrandType = kwargs["f"]
        n_steps: int = kwargs["n_steps"]
        lr: float = kwargs["lr"]
        adjust_grads_: Optional[Callable[[torch.nn.Parameter, ...], None]] = kwargs["adjust_grads_"]
        callback: Optional[Callable[[], None]] = kwargs["callback"]

        self.get_svi_stuff(self.get_part(1., f, p), self.q_plus, self._lazy_q_plus_svi, lr)
        self._optimize_proposal_part(n_steps, self._lazy_q_plus_svi.elbo, self._lazy_q_plus_svi.optim,
                                     adjust_grads_, callback=callback)

        self.get_svi_stuff(p, self.q_den, self._lazy_q_den_svi, lr)
        self._optimize_proposal_part(n_steps, self._lazy_q_den_svi.elbo, self._lazy_q_den_svi.optim,
                                     adjust_grads_, callback=callback)

        if self.q_minus is not None:
            self.get_svi_stuff(self.get_part(-1., f, p), self.q_minus, self._lazy_q_minus_svi, lr)
            self._optimize_proposal_part(n_steps, self._lazy_q_minus_svi.elbo, self._lazy_q_minus_svi.optim,
                                         adjust_grads_, callback=callback)

        msg["value"] = None
        msg["done"] = True

    def _pyro_expectation(self, msg) -> None:
        super()._pyro_expectation(msg)

        kwargs = msg_args_kwargs_to_kwargs(msg)
        q_plus = self.q_plus
        q_minus = self.q_minus
        q_den = self.q_den
        p = kwargs["p"]  # type: ModelType
        f = kwargs["f"]  # type: ExpectigrandType

        # TODO All this funkiness is to reuse logic between the component proposals, but its super opaque
        #  and annoying. It's also set up to get the actual proposal trace object to where it needs to go
        #  so that importance weights can be computed inside the expectigrand. Would like to clean up/simplify.
        #  This would allow us to get rid of KWTypeWithTrace also. Also, now that I've added the typing to sort
        #  out the redirection mess, it's like equally as verbose as not sharing code.

        def expectigrand(lf_: Optional[OppableDict[torch.Tensor]], q_trace: TraceType) -> KWType:
            with ReplayMessenger(trace=q_trace):
                unnorm_log_p = pyro.poutine.trace(p).get_trace().log_prob_sum()

            unnorm_log_q = q_trace.log_prob_sum()

            ret: KWType
            if lf_ is not None:
                ret = lf_.op(lambda v: torch.exp(v + unnorm_log_p - unnorm_log_q)).wrapped
            else:
                ret = OrderedDict(expected_importance_weight=torch.exp(unnorm_log_p - unnorm_log_q))
            return ret

        def get_signed_lf(kwstochastics: KWType, s: float) -> OppableDict[torch.Tensor]:
            fv = OppableDict(f(kwstochastics))
            return fv.op(lambda v: torch.log(torch.relu(s * v)))

        def expectigrand_plus(kwstochastics: KWTypeWithTrace) -> KWType:
            q_trace: TraceType = kwstochastics.pop(self.QTRACE_KEY)
            kwstochastics: KWType
            return expectigrand(lf_=get_signed_lf(kwstochastics, 1.), q_trace=q_trace)

        def expectigrand_minus(kwstochastics: KWTypeWithTrace) -> KWType:
            q_trace: TraceType = kwstochastics.pop(self.QTRACE_KEY)
            kwstochastics: KWType
            return expectigrand(lf_=get_signed_lf(kwstochastics, -1.), q_trace=q_trace)

        def expectigrand_den(kwstochastics: KWTypeWithTrace) -> KWType:
            q_trace: TraceType = kwstochastics.pop(self.QTRACE_KEY)
            kwstochastics: KWType
            return expectigrand(lf_=None, q_trace=q_trace)

        def get_get_qkwstochastics(q: ModelType) -> ModelType:
            def get_qkwstochastics() -> KWTypeWithTrace:
                qtr: TraceType = pyro.poutine.trace(q).get_trace()
                qkwstochastics = kft(qtr)
                # Add the log prob of the proposal for use down the line.
                assert self.QTRACE_KEY not in qkwstochastics
                qkwstochastics: KWTypeWithTrace
                qkwstochastics[self.QTRACE_KEY] = qtr
                return qkwstochastics

            return get_qkwstochastics

        with pyro.poutine.messenger.block_messengers(lambda m: m is self):

            with MonteCarloExpectation(self.num_samples):
                e_plus = expectation(
                    f=expectigrand_plus,
                    p=get_get_qkwstochastics(q_plus)
                )

            if q_minus is not None:
                with MonteCarloExpectation(self.num_samples):
                    e_minus = expectation(
                        f=expectigrand_minus,
                        p=get_get_qkwstochastics(q_minus)
                    )
            else:
                e_minus = 0.0

            with MonteCarloExpectation(self.num_samples):
                e_den: torch.Tensor = expectation(
                    f=expectigrand_den,
                    p=get_get_qkwstochastics(q_den)
                )["expected_importance_weight"]

        # Set this up to "broadcast" between the named tensors of the return dictionaries.
        e_plus_od = OppableDict(e_plus)
        e_minus_od: Union[OppableDict[torch.Tensor], float] = \
            OppableDict(e_minus) if isinstance(e_minus, OrderedDict) else e_minus

        msg["value"] = ((e_plus_od - e_minus_od) / e_den).wrapped
        msg["done"] = True


class ConstraintHandler(pyro.poutine.messenger.Messenger):
    def __init__(self, g: DecisionExpectigrandType, tau: float, threshold: float):
        """
        :param g: The function of decision parameters and stochastics that feeds into the constraint.
        :param tau: The scaling factor used to add the constraint gradient to the decision parameter gradient. This
        is required to convert constrained problems into unconstrained problems.
        :param threshold: The threshold value for the constraint.
        """

        self.g = g
        self.tau = tt(tau)
        self.threshold = tt(threshold)

# TODO Proposal Optimization for Constraints
# So I think these constraints need to have expectation handlers passed to them. They can add them to the stack
#  in the enter/exit, and then use the OOP strategy to call a specific optimize_proposal? I guess the constraints
#  should be able to preempt the optimize_proposal call and do their own optimization — they have everything they need
#  cz they have the constraint expectigrand passed directly to them. So no oop for optimize_proposal, the constraints
#  just preempt the call, run optimize_proposal in the context of whatever internal constraint expectation estimators
#  they happen to be using — they just need to block other expectation handlers during that execution.
# Remember that the DecisionOptimizationHandler automatically converts any optimize_proposal calls to operate over the
#  gradient, so proposal optimization here just needs to use the raw, non-differentiated function.


class MeanConstraintHandler(ConstraintHandler):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def _pyro_post_build_expectigrand_gradient(self, msg) -> None:
        kwargs = msg_args_kwargs_to_kwargs(msg)
        dparams: KWTypeNNParams = kwargs["dparams"]
        df_dd: ExpectigrandType = msg["value"]

        # Block all constraint handlers, because we just want the raw gradients here so we can add them.
        with pyro.poutine.messenger.block_messengers(lambda m: isinstance(m, ConstraintHandler)):
            dg_dd: ExpectigrandType = build_expectigrand_gradient(dparams, self.g)

        def rewrapped_df_dd(stochastics: KWType) -> KWType:

            # FIXME so the trick here is that these same stochastics aren't supposed to be sent into all
            #  three of these functions (df_dd, dg_dd, and g). Instead they're supposed to be three separate
            #  expectation operations. So what we need here is to actually make grad_expectation an
            #  operation, and then in post we need to call self.grad_expectation_handler.expectation and
            #  self.expectation_handler.expectation, and add everything up as needed.

            df_dd_ret = OppableDict(df_dd(stochastics))

            cret = self.g(dparams, stochastics)
            if len(cret) > 1:
                raise ValueError(f"{self.__class__.__name__} only supports scalar constraints,"
                                 f" but got return values named {tuple(cret.keys())}")

            cval = tuple(cret.values())[0]
            constraint_violated = cval > self.threshold

            if constraint_violated:
                print("Constraint Violated!")
                dg_dd_ret = OppableDict(dg_dd(stochastics))
                return (df_dd_ret + dg_dd_ret * self.tau).wrapped
            else:
                return df_dd_ret.wrapped

        msg["value"] = rewrapped_df_dd


class DecisionOptimizerHandler(pyro.poutine.messenger.Messenger):

    def __init__(self, dparams: KWTypeNNParams, lr: float,
                 proposal_update_steps: int = 0, proposal_update_lr: float = None,
                 proposal_adjust_grads_: Optional[Callable[[torch.nn.Parameter, ...], None]] = None):
        self.dparams = dparams
        self.lr = lr
        self.proposal_update_steps = proposal_update_steps
        self.proposal_update_lr = proposal_update_lr
        self.proposal_adjust_grads_ = proposal_adjust_grads_

        if self.proposal_update_steps > 0:
            assert self.proposal_update_lr is not None, "Learning rate for proposal update must be specified if " \
                                                        "proposal update steps is greater than 0."

    def _swap_f_for_df(self, msg) -> None:
        kwargs = msg_args_kwargs_to_kwargs(msg)
        f: DecisionExpectigrandType = kwargs["f"]
        fp: ExpectigrandType = build_expectigrand_gradient(self.dparams, f)
        kwargs["f"] = fp
        msg["kwargs"] = kwargs

    def _pyro_optimize_proposal(self, msg) -> None:
        # This handler replaces the function of interest with the gradient of that function with respect to the
        #  decision parameters. This supports any proposal optimization by giving it the correct target function.
        self._swap_f_for_df(msg)

    def _swap_f_for_f_of_d(self, msg) -> None:
        # This converts the decision function to a function just of stochastics so it can be used with
        #  standard expectation operations.
        kwargs = msg_args_kwargs_to_kwargs(msg)
        f: DecisionExpectigrandType = kwargs["f"]

        # Only actually swap this out if it's required. This allows users to pass in the results of
        #  build_expectigrand_gradient, which already returns a non-decision expectigrand, without
        #  having to block this handler's effect.
        if len(inspect.signature(f).parameters) == 1:
            return

        fp: ExpectigrandType = lambda stochastics: f(self.dparams, stochastics)
        kwargs["f"] = fp
        msg["kwargs"] = kwargs

    def _pyro_expectation(self, msg) -> None:
        # Expectation calls within this handler need to include the decision parameters as arguments,
        #  so this converts things to the default case not involving decision parameters.
        self._swap_f_for_f_of_d(msg)

    def _pyro_optimize_decision(self, msg) -> None:
        kwargs = msg_args_kwargs_to_kwargs(msg)
        f: DecisionExpectigrandType = kwargs["f"]
        p: ModelType = kwargs["p"]
        adjust_grads: Optional[Callable[[KWType], KWType]] = kwargs["adjust_grads"]
        terminal_condition: Callable[[KWTypeNNParams, int], bool] = kwargs["terminal_condition"]
        callback: Optional[Callable[[], None]] = kwargs["callback"]

        optim = torch.optim.SGD(list(self.dparams.values()), lr=self.lr)

        i = 0
        while not terminal_condition(self.dparams, i):

            optim.zero_grad()
            # Not sure if this necessary.
            for d in self.dparams.values():
                d.grad = None

            # Block self here because the gradient function already handles the partial evaluation. I.e.
            #  we don't want _swap_f_for_f_of_d called, because build_expectation_gradient already returns a valid
            #  ExpectigrandType.
            with pyro.poutine.messenger.block_messengers(lambda m: m is self):
                grad_estimate = expectation(f=build_expectigrand_gradient(self.dparams, f), p=p)

                res = OppableDict(grad_estimate).op(lambda v: torch.isfinite(v))
                if not torch.tensor(tuple(res.wrapped.values())).all():
                    print("Warning: Non-finite gradient estimate encountered. Skipping update.")  # TODO warnings module
                    continue

            if adjust_grads is not None:
                grad_estimate: KWType = adjust_grads(grad_estimate)

            # Sanity check.
            assert tuple(grad_estimate.keys()) == tuple(_gradify_dparam_name(k) for k in self.dparams.keys())

            # Assign gradients to parameters.
            for k, dp in self.dparams.items():
                dp.grad = grad_estimate[_gradify_dparam_name(k)]

            # And update those parameters with the specified optimzier.
            optim.step()

            if self.proposal_update_steps > 0:
                optimize_proposal(p=p, f=f, n_steps=self.proposal_update_steps, lr=self.proposal_update_lr,
                                  adjust_grads_=self.proposal_adjust_grads_)

            if callback is not None:
                callback()

            i += 1

        # msg["value"] = tuple(d.item() for d in self.dparams)
        msg["value"] = tuple(v.item() for v in self.dparams.values())
        msg["done"] = True


class MultiModalGuide1D(pyro.nn.PyroModule):

    @pyro.nn.PyroParam(constraint=dist.constraints.simplex)
    def pi(self):
        return torch.ones(self.num_components) / self.num_components

    @pyro.nn.PyroParam(constraint=dist.constraints.positive)
    def scale(self):
        return self.init_scale

    @pyro.nn.PyroParam(constraint=dist.constraints.real)
    def loc(self):
        return self.init_loc

    def __init__(self, *args, num_components: int, init_loc, init_scale, studentt=False, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_components = num_components
        self.init_loc = torch.tensor(init_loc)
        self.init_scale = torch.tensor(init_scale)
        self.studentt = studentt

        if self.init_loc.shape != (self.num_components,):
            raise ValueError("init_loc must be a tensor of shape (num_components,)")
        if self.init_scale.shape != (self.num_components,):
            raise ValueError("init_scale must be a tensor of shape (num_components,)")

    def forward(self):
        component_idx = pyro.sample("component_idx", dist.Categorical(self.pi), infer={'is_auxiliary': True})
        scale = torch.clip(self.scale[component_idx], torch.tensor(0.05), torch.tensor(1.5))

        if self.studentt:
            x = pyro.sample("x", dist.StudentT(10, self.loc[component_idx], scale))
        else:
            x = pyro.sample("x", dist.Normal(self.loc[component_idx], scale))
        return OrderedDict(x=x)

    def __call__(self, *args, **kwargs) -> KWType:
        return super().__call__(*args, **kwargs)


def clip_decision_grads(grads: KWType):
    # These blow up sometimes, so just clip them (they are 1d so no need to worry about norms).
    ret_grads = OrderedDict()
    for k in grads:
        ret_grads[k] = torch.clip(grads[k], -2e-1, 2e-1)
    return ret_grads


def abort_guide_grads_(*parameters: torch.nn.Parameter, lim=50.):
    # These gradients also blow up, but clipping them causes weird non-convergence. Just aborting
    #  the gradient update seems to work.
    if torch.any(torch.tensor([torch.any(param.grad > lim) for param in parameters])):
        for param in parameters:
            param.grad = torch.zeros_like(param.grad)


def main():

    # See https://www.desmos.com/calculator/ixtzpb4l75 for analytical computation of the expectation.
    # This all amounts to a functional on C and D, which can be set in the desmos graph.
    C = tt(1.)
    D = OrderedDict(d=torch.nn.Parameter(tt(0.5)))
    NSnaive = 100000
    GT = -1.1337
    print(f"Ground Truth: {GT}")

    # Freeze this decision parameter, as we're using it as a non-optimizeable constant.
    for d in D.values():
        d.requires_grad = False

    from toy_tabi_problem import (
        model as model_ttp,
        cost as _cost_ttp,
        q_optimal_normal_guide_mean_var,
        MODEL_DIST as MODEL_TTP_DIST
    )

    # A scaled up cost function, just because the original one is small.
    def dparam_scost_ttp(dparams: KWTypeNNParams, stochastics: KWType, c: torch.Tensor) -> KWType:
        return OrderedDict(cost=1e1 * _cost_ttp(**dparams, **stochastics, c=c))

    def get_scost_ttp(dparams: KWTypeNNParams, c: torch.Tensor) -> Callable[[KWType], KWType]:
        return lambda stochastics: dparam_scost_ttp(dparams, stochastics, c=c)

    def get_dparam_scost_ttp(c: torch.Tensor) -> Callable[[KWTypeNNParams, KWType], KWType]:
        return lambda dparams, stochastics: dparam_scost_ttp(dparams, stochastics, c=c)

    with MonteCarloExpectation(num_samples=NSnaive):
        print(f"MCE TABI Toy (N={NSnaive})",
              expectation(f=get_scost_ttp(dparams=D, c=C), p=model_ttp)["cost"])

    def subopt_guide() -> KWType:
        return OrderedDict(x=pyro.sample('x', dist.Normal(0.5, 2.)))

    with SNISExpectation(
        q=subopt_guide,
        num_samples=50000
    ):
        print(f"SNIS SubOptGuide TABI Toy (N={NSnaive})",
              expectation(f=get_scost_ttp(dparams=D, c=C), p=model_ttp)["cost"])

    # Do SNIS again but with an optimized proposal. We use studentt here because SNIS requires the division of the
    #  original probability by the proposal probability, so if something is sampled in the tails of the proposal
    #  but not in the tails of the original, this will blow up and cause problems. It does this before it ever
    #  sees the cost function in order self-normalize the weights.
    # noinspection PyUnresolvedReferences
    snis_opt_guidep = dist.StudentT(1, *q_optimal_normal_guide_mean_var(**D, c=C, z=False))
    # noinspection PyUnresolvedReferences
    snis_opt_guiden = dist.StudentT(1, *q_optimal_normal_guide_mean_var(**D, c=C, z=True))

    def opt_guide_mix() -> KWType:
        # noinspection PyUnresolvedReferences
        if pyro.sample('z', dist.Bernoulli(probs=torch.tensor([0.5]))):
            x = pyro.sample('x', snis_opt_guidep)
        else:
            x = pyro.sample('x', snis_opt_guiden)

        return OrderedDict(x=x)

    NS_snis_opt = 10000

    with SNISExpectation(
        # The bi-modal optimal proposal covers the product of the model and the absval of the cost function.
        q=opt_guide_mix,
        num_samples=NS_snis_opt
    ):
        print(f"SNIS OptGuide TABI Toy (N={NS_snis_opt})",
              expectation(f=get_scost_ttp(dparams=D, c=C), p=model_ttp)["cost"])

    tabi_opt_guidep = dist.Normal(*q_optimal_normal_guide_mean_var(**D, c=C, z=False))
    tabi_opt_guiden = dist.Normal(*q_optimal_normal_guide_mean_var(**D, c=C, z=True))

    with TABIExpectation(
        q_plus=lambda: OrderedDict(x=pyro.sample('x', tabi_opt_guidep)),
        q_minus=lambda: OrderedDict(x=pyro.sample('x', tabi_opt_guiden)),
        q_den=lambda: OrderedDict(x=pyro.sample('x', MODEL_TTP_DIST)),
        num_samples=1
    ):
        print(f"TABI Toy Exact (N=1)", expectation(f=get_scost_ttp(dparams=D, c=C), p=model_ttp)["cost"])

    # <Manual Inference of Proposals>

    # noinspection DuplicatedCode
    def pos_comp():
        xp = model_ttp()
        pos_fac = pyro.factor(
            'pos_fac', torch.log(1e-6 + torch.relu(dparam_scost_ttp(dparams=D, stochastics=xp, c=C)["cost"])))

        return OrderedDict(x=xp, pos_fac=pos_fac)

    # noinspection DuplicatedCode
    def neg_comp():
        xn = model_ttp()
        neg_fac = pyro.factor(
            'neg_fac', torch.log(1e-6 + torch.relu(-dparam_scost_ttp(dparams=D, stochastics=xn, c=C)["cost"])))

        return OrderedDict(x=xn, neg_fac=neg_fac)

    N_STEPS = 10000
    LR = 1e-3
    NS = 100

    # noinspection DuplicatedCode
    def run_svi_inference(model_, guide, n_steps=100, verbose=True):
        elbo = pyro.infer.Trace_ELBO()(model_, guide)
        elbo()
        optim = torch.optim.SGD(elbo.parameters(), lr=LR)
        for step in range(0, n_steps):
            optim.zero_grad()
            loss = elbo()
            loss.backward()
            optim.step()
            if (step % 100 == 0) & verbose:
                print("[iteration %04d] loss: %.4f" % (step, loss))
        return guide

    def get_num_guide_init(d, c, z: bool):
        iloc, istd = q_optimal_normal_guide_mean_var(d=d, c=c, z=z)
        return pyro.infer.autoguide.AutoNormal(
            model=pos_comp,
            init_loc_fn=pyro.infer.autoguide.initialization.init_to_value(values={'x': iloc}),
            init_scale=istd.item() * 3.,  # scale up the std to give SVI something to do.
        ), iloc, istd
    pos_guide, pos_iloc, pos_istd = get_num_guide_init(**D, c=C, z=False)
    neg_guide, neg_iloc, neg_istd = get_num_guide_init(**D, c=C, z=True)

    def get_den_guide_init():
        return pyro.infer.autoguide.AutoNormal(
            model=model_ttp,
            init_loc_fn=pyro.infer.autoguide.initialization.init_to_value(
                # Very important to get a COPY of this tensor and not pass the model parameter itself. Otherwise
                #  when the guide updates the model will also change, which naturally leads to insanity.
                values={'x': torch.tensor(MODEL_TTP_DIST.loc.item())}),
            init_scale=MODEL_TTP_DIST.scale.item() * 3.,  # scale up the std to give SVI something to do.
        )
    den_guide = get_den_guide_init()

    def plot_tabi_guides(tabi_handler: TABIExpectation, og_pos, og_neg, og_den):
        plt.figure()
        xx_ = torch.linspace(-10, 10, 1000)
        sns.kdeplot([tabi_handler.q_plus.forward()['x'].item() for _ in range(10000)],
                    label='pos', linestyle='--', color='red')
        sns.kdeplot([tabi_handler.q_minus.forward()['x'].item() for _ in range(10000)],
                    label='neg', linestyle='--', color='blue')
        sns.kdeplot([tabi_handler.q_den.forward()['x'].item() for _ in range(10000)],
                    label='den', linestyle='--', color='green')

        plt.plot(xx_, og_den.log_prob(xx_).exp(), color='green', alpha=0.5)
        plt.plot(xx_, og_pos.log_prob(xx_).exp(), color='red', alpha=0.5)
        plt.plot(xx_, og_neg.log_prob(xx_).exp(), color='blue', alpha=0.5)

        plt.show()

    # # <-----Manual execution>
    #
    # run_svi_inference(pos_comp, pos_guide, n_steps=N_STEPS, verbose=False)
    # run_svi_inference(neg_comp, neg_guide, n_steps=N_STEPS, verbose=False)
    # run_svi_inference(model_ttp, den_guide, n_steps=N_STEPS, verbose=False)
    # with TABIExpectation(
    #     q_plus=pos_guide,
    #     q_minus=neg_guide,
    #     q_den=den_guide,
    #     num_samples=NS
    # ) as te:
    #     print(f"TABI Toy Learned Guide (Manual) (N={NS})",
    #           expectation(f=get_scost_ttp(D, c=C), p=model_ttp))
    #
    #     plot_tabi_guides(te, dist.Normal(pos_iloc, pos_istd), dist.Normal(neg_iloc, neg_istd), MODEL_TTP_DIST)
    #
    # # </-----Manual execution>

    # </Manual Inference of Proposals>

    # <Integrated Guide Learning>

    with TABIExpectation(
        q_plus=get_num_guide_init(**D, c=C, z=False)[0],
        q_minus=get_num_guide_init(**D, c=C, z=True)[0],
        q_den=get_den_guide_init(),
        num_samples=NS
    ) as te:
        optimize_proposal(p=model_ttp, f=get_scost_ttp(dparams=D, c=C), n_steps=N_STEPS, lr=LR)

        tabi_ress = []
        for _ in range(100):
            tabi_ress.append(expectation(f=get_scost_ttp(dparams=D, c=C), p=model_ttp)["cost"])

        plt.figure()
        plt.suptitle(f"TABI Toy Learned Guide (Integrated) (N={NS})")
        plt.hist(tabi_ress, bins=20)
        plt.axvline(x=GT, color='black', linestyle='--')

        plot_tabi_guides(te, dist.Normal(pos_iloc, pos_istd), dist.Normal(neg_iloc, neg_istd), MODEL_TTP_DIST)

    # </Integrated Guide Learning>

    # <Manual Decision Optimization>

    # Wrap in a tensor so we can optimize it.
    dparams = OrderedDict(d=torch.nn.Parameter(torch.tensor(-1.)))
    cval = tt(2.)  # A value of 2 makes this a bit more difficult than the above.

    dprogression = [dparams['d'].item()]
    dgrads = []

    # The guides for the gradients have to be multi-modal to track with the multi-modal cost function.
    # While the positive and negative components of the non-differential are themselves unimodal, when working
    #  directly with the gradients, each pos/neg component has a positive and negative component themselves.
    # This means each positive/negative guide component has to be multimodal.

    # Plot the d/dd expectigrand as a function of x.
    plt.figure()
    xx = torch.linspace(-5., 5., 100)
    ddf = build_expectigrand_gradient(dparams, get_dparam_scost_ttp(c=cval))
    plt.plot(xx, [
        (OppableDict(ddf(OrderedDict(x=x)))*MODEL_TTP_DIST.log_prob(x).exp())[_gradify_dparam_name("d")].detach().item()
        for x in xx])
    plt.show()

    # We initialize guides with components at both the positive and negative components of the cost function. They
    #  will then adjust to capture the two components of each side of the gradient.
    pos_iloc_grad, pos_istd_grad = q_optimal_normal_guide_mean_var(**D, c=cval, z=False)
    neg_iloc_grad, neg_istd_grad = q_optimal_normal_guide_mean_var(**D, c=cval, z=True)
    pos_guide_grad = MultiModalGuide1D(
        num_components=2,
        init_loc=[pos_iloc_grad.item(), neg_iloc_grad.item()],
        init_scale=[pos_istd_grad.item(), neg_istd_grad.item()]
    )
    neg_guide_grad = MultiModalGuide1D(
        num_components=2,
        init_loc=[pos_iloc_grad.item(), neg_iloc_grad.item()],
        init_scale=[pos_istd_grad.item(), neg_istd_grad.item()]
    )
    # The denominator doesn't involve the cost function, so it stays the same.

    def plotss_(title, te):
        # Plot the d/dd expectigrand as a function of x.
        plt.figure()
        plt.suptitle(title)

        def plot_part_(sign, color):
            xy = []
            with pyro.poutine.trace() as og_tr:
                te.get_part(sign, f=ddf, p=model_ttp)()
            for x in torch.linspace(-4., 4., 1000):
                og_tr.get_trace().nodes['x']['value'] = x
                with ReplayMessenger(og_tr.trace):
                    with pyro.poutine.trace() as tr:
                        te.get_part(sign, f=ddf, p=model_ttp)()
                xy.append((
                    tr.get_trace().nodes['x']['value'].item(),
                    tr.get_trace().log_prob_sum().exp().item()
                ))
            xy = np.array(xy)
            plt.plot(*xy.T, color=color, linestyle='--')

        plot_part_(1., 'blue')
        plot_part_(-1., 'red')

        # Plot an sns density plot of the guides. Use the same figure.
        sns.kdeplot([te.q_plus()['x'].item() for _ in range(1000)], label="q_plus", bw_method=0.05, color='blue')
        sns.kdeplot([te.q_minus()['x'].item() for _ in range(1000)], label="q_minus", bw_method=0.05, color='red')
        sns.kdeplot([te.q_den()['x'].item() for _ in range(1000)], label="q_den", bw_method=0.25, color='green')

        # Also plot the true model density.
        sns.kdeplot([model_ttp()['x'].item() for _ in range(1000)],
                    label="model", bw_method=0.25, color='green', linestyle='--')

        plt.legend()
        plt.xlim(-5., 5.)
        plt.show()
        plt.close()

    # # <----Manual Execution>
    # with TABIExpectation(
    #     q_plus=pos_guide_grad,
    #     q_minus=neg_guide_grad,
    #     q_den=get_den_guide_init(),
    #     num_samples=1
    # ) as te_:
    #
    #     plotss_("Before", te_)
    #
    #     grad_estimate = expectation(f=ddf, p=model_ttp)
    #     print(f"TABI Toy Learned Grads (Truth: -0.0575) (Before): {grad_estimate}")
    #
    #     for _ in range(5):
    #         optimize_proposal(p=model_ttp, f=ddf, n_steps=3000, lr=1e-4, adjust_grads_=abort_guide_grads_)
    #
    #         plotss_("After", te_)
    #
    #         grad_estimate = expectation(f=ddf, p=model_ttp)
    #         print(f"TABI Toy Learned Grads (Truth: -0.0575) (After): {grad_estimate}")
    #
    #     # Iteratively optimize the decision variable and the proposals.
    #     optim_ = torch.optim.SGD(tuple(dparams.values()), lr=1e-1)
    #
    #     ii = 1
    #     cii = 0
    #     while cii < 30:
    #
    #         optim_.zero_grad()
    #         for dp in dparams.values():
    #             dp.grad = None
    #
    #         # Move the decision variable.
    #         grad_estimate = expectation(f=ddf, p=model_ttp)
    #         print("Gradient Estimate", grad_estimate["d"].item())
    #
    #         grad_estimate = clip_decision_grads(grad_estimate)
    #
    #         for k, dp in dparams.items():
    #
    #             dp.grad = grad_estimate[k]
    #
    #         optim_.step()
    #
    #         # Then update the proposals.
    #         optimize_proposal(p=model_ttp, f=ddf, n_steps=300, lr=3e-5, adjust_grads_=abort_guide_grads_)
    #
    #         if ii % 30 == 0:
    #             plotss_(f"Decision {round(dparams['d'].item(), 3)}", te_)
    #
    #         # And append the state of dten.
    #         dprogression.append(dparams["d"].item())
    #         dgrads.append(grad_estimate["d"].item())
    #
    #         ii += 1
    #
    #         if abs(dparams["d"].item()) < 1e-3:
    #             cii += 1
    #         else:
    #             cii = 0
    #
    # plt.figure()
    # plt.plot(dprogression)
    # plt.title("Decision Progression")
    #
    # plt.figure()
    # plt.plot(dgrads)
    # plt.title("Decision Gradients")
    #
    # plt.show()
    # # </----Manual Execution>

    # </Manual Decision Optimization>

    # <Integrated Decision Optimization>

    te_ = TABIExpectation(
        q_plus=pos_guide_grad,
        q_minus=neg_guide_grad,
        q_den=get_den_guide_init(),
        num_samples=1)
    dh_ = DecisionOptimizerHandler(dparams=dparams, lr=1e-1, proposal_update_lr=3e-5, proposal_update_steps=300,
                                   proposal_adjust_grads_=abort_guide_grads_)

    with te_, dh_:

        plotss_("Before", te_)

        grad_estimate = expectation(f=ddf, p=model_ttp)
        print(f"TABI Toy Learned Grads (Before): {grad_estimate}")

        # Perform initial optimization of proposals.
        for _ in range(2):
            # In the dh_ handler, the function here will be converted to the gradient with respect to d.
            # TODO I don't know if I like this being implicit...?
            optimize_proposal(p=model_ttp, f=get_dparam_scost_ttp(c=cval), n_steps=3000, lr=1e-4,
                              adjust_grads_=abort_guide_grads_)

            plotss_("After", te_)

            grad_estimate = expectation(f=ddf, p=model_ttp)
            print(f"TABI Toy Learned Grads (After): {grad_estimate}")

        def terminal_condition_(dparams_, i):

            if i % 10 == 0:
                plotss_(f"Decision {round(dparams_['d'].item(), 3)}", te_)

            ret = abs(dparams_['d'].item()) < 1e-2

            if ret:
                plotss_(f"Decision {round(dparams_['d'].item(), 3)}", te_)

            return ret

        optimize_decision(f=get_dparam_scost_ttp(c=cval), p=model_ttp,
                          terminal_condition=terminal_condition_,
                          adjust_grads=clip_decision_grads)

    # </Integrated Decision Optimization>

    exit()


if __name__ == "__main__":
    main()
