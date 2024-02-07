import torch
import chirho.contrib.compexp as ep
from typing import Tuple
import chirho.contrib.experiments.closed_form as cfe
from typing import Callable
from chirho.contrib.compexp.typedecs import ModelType
import pyro
from chirho.contrib.compexp.utils import kft
import warnings


class DecisionOptimizerAbstract:
    optim: torch.optim.Optimizer
    _lr: float
    flat_dparams: torch.nn.Parameter

    @property
    def lr(self):
        return self._lr

    @lr.setter
    def lr(self, lr):
        self._lr = lr
        for g in self.optim.param_groups:
            g['lr'] = lr

    def estimate_grad(self):
        raise NotImplementedError()

    def step_grad(self, grad_estimate_):
        self.flat_dparams.grad = grad_estimate_
        self.optim.step()

    def step(self):
        grad_estimate = self.estimate_grad()
        self.step_grad(grad_estimate)


class DecisionOptimizer(DecisionOptimizerAbstract):
    def __init__(
            self,
            flat_dparams: torch.nn.Parameter,
            model: ep.typedecs.ModelType,
            cost: ep.ComposedExpectation,
            expectation_handler: ep.ExpectationHandler,
            lr: float
    ):
        self.flat_dparams = flat_dparams
        self.model = model
        self.cost_grad: ep.ComposedExpectation = cost.grad(self.flat_dparams) if cost is not None else None
        self.expectation_handler = expectation_handler
        self._lr = lr

        self.optim = torch.optim.SGD((self.flat_dparams,), lr=self._lr)

    def estimate_grad(self):
        self.optim.zero_grad()
        self.flat_dparams.grad = None

        with self.expectation_handler:
            grad_estimate_: torch.tensor = self.cost_grad(self.model)

        return grad_estimate_


class DecisionOptimizerAnalyticCFE(DecisionOptimizerAbstract):
    # TODO this belongs inside the closed_form folder.
    def __init__(
            self,
            flat_dparams: torch.nn.Parameter,
            lr: float,
            problem: "cfe.CostRiskProblem"
    ):
        self.flat_dparams = flat_dparams
        self._lr = lr
        self.problem = problem

        self.optim = torch.optim.SGD((self.flat_dparams,), lr=self._lr)

    def estimate_grad(self):
        self.optim.zero_grad()
        self.flat_dparams.grad = None

        grad_estimate_ = self.problem.ana_loss_grad(self.flat_dparams)

        return grad_estimate_


class DecisionOptimizationManualMC(DecisionOptimizerAbstract):
    def __init__(
            self,
            flat_dparams: torch.nn.Parameter,
            lr: float,
            num_samples: int,
            model: ModelType,
            cost_grad_manual: Callable):
        super().__init__()

        self.flat_dparams = flat_dparams
        self._lr = lr
        self.cost_grad_manual = cost_grad_manual
        self.num_samples = num_samples
        self.model = model
        self.optim = torch.optim.SGD((self.flat_dparams,), lr=self._lr)

    def estimate_grad(self):
        self.optim.zero_grad()
        self.flat_dparams.grad = None

        with pyro.plate("samples", self.num_samples):
            stochastics = self.model()

        return self.cost_grad_manual(stochastics)


class DecisionOptimizationManualSNISNoGrad(DecisionOptimizerAbstract):
    SITES = ["z", "denormalization_factor"]  # TODO HACK 2789hfd cz we need elementwise logprobs

    def __init__(
            self,
            flat_dparams: torch.nn.Parameter,
            lr: float,
            num_samples: int,
            model: ModelType,
            gr,
            svi_lr: float,
            cost_grad_manual: Callable,
    ):
        super().__init__()

        self.flat_dparams = flat_dparams
        self._lr = lr
        self.cost_grad_manual = cost_grad_manual
        self.num_samples = num_samples
        self.model = model
        self.gr = gr
        self.svi_lr = svi_lr
        all_guides = list(self.gr.guides.values())
        assert len(all_guides) == 1, "SNIS no grad guide registry should only have one guide."
        self.guide = all_guides[0]
        self.optim = torch.optim.SGD((self.flat_dparams,), lr=self._lr)

    # TODO HACK 2789hfd cz we need elementwise logprobs
    def elementwise_log_probs_for_model(self, tr):
        ele_log_prob = torch.tensor(0.)
        for site_name in self.SITES:
            site = tr.nodes[site_name]
            elp = site["fn"].log_prob(site["value"], *site["args"], **site["kwargs"])
            ele_log_prob = elp + ele_log_prob
        return ele_log_prob

    # TODO HACK 2789hfd cz we need elementwise logprobs
    def elementwise_log_probs_for_guide(self, tr):
        ele_log_prob = torch.tensor(0.)
        for name, site in tr.nodes.items():
            if name.endswith("_latent"):
                elp = site["fn"].log_prob(site["value"], *site["args"], **site["kwargs"])
                ele_log_prob = elp + ele_log_prob
        return ele_log_prob

    def opt_guides(self):
        self.gr.optimize_guides(
            lr=self.svi_lr,
            n_steps=self.num_samples
        )

    def estimate_grad(self):
        self.optim.zero_grad()
        self.flat_dparams.grad = None

        self.opt_guides()

        with pyro.poutine.trace() as qtr:
            with pyro.plate("samples", self.num_samples):
                self.guide()
        qtr = qtr.get_trace()
        ptr = pyro.poutine.trace(pyro.poutine.replay(self.model, trace=qtr)).get_trace()
        stochastics = kft(qtr, "z")  # TODO HACK assumes we want z.

        grad_estimate_ = self.cost_grad_manual(stochastics)

        plp = self.elementwise_log_probs_for_model(ptr)
        qlp = self.elementwise_log_probs_for_guide(qtr)

        lw = plp - qlp
        lf_pos = lw[:, None] + torch.log(1e-25 + torch.relu(grad_estimate_))
        lf_neg = lw[:, None] + torch.log(1e-25 + torch.relu(-grad_estimate_))

        log_pos_num = torch.logsumexp(lf_pos, dim=0)
        log_neg_num = torch.logsumexp(lf_neg, dim=0)
        log_denominator = torch.logsumexp(lw, dim=0)

        est = (log_pos_num.exp() - log_neg_num.exp()) / log_denominator.exp()

        return est.detach()


class DecisionOptimizerHandlerPerPartial(DecisionOptimizer):
    """
    A quick, hacky version of the decision optimizer that takes multiple handlers, one for each dimension,
    where those handlers are presumed to be ...AllShared handlers with set guides where relevant.
    """
    def __init__(
            self,
            *args,
            expectation_handlers: Tuple[ep.ExpectationHandler],
            **kwargs):
        super().__init__(*args, expectation_handler=None, **kwargs)
        self.expectation_handlers = expectation_handlers

    def estimate_grad(self):
        self.optim.zero_grad()
        self.flat_dparams.grad = None

        grad_estimates = []
        for i, handler in enumerate(self.expectation_handlers):
            with handler:
                # This is the quick, hacky part — evaluate the whole gradient with this handler
                #  but only take the partial that its proposal is (presumably) targeting.
                grad_estimates.append(self.cost_grad(self.model)[i])

        return torch.stack(grad_estimates)
