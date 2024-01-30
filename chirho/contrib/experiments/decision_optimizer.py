import torch
import chirho.contrib.compexp as ep
from typing import Tuple


class DecisionOptimizer:
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

    @property
    def lr(self):
        return self._lr

    @lr.setter
    def lr(self, lr):
        self._lr = lr
        for g in self.optim.param_groups:
            g['lr'] = lr

    def estimate_grad(self):
        self.optim.zero_grad()
        self.flat_dparams.grad = None

        with self.expectation_handler:
            grad_estimate_: torch.tensor = self.cost_grad(self.model)

        return grad_estimate_

    def step_grad(self, grad_estimate_):
        self.flat_dparams.grad = grad_estimate_
        self.optim.step()

    def step(self):
        grad_estimate = self.estimate_grad()
        self.step_grad(grad_estimate)


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
