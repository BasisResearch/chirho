# from pyro.poutine.messenger import block_messengers
from pyro.poutine import block
from chirho.observational.handlers import condition
from pyro.poutine.replay_messenger import ReplayMessenger
import torch
import pyro


class FunctionalFactorAdded(torch.nn.Module):
    def __init__(self, prior, full_model_functional_of_prior, data, functional, pos_factor: bool):
        super().__init__()
        self.full_model_functional_of_prior = full_model_functional_of_prior
        self.prior = prior
        self.data = data
        self.pos_factor = pos_factor

        self.unconditioned_model = self.full_model_functional_of_prior(self.prior)

        # FIXME HACK for a specific situation.
        plated_unconditioned_model = lambda: self.unconditioned_model(n=100)

        self.conditioned_model = condition(plated_unconditioned_model, data=self.data)
        self.functional_estimator = functional(self.unconditioned_model)

    def forward(self):
        # This is a bit weird, because we need to
        # 1. run the conditioned model to add the prior and likelihood log probability contributions.
        # 2. run the functional wrt the model, but not conditioned on data, and instead just propagating the latents forward into the functional.
        # So ideally, we could block the functional from outside traces, except that the same guide trace replay from the conditioned model also needs to
        #  be replayed when the functional is computed.
        # 3. add a log factor for the functional.

        # Apply a trace here to recover any latent sites that the guide is replaying to. Note that this runs on post sample, while the guide runs on sample.
        with pyro.poutine.trace() as tr:
            self.conditioned_model()

        trace = tr.get_trace()

        # Manually filter out anything that isn't a latent site, so we can replay the guide trace in the functional while blocking everything outside.
        latents_site_name = set(trace.stochastic_nodes)
        for site_name in trace.nodes.keys():
            if site_name not in latents_site_name:
                trace.remove_node(site_name)

        with block(hide_all=True):  # FIXME obviously very heavy handed, and prevents e.g. outer contexts from making interventions etc.
            with ReplayMessenger(trace=trace):
                try:
                    functional_estimate = self.functional_estimator()
                except Exception as e:
                    raise  # just for breakpoint.

        sign = torch.tensor(1.) if self.pos_factor else torch.tensor(-1.)
        pyro.factor("functional_factor", torch.log(torch.relu(sign * functional_estimate) + 1e-20))
