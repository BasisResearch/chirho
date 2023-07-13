import logging
from contextlib import ExitStack

import pyro
import pytest
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoMultivariateNormal, AutoNormal

from chirho.dynamical.handlers import (
    NonInterruptingPointObservationArray,
    PointObservation,
    SimulatorEventLoop,
    simulate,
)
from chirho.dynamical.ops import State

from chirho.counterfactual.handlers import TwinWorldCounterfactual

from tests.dynamical.dynamical_fixtures import (
    SimpleSIRDynamics,
    SimpleSIRDynamicsBayes,
    bayes_sir_model,
    check_trajectories_match,
)

from chirho.observational.handlers.soft_conditioning import (
    AutoSoftConditioning,
    KernelSoftConditionReparam,
    RBFKernel,
    SoftEqKernel,
)

from chirho.dynamical.handlers import ODEDynamics, PointIntervention
from chirho.dynamical.ops import State, Trajectory

from pyro.distributions import Normal, Poisson, Uniform, constraints, Gamma

pyro.settings.set(module_local_params=True)

logger = logging.getLogger(__name__)

# Global variables for tests
init_state = State(S=torch.tensor(10.0), I=torch.tensor(1.0), R=torch.tensor(0.0))
tspan = torch.tensor([0.0, .3, .6, .9, 1.2])


# def run_svi_inference(model, n_steps=100, verbose=True, **model_kwargs):
#     guide = pyro.infer.autoguide.AutoMultivariateNormal(model)
#     adam = pyro.optim.Adam({"lr": 0.03})
#     svi = pyro.infer.SVI(model, guide, adam, loss=pyro.infer.Trace_ELBO())
#     # Do gradient steps
#     pyro.clear_param_store()
#     for step in range(1, n_steps + 1):
#         loss = svi.step(**model_kwargs)
#         if (step % 100 == 0) or (step == 1) & verbose:
#             print("[iteration %04d] loss: %.4f" % (step, loss))
#
#     return guide


def run_svi_inference(model, n_steps=100, verbose=True, **model_kwargs):
    guide = pyro.infer.autoguide.AutoMultivariateNormal(model)
    elbo = pyro.infer.Trace_ELBO()(model, guide)
    # initialize parameters
    elbo(**model_kwargs)
    adam = torch.optim.Adam(elbo.parameters(), lr=0.03)
    # Do gradient steps
    for step in range(1, n_steps + 1):
        adam.zero_grad()
        loss = elbo(**model_kwargs)
        loss.backward()
        adam.step()
        if (step % 100 == 0) or (step == 1) & verbose:
            print("[iteration %04d] loss: %.4f" % (step, loss))
    return guide


class SimpleSIRDynamicsBayesReparam(SimpleSIRDynamics):

    def observation(self, X: State[torch.Tensor]):
        # super().observation(X)

        # A flight arrives in a country that tests all arrivals for a disease. The number of people infected on the
        #  plane is a noisy function of the number of infected people in the country of origin at that time.
        u_ip = pyro.sample("u_ip", Normal(7., 2.))
        pyro.deterministic("infected_passengers", X.I + u_ip, event_dim=0)
        pyro.deterministic("X_I", X.I, event_dim=0)


def condition_counterfactual_commutes():

    # Four passengers tested positive for a disease after landing
    landing_data = {"infected_passengers": torch.tensor(15.0)}
    landing_time = .3 + 1e-4

    # In the counterfactual world, a super-spreader event occured at time 0.1
    # In the factual world, however, this did not occur.
    counterfactual = State(
        S=lambda s: s - torch.tensor(2.0),
        I=lambda i: i + torch.tensor(2.0),
    )
    superspreader_time = 0.1

    # We want to know how many people passengers would have bene infected at the
    #  time of landing had the super-spreader event not occurred.

    reparam_config = AutoSoftConditioning(scale=0.01, alpha=0.5)

    def counterf_model(tr=None):
        with SimulatorEventLoop():
            with PointObservation(time=landing_time, data=landing_data):
                with pyro.poutine.reparam(config=reparam_config):
                    with TwinWorldCounterfactual():
                        with PointIntervention(time=superspreader_time, intervention=counterfactual):
                            ret = simulate(
                                SimpleSIRDynamicsBayesReparam(),
                                init_state,
                                tspan,
                            )
                            return ret

    def conditioned_model(tr=None):
        with SimulatorEventLoop():
            with pyro.poutine.reparam(config=reparam_config):
                with PointObservation(time=landing_time, data=landing_data):
                    ret = simulate(
                        SimpleSIRDynamicsBayesReparam(),
                        init_state,
                        tspan,
                    )
                    return ret

    with pyro.poutine.trace() as tr:
        ret = counterf_model(tr)

    guide = run_svi_inference(conditioned_model, n_steps=500, verbose=True)

    pred = pyro.infer.Predictive(counterf_model, guide=guide, num_samples=100)()

    assert False


if __name__ == "__main__":
    condition_counterfactual_commutes()

