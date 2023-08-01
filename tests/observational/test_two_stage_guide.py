import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from chirho.indexed.handlers import IndexPlatesMessenger
from chirho.observational.handlers.cut import (
    CutModule,
    CutComplementModule,
    SingleStageCut,
)
from pyro.infer import Predictive

pyro.settings.set(module_local_params=True)

pyro.set_rng_seed(321)  # for reproducibility


# Define a helper function to run SVI. (Generally, Pyro users like to have more control over the training process!)
def run_svi_inference(
    model,
    n_steps=100,
    verbose=True,
    lr=0.03,
    vi_family=AutoNormal,
    guide=None,
    module_one_guide=None,
    **model_kwargs
):
    if guide is None:
        guide = vi_family(model)
    elbo = pyro.infer.Trace_ELBO()(model, guide)
    # initialize parameters
    if module_one_guide is None:
        elbo(**model_kwargs)
    else:
        elbo(module_one_guide)
    adam = torch.optim.Adam(elbo.parameters(), lr=lr)
    # Do gradient steps
    for step in range(1, n_steps + 1):
        adam.zero_grad()
        if module_one_guide is None:
            loss = elbo(**model_kwargs)
        else:
            loss = elbo(module_one_guide)
        loss.backward()
        adam.step()
        if (step % 1000 == 0) or (step == 1) & verbose:
            print("[iteration %04d] loss: %.4f" % (step, loss))
    return guide


def cut_guide_wrapper(module_one_guide, module_two_guide):
    def cut_guide():
        with pyro.poutine.trace() as module_one_tr:
            module_one_guide()
        module_one_guide.requires_grad_(False)  # freeze module one parameters
        module_two_guide(module_one_tr)

    return cut_guide


def cut_svi_inference(model, module_one_vars, vi_family_one, vi_family_two, **kwargs):
    module_one = CutModule(module_one_vars)(model)
    module_two = CutComplementModule(module_one_vars)(model)

    # Fit module one
    module_one_guide = run_svi_inference(
        module_one, guide=vi_family_one(module_one), **kwargs
    )

    # Define guide for module two conditioned on module one, i.e. q(module_two latents | module_one latents)
    def module_two_cond_guide(module_one_tr, **kwargs):
        latent_names = [
            name
            for name, msg in module_one_tr.trace.nodes.items()
            if not msg["is_observed"]
        ]
        latent_names = [name for name in latent_names if name in module_one_vars]
        module_one_latents = dict(
            [(name, module_one_tr.trace.nodes[name]["value"]) for name in latent_names]
        )
        module_two_cond = pyro.condition(module_two, data=module_one_latents)
        vi_family_two(module_two_cond, **kwargs)()

    # Fit cut posterior
    cut_guide = cut_guide_wrapper(module_one_guide, module_two_cond_guide)
    cut_guide = run_svi_inference(model, guide=cut_guide, **kwargs)
    return cut_guide


class TwoStageCutGuide(pyro.nn.PyroModule):
    def __init__(self, model, module_one_vars):
        super().__init__()
        self.model = model
        self.module_one_vars = module_one_vars
        self.module_two = CutComplementModule(module_one_vars)(model)

    def forward(self, module_one_guide, **model_kwargs):
        module_one_guide.requires_grad_(False)  # freeze module one parameters
        with pyro.poutine.trace() as module_one_tr:
            module_one_guide(DUMMY_MODULE_ONE_HACK)
        module_one_guide.requires_grad_(False)

        latent_names = [
            name
            for name, msg in module_one_tr.trace.nodes.items()
            if not msg["is_observed"]
        ]
        latent_names = [name for name in latent_names if name in self.module_one_vars]
        module_one_latents = dict(
            [(name, module_one_tr.trace.nodes[name]["value"]) for name in latent_names]
        )
        module_two_cond = pyro.condition(self.module_two, data=module_one_latents)
        AutoNormal(module_two_cond)(DUMMY_MODULE_ONE_HACK)


def test_two_stage_inference_attempt1():
    def linear_gaussian_model():
        eta = pyro.sample("eta", dist.Normal(0, 1))
        theta = pyro.sample("theta", dist.Normal(0, 1))
        return observation_model(eta, theta)

    def observation_model(eta, theta):
        w = pyro.sample("w", dist.Normal(eta, 1))
        z = pyro.sample("z", dist.Normal(eta + theta, 1))
        return {"w": w, "z": z}

    # Fit module one
    module_one_vars = ["eta", "w"]
    cut_guide = cut_svi_inference(
        linear_gaussian_model,
        module_one_vars,
        vi_family_one=AutoNormal,
        vi_family_two=AutoNormal,
        n_steps=100,
    )


DUMMY_MODULE_ONE_HACK = 1.0


def test_two_stage_inference_attempt2():
    def linear_gaussian_model(module_one_guide):
        eta = pyro.sample("eta", dist.Normal(0, 1))
        theta = pyro.sample("theta", dist.Normal(0, 1))
        return observation_model(eta, theta)

    def observation_model(eta, theta):
        w = pyro.sample("w", dist.Normal(eta, 1))
        z = pyro.sample("z", dist.Normal(eta + theta, 1))
        return {"w": w, "z": z}

    # Fit module one
    module_one_vars = ["eta", "w"]
    module_one = CutModule(module_one_vars)(linear_gaussian_model)
    module_one_guide = run_svi_inference(
        module_one, guide=AutoNormal(module_one), module_one_guide=DUMMY_MODULE_ONE_HACK
    )

    # Fit module two
    two_stage_guide = TwoStageCutGuide(linear_gaussian_model, module_one_vars)

    run_svi_inference(
        linear_gaussian_model, guide=two_stage_guide, module_one_guide=module_one_guide
    )
