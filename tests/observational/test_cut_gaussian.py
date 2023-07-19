import torch
import math
import pyro
import pyro.distributions as dist
import pyro.infer.reparam
from pyro.infer.autoguide import AutoMultivariateNormal
from pyro.infer import SVI, Trace_ELBO


from chirho.indexed.handlers import IndexPlatesMessenger
from chirho.observational.handlers.soft_conditioning import IndexCutModule, cut


# Simulated data configuration
TRUE_ETA = torch.tensor(2.0)
TRUE_THETA = torch.tensor(0.0)
NUM_SAMPS_MODULE_ONE = 2
NUM_SAMPS_MODULE_TWO = 1000
SIGMA_ONE = 3.0
SIGMA_TWO = 0.1
ETA_COEF = 1.0


def observation_model(eta, theta):
    with pyro.plate("module_one_plate", NUM_SAMPS_MODULE_ONE):
        w = pyro.sample("w", dist.Normal(eta, SIGMA_ONE))
    with pyro.plate("module_two_plate", NUM_SAMPS_MODULE_TWO):
        z = pyro.sample("z", dist.Normal(ETA_COEF * eta + theta, SIGMA_TWO))
    return {"w": w, "z": z}


def linear_gaussian_model():
    eta = pyro.sample("eta", dist.Normal(0, 1))
    theta = pyro.sample("theta", dist.Normal(0, 1))
    return observation_model(eta, theta)


def make_cut_index_model(model, module_one_vars, *args, **kwargs):
    def cut_index_model():
        with IndexPlatesMessenger(), IndexCutModule(module_one_vars):
            model(*args, **kwargs)

    return cut_index_model


def run_svi_inference(model, n_steps=100, verbose=True, **model_kwargs):
    guide = AutoMultivariateNormal(model)
    adam = pyro.optim.Adam({"lr": 0.03})
    svi = SVI(model, guide, adam, loss=Trace_ELBO())
    # Do gradient steps
    pyro.clear_param_store()
    for step in range(1, n_steps + 1):
        loss = svi.step(**model_kwargs)
        if (step % 250 == 0) or (step == 1) & verbose:
            print("[iteration %04d] loss: %.4f" % (step, loss))

    return guide


# https://stephens999.github.io/fiveMinuteStats/shiny_normal_example.html
def analytical_linear_gaussian_cut_posterior(data):
    w = data["w"]
    z = data["z"]
    post_sd_mod_one = math.sqrt((1 + NUM_SAMPS_MODULE_ONE / SIGMA_ONE**2) ** (-1))
    pr_eta_cut = dist.Normal(
        1
        / SIGMA_ONE**2
        * data["w"].sum()
        / (NUM_SAMPS_MODULE_ONE * 1 / SIGMA_ONE**2 + 1),
        scale=post_sd_mod_one,
    )
    post_mean_mod_two = lambda eta: (
        (data["z"] - ETA_COEF * eta).sum() / SIGMA_TWO**2
    ) * (1 / (1 + NUM_SAMPS_MODULE_TWO / SIGMA_TWO**2))
    post_sd_mod_two = math.sqrt((1 + NUM_SAMPS_MODULE_TWO / SIGMA_TWO**2) ** (-1))

    pr_theta_cut_cond_eta = lambda eta: dist.Normal(
        post_mean_mod_two(eta), scale=post_sd_mod_two
    )
    return pr_eta_cut, pr_theta_cut_cond_eta


def test_linear_gaussian_inference():
    data = observation_model(TRUE_ETA, TRUE_THETA)

    conditioned_model = pyro.condition(linear_gaussian_model, data=data)
    module_one_vars = ["eta", "w"]
    cut_model = make_cut_index_model(conditioned_model, module_one_vars)

    guide_cut = run_svi_inference(cut_model, n_steps=1500)
    guide_vanilla_post = run_svi_inference(conditioned_model, n_steps=1500)
    guide_only_w = run_svi_inference(
        pyro.poutine.block(conditioned_model, hide=["z", "theta"]), n_steps=1500
    )

    import pdb

    # eta estimates
    guide_cut_eta = guide_cut.median()["eta"].squeeze()[0]
    guide_approx_post_cut_eta = guide_only_w.median()["eta"]
    guide_vanilla_post_eta = guide_vanilla_post.median()["eta"]

    # theta estimates
    guide_cut_theta = guide_cut.median()["theta"].squeeze()[1]
    guide_vanilla_post_theta = guide_vanilla_post.median()["theta"]

    pr_eta_cut, pr_theta_cut_cond_eta = analytical_linear_gaussian_cut_posterior(data)

    cut_eta_samps = pr_eta_cut.sample((10000,))
    cut_theta_samps = []
    for eta_samp in cut_eta_samps:
        cut_theta_samps.append(pr_theta_cut_cond_eta(eta_samp)())
    cut_theta_samps = torch.tensor(cut_theta_samps).mean()

    # Formula here (assume mu_0=0 and tau_0=1 in the formula below):
    # https://stephens999.github.io/fiveMinuteStats/shiny_normal_example.html
    cut_theta_mean = cut_theta_samps.mean()

    pdb.set_trace()

    print("Cut eta mean: ", pr_eta_cut.loc.item())
    print("Cut eta estimate: ", guide_cut_eta.item())
    print("Posterior eta estimate: ", guide_vanilla_post_eta.item())

    print("Cut theta mean: ", cut_theta_mean.item())
    print("Cut theta estimate: ", guide_cut_theta.item())
    print("Posterior theta estimate: ", guide_vanilla_post_theta.item())
