import math

import pyro
import pyro.distributions as dist
import pyro.infer.reparam
import torch
from pyro.infer.autoguide import AutoMultivariateNormal

from chirho.indexed.handlers import IndexPlatesMessenger
from chirho.observational.handlers.cut import (
    CutComplementModule,
    CutModule,
    SingleStageCut,
)

pyro.settings.set(module_local_params=True)

# Observed data assumed in the closed-form posterior expressions
BERN_DATA = {"z": torch.tensor(1.0), "w": torch.tensor(1.0)}

# Simulated data configuration for linear Gaussian model
TRUE_ETA = torch.tensor(2.0)
TRUE_THETA = torch.tensor(0.0)
NUM_SAMPS_MODULE_ONE = 2
NUM_SAMPS_MODULE_TWO = 1000
SIGMA_ONE = 3.0
SIGMA_TWO = 0.1
ETA_COEF = 1.0


@pyro.infer.config_enumerate
def bern_model():
    eta = pyro.sample("eta", dist.Bernoulli(0.5))
    pyro.sample("w", dist.Bernoulli(torch.where(eta == 1, 0.8, 0.2)))
    theta = pyro.sample("theta", dist.Bernoulli(0.5))
    pyro.sample("z", dist.Bernoulli(torch.where((eta == 1) & (theta == 1), 0.8, 0.2)))


def bern_posterior():
    p_eta = dist.Bernoulli(10 / 11)  # pr(eta | w=1, z=1)

    def p_theta_cond_eta(eta):  # pr(theta | eta, z=1)
        if eta == torch.tensor(1.0):
            return dist.Bernoulli(0.8)
        elif eta == torch.tensor(0.0):
            return dist.Bernoulli(0.5)
        else:
            raise ValueError("eta must be 0 or 1")

    return p_eta, p_theta_cond_eta


def bern_cut_posterior():
    # pcut(eta, theta | w=1, z=1) = pr(eta | w=1) x pr(theta | eta, z=1),
    # where pr() denotes the posterior distribution
    p_eta = dist.Bernoulli(0.8)  # pr(eta | w=1)

    def p_theta_cond_eta(eta):  # pr(theta | eta, z=1)
        if eta == torch.tensor(1.0):
            return dist.Bernoulli(0.8)
        elif eta == torch.tensor(0.0):
            return dist.Bernoulli(0.5)
        else:
            raise ValueError("eta must be 0 or 1")

    return p_eta, p_theta_cond_eta


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


def test_cut_module_raises_assertion_error():
    conditioned_model = pyro.condition(bern_model, data=BERN_DATA)
    module_one_vars = ["eta", "w"]
    module_two = CutComplementModule(module_one_vars)(conditioned_model)
    try:
        pyro.infer.infer_discrete(module_two, first_available_dim=-2)
        assert (
            False
        ), "AssertionError should have been raised since module_one is not conditioned on"
    except AssertionError:
        print("AssertionError raised as expected")


def test_plate_cut_module_runs():
    # Runs for discrete model
    conditioned_model = pyro.condition(bern_model, data=BERN_DATA)
    module_one_vars = ["eta", "w"]
    with IndexPlatesMessenger(), SingleStageCut(module_one_vars):
        conditioned_model()

    # Runs for continous model
    data = linear_gaussian_model()
    conditioned_model = pyro.condition(linear_gaussian_model, data=data)
    module_one_vars = ["eta", "w"]
    with IndexPlatesMessenger(), SingleStageCut(module_one_vars):
        conditioned_model()


def test_cut_module_discrete():
    conditioned_model = pyro.condition(bern_model, data=BERN_DATA)
    module_one_vars = ["eta", "w"]
    module_one = CutModule(module_one_vars)(conditioned_model)
    module_two = CutComplementModule(module_one_vars)(conditioned_model)
    module_one_post = pyro.infer.infer_discrete(module_one, first_available_dim=-2)
    module_one_samples = []

    # Module 1 check
    for _ in range(1000):
        with pyro.poutine.trace() as tr:
            module_one_post()
        module_one_samples.append(tr.trace.nodes["eta"]["value"].item())

    rand_error_tol = 3 * 0.5 / math.sqrt(1000)
    assert (
        torch.abs(torch.tensor(module_one_samples).mean() - 0.8).item() < rand_error_tol
    )

    # Module 2 check (eta = 1)
    module_two_cond = pyro.condition(module_two, data={"eta": torch.tensor(1.0)})
    module_two_post = pyro.infer.infer_discrete(module_two_cond, first_available_dim=-2)
    module_two_samples = []
    for _ in range(1000):
        with pyro.poutine.trace() as tr:
            module_two_post()
        module_two_samples.append(tr.trace.nodes["theta"]["value"].item())

    assert (
        torch.abs(torch.tensor(module_two_samples).mean() - 0.8).item() < rand_error_tol
    )

    # Module 2 check (eta = 0)
    module_two_cond = pyro.condition(module_two, data={"eta": torch.tensor(0.0)})
    module_two_post = pyro.infer.infer_discrete(module_two_cond, first_available_dim=-2)
    module_two_samples = []
    for _ in range(1000):
        with pyro.poutine.trace() as tr:
            module_two_post()
        module_two_samples.append(tr.trace.nodes["theta"]["value"].item())

    assert (
        torch.abs(torch.tensor(module_two_samples).mean() - 0.5).item() < rand_error_tol
    )


def test_correctly_duplicates_module_one_vars():
    def dummy_model():
        with IndexPlatesMessenger(), SingleStageCut(["x"]):
            return pyro.sample("x", dist.Normal(0.0, 1.0))

    x = dummy_model()
    assert x[0] == x[1]

    # Check composability with replay
    def dummy_guide():
        pyro.sample("x", dist.Normal(0.0, 1.0).expand([2, 1, 1, 1, 1]))

    with pyro.poutine.trace() as dummy_guide_tr:
        dummy_guide()

    dummy_guide_x = dummy_guide_tr.trace.nodes["x"]["value"][0]
    replayed_model_x = pyro.poutine.replay(dummy_model, dummy_guide_tr.trace)()
    assert replayed_model_x[0].squeeze() == dummy_guide_x[0].squeeze()
    assert replayed_model_x[1].squeeze() == dummy_guide_x[0].squeeze()

    with IndexPlatesMessenger(), SingleStageCut(["x", "x2"]):
        x2 = pyro.sample("x2", dist.Normal(0.0, 1.0))
        with pyro.poutine.trace() as tr:
            pyro.sample("x", dist.Normal(0.0, 1.0))

        assert x2[0] == x2[1]
        assert tr.trace.nodes["x"]["value"][0] == tr.trace.nodes["x"]["value"][1]


def make_cut_index_model(model, module_one_vars, cut=SingleStageCut, *args, **kwargs):
    def cut_index_model():
        with IndexPlatesMessenger(), cut(module_one_vars):
            model(*args, **kwargs)

    return cut_index_model


def run_svi_inference(model, n_steps=1000, verbose=True, lr=0.03, **model_kwargs):
    guide = AutoMultivariateNormal(model)
    elbo = pyro.infer.Trace_ELBO()(model, guide)
    # initialize parameters
    elbo(**model_kwargs)
    adam = torch.optim.Adam(elbo.parameters(), lr=lr)
    # Do gradient steps
    for step in range(1, n_steps + 1):
        adam.zero_grad()
        loss = elbo(**model_kwargs)
        loss.backward()
        adam.step()
        if (step % 250 == 0) or (step == 1) & verbose:
            print("[iteration %04d] loss: %.4f" % (step, loss))
    return guide


# See https://stephens999.github.io/fiveMinuteStats/shiny_normal_example.html
def analytical_linear_gaussian_cut_posterior(data):
    post_sd_mod_one = math.sqrt((1 + NUM_SAMPS_MODULE_ONE / SIGMA_ONE**2) ** (-1))
    pr_eta_cut = dist.Normal(
        1 / SIGMA_ONE**2 * data["w"].sum() / (1 + NUM_SAMPS_MODULE_ONE / SIGMA_ONE**2),
        scale=post_sd_mod_one,
    )
    post_mean_mod_two = lambda eta: (  # noqa
        (data["z"] - ETA_COEF * eta).sum() / SIGMA_TWO**2
    ) * (1 / (1 + NUM_SAMPS_MODULE_TWO / SIGMA_TWO**2))
    post_sd_mod_two = math.sqrt((1 + NUM_SAMPS_MODULE_TWO / SIGMA_TWO**2) ** (-1))

    pr_theta_cut_cond_eta = lambda eta: dist.Normal(  # noqa
        post_mean_mod_two(eta), scale=post_sd_mod_two
    )
    return pr_eta_cut, pr_theta_cut_cond_eta


def _test_linear_gaussian_inference():
    """
    This test compares the cut posterior to the true cut posterior from
    analytical calculations for the linear Gaussian case.
    This test is not run in our testing suite because
    the results can vary substantially due to non-deterministic behavior
    of SVI for inference. Nevertheless, this test prints out the differences
    between the approximate cut posterior using `SingleStageCut`
    and the true cut posterior.
    """
    data = observation_model(TRUE_ETA, TRUE_THETA)
    module_one_vars = ["eta", "w"]

    conditioned_model = pyro.condition(linear_gaussian_model, data=data)
    cut_model = make_cut_index_model(conditioned_model, module_one_vars)
    guide_cut = run_svi_inference(cut_model, n_steps=1500)

    # Estimates
    guide_cut_eta = guide_cut.median()["eta"].squeeze()[0]
    guide_cut_theta = guide_cut.median()["theta"].squeeze()[1]

    # True values of latents
    pr_eta_cut, pr_theta_cut_cond_eta = analytical_linear_gaussian_cut_posterior(data)

    # Don't have a closed form formula for the marginal over theta
    # Approximate by sampling
    eta_samps = pr_eta_cut.sample((1000,))
    theta_samps = []
    for eta_samp in eta_samps:
        theta_samps.append(pr_theta_cut_cond_eta(eta_samp)())
    theta_samps = torch.tensor(theta_samps).mean()

    true_cut_eta_mean = pr_eta_cut.loc
    true_cut_theta_mean = theta_samps.mean()

    # Formula here (assume mu_0=0 and tau_0=1 in the formula below):
    # https://stephens999.github.io/fiveMinuteStats/shiny_normal_example.html

    print("True Expected eta in Cut Posterior: ", true_cut_eta_mean.item())
    print("Approximate eta in Cut Posterior: ", guide_cut_eta.item())
    print("True Expected Theta in Cut Posterior: ", true_cut_theta_mean.item())
    print("Approximate Theta in Cut Posterior: ", guide_cut_theta.item())
