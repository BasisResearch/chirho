import math

import pyro
import pyro.distributions as dist
import pyro.infer.reparam
import torch

from chirho.indexed.handlers import IndexPlatesMessenger
from chirho.observational.handlers.soft_conditioning import IndexCutModule, cut

# Observed data assumed in the closed-form posterior expressions
BERN_DATA = {"z": torch.tensor(1.0), "w": torch.tensor(1.0)}


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


def linear_gaussian_model(N1=10, N2=10, sigma1=1.0, sigma2=1.0):
    eta = pyro.sample("eta", dist.Normal(0, 1))
    theta = pyro.sample("theta", dist.Normal(0, 1))
    with pyro.plate("module_one_plate", N1):
        w = pyro.sample("w", dist.Normal(eta, sigma1))
    with pyro.plate("module_two_plate", N2):
        z = pyro.sample("z", dist.Normal(eta + theta, sigma2))
    return {"w": w, "z": z}


def test_cut_module_raises_assertion_error():
    conditioned_model = pyro.condition(bern_model, data=BERN_DATA)
    module_one_vars = ["eta", "w"]
    module_one, module_two = cut(conditioned_model, vars=module_one_vars)
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
    with IndexPlatesMessenger(), IndexCutModule(module_one_vars):
        conditioned_model()

    # Runs for continous model
    data = linear_gaussian_model()
    conditioned_model = pyro.condition(linear_gaussian_model, data=data)
    module_one_vars = ["eta", "w"]
    with IndexPlatesMessenger(), IndexCutModule(module_one_vars):
        conditioned_model()


def test_cut_module_discrete():
    conditioned_model = pyro.condition(bern_model, data=BERN_DATA)
    module_one_vars = ["eta", "w"]
    module_one, module_two = cut(conditioned_model, vars=module_one_vars)
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
        with IndexPlatesMessenger(), IndexCutModule(["x"]):
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

    # Check composability with trace
    # TODO: ask Eli if the test below should fail based on trace implementation. If not, then
    # the lines below should be uncommented.
    # with pyro.poutine.trace() as tr:
    #     with IndexPlatesMessenger(), IndexCutModule(["x"]):
    #         pyro.sample("x", dist.Normal(0.0, 1.0))

    # assert tr.trace.nodes["x"]["value"][0] == tr.trace.nodes["x"]["value"][1]
