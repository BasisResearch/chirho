import math
import pyro
import pyro.distributions as dist
import pyro.infer.reparam
import torch
import pyro.poutine as poutine

from causal_pyro.indexed.handlers import IndexPlatesMessenger
from causal_pyro.observational.handlers import IndexCutModule, cut


# Observed data assumed in the closed-form posterior expressions
BERN_DATA = {"z": torch.tensor(1.0), "w": torch.tensor(1.0)}


@pyro.infer.config_enumerate
def bern_model():
    eta = pyro.sample("eta", dist.Bernoulli(0.5))
    w = pyro.sample("w", dist.Bernoulli(torch.where(eta == 1, 0.8, 0.2)))
    theta = pyro.sample("theta", dist.Bernoulli(0.5))
    z = pyro.sample(
        "z", dist.Bernoulli(torch.where((eta == 1) & (theta == 1), 0.8, 0.2))
    )


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


def linear_gaussian_model(N1=10, N2=10):
    eta = pyro.sample("eta", dist.Normal(0, 1))
    theta = pyro.sample("theta", dist.Normal(0, 1))
    with pyro.plate("module_one_plate", N1):
        w = pyro.sample("w", dist.Normal(eta, 1))
    with pyro.plate("module_two_plate", N2):
        z = pyro.sample("z", dist.Normal(eta + theta, 1))
    return {"w": w, "z": z}


def linear_gaussian_cut_posterior(data):
    w = data["w"]
    z = data["z"]
    pr_eta_cut = dist.Normal(data["w"].sum() / (N1 + 1), scale=1 / torch.sqrt(N1 + 1))
    pr_theta_cut_cond_eta = lambda eta: dist.Normal(
        (data["z"] - eta).sum() / (N2 + 1), scale=1 / torch.sqrt(N2 + 1)
    )
    return pr_eta_cut, pr_theta_cut_cond_eta


def module_raises_assertion_error():
    conditioned_model = pyro.condition(
        model, data={"z": torch.tensor(1.0), "w": torch.tensor(1.0)}
    )
    module_one_vars = ["eta", "w"]
    module_one, module_two = cut(conditioned_model, vars=module_one_vars)
    try:
        module_two_post = pyro.infer.infer_discrete(module_two, first_available_dim=-2)
        assert (
            False
        ), "AssertionError should have been raises since module_one is not conditioned on"
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
        z = conditioned_model()


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


def compute_exact_elbo_discrete(model, psi):
    psi_one, psi_two, psi_three = psi

    # Compute probability of each configuration
    q_00 = (1 - psi_one) * (1 - psi_two)
    q_01 = (1 - psi_one) * psi_two
    q_10 = psi_one * (1 - psi_three)
    q_11 = psi_one * psi_three
    elbo = torch.tensor(0.0)

    # Add entropy term -E[log q(eta, theta)]
    elbo -= (
        q_00 * torch.log(q_00)
        + q_01 * torch.log(q_01)
        + q_10 * torch.log(q_10)
        + q_11 * torch.log(q_11)
    )

    # Add E[log p(eta, theta, w=1, z=1)] term
    tr00 = {"eta": torch.tensor(0.0), "theta": torch.tensor(0.0)}
    tr01 = {"eta": torch.tensor(0.0), "theta": torch.tensor(1.0)}
    tr10 = {"eta": torch.tensor(1.0), "theta": torch.tensor(0.0)}
    tr11 = {"eta": torch.tensor(1.0), "theta": torch.tensor(1.0)}
    log_p_00 = (
        poutine.trace(poutine.condition(model, data=tr00)).get_trace().log_prob_sum()
    )
    log_p_01 = (
        poutine.trace(poutine.condition(model, data=tr01)).get_trace().log_prob_sum()
    )
    log_p_10 = (
        poutine.trace(poutine.condition(model, data=tr10)).get_trace().log_prob_sum()
    )
    log_p_11 = (
        poutine.trace(poutine.condition(model, data=tr11)).get_trace().log_prob_sum()
    )
    elbo += q_00 * log_p_00 + q_01 * log_p_01 + q_10 * log_p_10 + q_11 * log_p_11

    return elbo


def test_index_module_discrete():
    # psi_one = p(eta = 1)
    # psi_two = p(theta = 1 | eta = 0)
    # psi_three = p(theta = 1 | eta = 1)
    module_one_vars = ["eta", "w"]
    conditioned_model = pyro.condition(bern_model, data=BERN_DATA)
    with IndexPlatesMessenger():
        with IndexCutModule(module_one_vars):
            psi_one_grid = torch.arange(0, 1, 0.1)
            psi_two_grid = torch.arange(0, 1, 0.1)
            psi_three_grid = torch.arange(0, 1, 0.1)
            best_config = None
            best_elbo = -float("inf")
            for psi_one in psi_one_grid:
                for psi_two in psi_two_grid:
                    for psi_three in psi_three_grid:
                        # Compute ELBO loss exactly
                        elbo = compute_exact_elbo_discrete(
                            conditioned_model, (psi_one, psi_two, psi_three)
                        )
                        if elbo > best_elbo:
                            best_elbo = elbo
                            best_config = (psi_one, psi_two, psi_three)

    assert torch.abs(best_config[0] - 0.8).item() < 0.025
    assert torch.abs(best_config[1] - 0.5).item() < 0.025
    assert torch.abs(best_config[2] - 0.8).item() < 0.025
