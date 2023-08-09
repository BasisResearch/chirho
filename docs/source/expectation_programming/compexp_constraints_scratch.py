import pyro
import torch
from torch import tensor as tt
import torch
from pyro.infer.autoguide import AutoGuide

from toy_tabi_problem import (
        cost as cost_ttp,
        model as model_ttp,
        q_optimal_normal_guide_mean_var,
        MODEL_DIST as MODEL_TTP_DIST
    )
import pyro.distributions as dist
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import chirho.contrib.compexp as compexp

import old_ep_demo_scratch as stor

pyro.settings.set(module_local_params=True)


def scost(d, x, c):
    return 1e1*cost_ttp(d=d, c=c, x=x)


def risk(*args, **kwargs):
    # Just the positive component of the toy cost function
    return torch.relu(scost(*args, **kwargs))


def cost(*args, **kwargs):
    # The negative part of the toy cost function.
    return -torch.relu(-scost(*args, **kwargs))


def main():
    dps = torch.nn.Parameter(tt([0.5]))
    c = tt(1.0)

    # A constrained objective, converted into an unconstrained objective.
    #  The objective is the negative of the cost function.
    exp_risk = compexp.E(
        f=lambda s: risk(d=dps[0], c=c, **s),
        name='exp_risk'
    )

    exp_cost = compexp.E(
        f=lambda s: cost(d=dps[0], c=c, **s),
        name='exp_cost'
    )

    # We can represent the risk-mean-constrained optimization as follows, where the risk quantity activates only
    #  when it exceeds the risk mean threshold.
    tau = compexp.C(tt(2.))
    risk_mean_thresh = compexp.C(tt(0.5))

    risk_mean_constrained_obj = exp_cost + tau * (exp_risk - risk_mean_thresh).relu()

    # Now, take the gradient and split expectation atoms into TABI components.
    risk_mean_constrained_obj_grad = risk_mean_constrained_obj.grad(params=dps, split_atoms=True)
    with compexp.MonteCarloExpectationHandler(num_samples=1):
        print(risk_mean_constrained_obj_grad(model_ttp))

    # Now, define the CVaR constraint. For this, we need to add an auxiliary parameter to jointly optimize.
    dps = torch.nn.Parameter((tt([0.5, 1.0])))  # [d, gamma]

    # Taking the expectation of the largest 20% of the risk values.
    alpha = tt(0.2)
    alpha_1o1ma_c = compexp.C(1. / (1. - alpha))

    # We need that expectation to be below...
    risk_cvar_thresh = tt(0.1)

    # HACK Push gamma inside the expectation, because we don't currently support gradients wrt parameters
    #  outside the expectation. This means we have to unscale it by the alpha_1o1ma_c factor.
    exp_grisk = compexp.E(
        f=lambda s: (1. - alpha) * dps[1] + torch.relu(risk(d=dps[0], c=c, **s) - dps[1] - risk_cvar_thresh),
        name='exp_grisk'
    )

    cvar_risk_constrained_obj = exp_cost + tau * (alpha_1o1ma_c * exp_grisk).relu()

    # Now, take the gradient and split expectation atoms into TABI components.
    cvar_risk_constrained_obj_grad = cvar_risk_constrained_obj.grad(params=dps, split_atoms=True)
    with compexp.MonteCarloExpectationHandler(num_samples=100):
        print(cvar_risk_constrained_obj_grad(model_ttp))

    # Before proceeding with the IS guide creation, we need to split the exp_grisk expectation into positive components.
    # TODO make split_atoms split stuff that doesn't get gradified.
    exp_grisk_atom = cvar_risk_constrained_obj_grad['exp_grisk']
    exp_grisk_tabi = exp_grisk_atom.split_into_positive_components()
    exp_grisk_atom.swap_self_for_other_child(
        exp_grisk_tabi
    )
    cvar_risk_constrained_obj_grad.recursively_refresh_parts()

    iseh = compexp.ImportanceSamplingExpectationHandler(num_samples=300)
    iseh.register_guides(
        model=model_ttp,
        ce=cvar_risk_constrained_obj_grad,
        # ce=exp_grisk.split_into_positive_components(),
        # ce=exp_cost.split_into_positive_components(),
        # ce=(exp_cost.split_into_positive_components() + exp_grisk.split_into_positive_components()),
        auto_guide=pyro.infer.autoguide.AutoNormal,
        auto_guide_kwargs=dict(init_scale=1.)
    )

    def plot_callback_(k, i):
        if i % 3000 == 0:
            figs = iseh.plot_guide_pseudo_likelihood(
                rv_name='x',
                guide_kde_kwargs=dict(bw_method=0.1, color='orange'),
                pseudo_density_plot_kwargs=dict(color='purple'),
                keys=[k] if k is not None else None
            )
            plt.show()
            for f in figs:
                plt.close(f)

    iseh.optimize_guides(
        lr=1e-3, n_steps=6001,
        callback=plot_callback_
    )

    # with iseh:
    #     print(exp_cost(model_ttp))

    return


if __name__ == "__main__":
    main()
