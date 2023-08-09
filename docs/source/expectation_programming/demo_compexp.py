# TODO Needs supporting stuff ported here. Also break out into notebook and tests.


import pyro
import torch
from torch import tensor as tt
from pyro.infer.autoguide import AutoGuide

from toy_tabi_problem import (
        cost,
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


def main():
    # noinspection PyPep8Naming
    D = tt(0.5)
    # noinspection PyPep8Naming
    C = tt(1.0)
    # noinspection PyPep8Naming
    GT = -1.1337

    # Make sure we match a ground truth value.
    toy_opt_guidep = dist.Normal(*q_optimal_normal_guide_mean_var(d=D, c=C, z=False))
    toy_opt_guiden = dist.Normal(*q_optimal_normal_guide_mean_var(d=D, c=C, z=True))

    # decomposed toy expectation
    # noinspection PyPep8Naming
    ttE: compexp.ComposedExpectation = compexp.E(
        f=lambda s: 1e1*cost(d=tt(0.5), c=tt(1.), **s), name='ttE'
    ).split_into_positive_components(
            # TODO bdt18dosjk
            pos_guide=lambda: OrderedDict(x=pyro.sample('x', toy_opt_guidep)),
            neg_guide=lambda: OrderedDict(x=pyro.sample('x', toy_opt_guiden)),
            den_guide=lambda: OrderedDict(x=pyro.sample('x', MODEL_TTP_DIST))
    )

    # Just to make sure everything still works after this is called.
    ttE.recursively_refresh_parts()

    # <Monte Carlo------------------------------------------------------>
    mcestimates = []
    for _ in range(1000):
        with compexp.MonteCarloExpectationHandler(num_samples=100):
            mcestimates.append(ttE(model_ttp))

    plt.suptitle("MC Estimate")
    plt.hist(mcestimates, bins=30)
    mcestimate = torch.mean(torch.stack(mcestimates))
    plt.axvline(x=mcestimate, color='r')
    plt.axvline(x=GT, color='black', linestyle='--')
    plt.suptitle(f"MC Estimate: {mcestimate}\n GT = {GT}")
    plt.show()
    # </Monte Carlo------------------------------------------------------>

    # <TABI------------------------------------------------------>
    # When using the decomposition above with per-atom guides and importance sampling, we get TABI. Because
    #  we've preset the optimal guides above, we will get an exact estimate.
    iseh = compexp.ImportanceSamplingExpectationHandler(num_samples=1)
    iseh.register_guides(ce=ttE, model=model_ttp, auto_guide=None, auto_guide_kwargs=None)
    with iseh:
        tabiestimate = ttE(model_ttp)
    print(f"TABI Estimate: {tabiestimate}",  f"GT = {GT}")
    assert torch.isclose(tabiestimate, tt(GT), atol=1e-4)
    # </TABI------------------------------------------------------>

    # <TABI Learned Guides------------------------------------------------------>

    ttE = compexp.E(
        f=lambda s: 1e1 * cost(d=tt(0.5), c=tt(1.), **s), name='ttE'
    ).split_into_positive_components()

    iseh2 = compexp.ImportanceSamplingExpectationHandler(num_samples=300)
    iseh2.register_guides(
        ce=ttE,
        model=model_ttp,
        auto_guide=pyro.infer.autoguide.AutoNormal,
        auto_guide_kwargs=dict(init_scale=2.))

    def plot_callback_(k, i):
        if i % 5000 == 0:
            figs = iseh2.plot_guide_pseudo_likelihood(
                rv_name='x',
                guide_kde_kwargs=dict(bw_method=0.1, color='orange'),
                pseudo_density_plot_kwargs=dict(color='purple'),
                keys=[k] if k is not None else None
            )
            plt.show()
            for f in figs:
                plt.close(f)

    iseh2.optimize_guides(
        lr=5e-3, n_steps=10001,
        callback=plot_callback_
    )

    tabi_learned_estimates = []
    for _ in range(100):
        with iseh2:
            tabi_learned_estimates.append(ttE(model_ttp))
    tabi_learned_estimates = torch.stack(tabi_learned_estimates).detach().numpy()

    plt.suptitle("TABI Gradient Estimate")
    plt.hist(tabi_learned_estimates, bins=30)
    plt.axvline(x=np.mean(tabi_learned_estimates), color='r')
    plt.axvline(x=GT, color='black', linestyle='--')
    plt.suptitle(f"TABI Gradient Estimate: {np.mean(tabi_learned_estimates)}\n GT = {GT}")
    plt.show()

    # </TABI Learned Guides------------------------------------------------------>

    return

    # <MonteCarlo Gradient 1d------------------------------------------------------>
    dps = torch.nn.Parameter(tt([0.5]))

    # noinspection PyPep8Naming
    tte1: compexp.ComposedExpectation = compexp.E(
        # f=lambda s: 1e1*cost(d=dps[0] * dps[1], c=tt(1.), **s), name='ttgradE'
        f=lambda s: 1e1*cost(d=dps[0], c=tt(1.), **s), name='tte1'
    ).split_into_positive_components()
    tte1['tte1_split_den'].requires_grad = False
    dtte1_ddparams = tte1.grad(params=dps)

    mc_grad_estimates = []
    for _ in range(200):
        with compexp.MonteCarloExpectationHandler(num_samples=100):
            mc_grad_estimates.append(dtte1_ddparams(model_ttp))
    mc_grad_estimates = torch.stack(mc_grad_estimates).detach().numpy().T[0]

    plt.suptitle("MC Gradient Estimate")
    plt.hist(mc_grad_estimates, bins=30)
    plt.axvline(x=np.mean(mc_grad_estimates), color='r')
    plt.axvline(x=2.358, color='black', linestyle='--')
    plt.suptitle(f"MC Gradient Estimate: {np.mean(mc_grad_estimates)}\n GT = {2.358}")
    plt.show()
    # </MonteCarlo Gradient 1d------------------------------------------------------>

    # <MonteCarlo Gradient 2d------------------------------------------------------>
    dps = torch.nn.Parameter(tt([0.25, 2.]))

    # noinspection PyPep8Naming
    tte2: compexp.ComposedExpectation = compexp.E(
        f=lambda s: 1e1*cost(d=dps[0] * dps[1], c=tt(1.), **s), name='ttgradE'
    ).split_into_positive_components()
    tte2['ttgradE_split_den'].requires_grad = False
    dtte2_ddparams = tte2.grad(params=dps)

    with compexp.MonteCarloExpectationHandler(num_samples=10):
        print(dtte2_ddparams(model_ttp), "GT unknown but it runs")
    # </MonteCarlo Gradient 2d------------------------------------------------------>

    # <Unlearned TABI Guides for Grads 1d------------------------------------------------------>
    dps = torch.nn.Parameter(tt([0.5]))

    tte_tabi_unfit = compexp.E(
        f=lambda s: 1e1*cost(d=dps[0], c=tt(1.), **s), name='ttgradE'
    )
    dtte_tabi_unfit_ddparams = tte_tabi_unfit.grad(params=dps, split_atoms=True)

    iseh2 = compexp.ImportanceSamplingExpectationHandler(num_samples=50)
    iseh2.register_guides(
        ce=dtte_tabi_unfit_ddparams,
        model=model_ttp,
        auto_guide=pyro.infer.autoguide.AutoNormal,
        auto_guide_kwargs=dict(init_scale=1.5))
    iseh2.plot_guide_pseudo_likelihood(
        rv_name='x',
        guide_kde_kwargs=dict(bw_method=0.1, color='orange'),
        pseudo_density_plot_kwargs=dict(color='purple')
    )
    plt.show()

    tabi_unlearned_grad_estimates = []
    for _ in range(50):
        with iseh2:
            tabi_unlearned_grad_estimates.append(dtte_tabi_unfit_ddparams(model_ttp))
    tabi_unlearned_grad_estimates = torch.stack(tabi_unlearned_grad_estimates).detach().numpy()

    plt.suptitle("TABI Gradient Estimate")
    plt.hist(tabi_unlearned_grad_estimates, bins=30)
    plt.axvline(x=np.mean(tabi_unlearned_grad_estimates), color='r')
    plt.axvline(x=2.358, color='black', linestyle='--')
    plt.suptitle(f"TABI Gradient Estimate: {np.mean(tabi_unlearned_grad_estimates)}\n GT = {2.358}")
    plt.show()
    # </Unlearned TABI Guides for Grads 1d------------------------------------------------------>

    # <Unlearned Good TABI Guides for Grads 1d------------------------------------------------------>
    dps = torch.nn.Parameter(tt([0.5]))

    tte3_unfit = compexp.E(
        f=lambda s: 1e1 * cost(d=dps[0], c=tt(1.), **s), name='ttgradE'
    )
    tte3grad_tabi_unfit = tte3_unfit.grad(params=dps, split_atoms=True)

    iseh2 = compexp.ImportanceSamplingExpectationHandler(num_samples=300)
    iseh2.register_guides(
        ce=tte3grad_tabi_unfit,
        model=model_ttp,
        auto_guide=pyro.infer.autoguide.AutoNormal,
        auto_guide_kwargs=dict(init_scale=1.))

    # Make guides that are roughly in the correct positions.
    pos_iloc_grad, pos_istd_grad = q_optimal_normal_guide_mean_var(d=tt(0.5), c=tt(1.0), z=False)
    neg_iloc_grad, neg_istd_grad = q_optimal_normal_guide_mean_var(d=tt(0.5), c=tt(1.0), z=True)
    pos_guide_grad = stor.MultiModalGuide1D(
        num_components=2,
        init_loc=[pos_iloc_grad.item(), neg_iloc_grad.item()],
        init_scale=[pos_istd_grad.item(), neg_istd_grad.item()]
    )
    neg_guide_grad = stor.MultiModalGuide1D(
        num_components=2,
        init_loc=[pos_iloc_grad.item(), neg_iloc_grad.item()],
        init_scale=[pos_istd_grad.item(), neg_istd_grad.item()]
    )

    # TODO is this a good pattern for specifying guides? If so should add some more runtime checking to make sure
    #  e.g. names exist and that the guides spit out the right stochastics.
    iseh2.guides['dttgradE_dp0_split_pos'] = pos_guide_grad
    iseh2.guides['dttgradE_dp0_split_neg'] = neg_guide_grad
    # Leave 'dttgradE_dd0_split_den' as default AutoNormal.

    iseh2.plot_guide_pseudo_likelihood(
        rv_name='x',
        guide_kde_kwargs=dict(bw_method=0.1, color='orange'),
        pseudo_density_plot_kwargs=dict(color='purple')
    )
    plt.show()

    tabi_unlearned_good_grad_estimates = []
    for _ in range(100):
        with iseh2:
            tabi_unlearned_good_grad_estimates.append(tte3grad_tabi_unfit(model_ttp))
    tabi_unlearned_good_grad_estimates = torch.stack(tabi_unlearned_good_grad_estimates).detach().numpy()

    plt.suptitle("TABI Unlearned Good Gradient Estimate")
    plt.hist(tabi_unlearned_good_grad_estimates, bins=30)
    plt.axvline(x=np.mean(tabi_unlearned_good_grad_estimates), color='r')
    plt.axvline(x=2.358, color='black', linestyle='--')
    plt.suptitle(f"TABI Unlearned Good Gradient Estimate: {np.mean(tabi_unlearned_good_grad_estimates)}\n GT = {2.358}")
    plt.show()
    # </Unlearned Good TABI Guides for Grads 1d------------------------------------------------------>

    # <Learned TABI Guides for Grads 1d------------------------------------------------------>
    dps = torch.nn.Parameter(tt([0.5]))

    tte4_unfit = compexp.E(
        f=lambda s: 1e1 * cost(d=dps[0], c=tt(1.), **s), name='ttgradE'
    )  # .split_into_positive_components()
    tte4grad_tabi_unfit = tte4_unfit.grad(params=dps, split_atoms=True)

    iseh2 = compexp.ImportanceSamplingExpectationHandler(num_samples=50)
    iseh2.register_guides(
        ce=tte4grad_tabi_unfit,
        model=model_ttp,
        auto_guide=pyro.infer.autoguide.AutoNormal,
        auto_guide_kwargs=dict(init_scale=2.))

    # Start the guides roughly in the correct positions.
    pos_iloc_grad, pos_istd_grad = q_optimal_normal_guide_mean_var(d=tt(0.5), c=tt(1.0), z=False)
    neg_iloc_grad, neg_istd_grad = q_optimal_normal_guide_mean_var(d=tt(0.5), c=tt(1.0), z=True)
    pos_guide_grad = stor.MultiModalGuide1D(
        num_components=2,
        init_loc=[pos_iloc_grad.item(), neg_iloc_grad.item()],
        init_scale=[pos_istd_grad.item() * 2., neg_istd_grad.item() * 2.],
        studentt=True
    )
    neg_guide_grad = stor.MultiModalGuide1D(
        num_components=2,
        init_loc=[pos_iloc_grad.item(), neg_iloc_grad.item()],
        init_scale=[pos_istd_grad.item() * 2., neg_istd_grad.item() * 2.],
        studentt=True
    )

    # TODO is this a good pattern for specifying guides? If so should add some more runtime checking to make sure
    #  e.g. names exist and that the guides spit out the right stochastics.
    iseh2.guides['dttgradE_dp0_split_pos'] = pos_guide_grad
    iseh2.guides['dttgradE_dp0_split_neg'] = neg_guide_grad
    # Leave 'dttgradE_dp0_split_den' as default AutoNormal.

    NSTEPS = 10000

    def plot_callback_(k, i):
        if i % NSTEPS == 0:
            figs = iseh2.plot_guide_pseudo_likelihood(
                rv_name='x',
                guide_kde_kwargs=dict(bw_method=0.1, color='orange'),
                pseudo_density_plot_kwargs=dict(color='purple'),
                keys=[k] if k is not None else None
            )
            plt.show()
            for f in figs:
                plt.close(f)

    # plot_callback_(None, 0)

    iseh2.optimize_guides(
        lr=1e-3, n_steps=NSTEPS + 1,
        adjust_grads_=stor.abort_guide_grads_,
        callback=plot_callback_
    )

    tabi_learned_grad_estimates = []
    for _ in range(100):
        with iseh2:
            tabi_learned_grad_estimates.append(tte4grad_tabi_unfit(model_ttp))
    tabi_learned_grad_estimates = torch.stack(tabi_learned_grad_estimates).detach().numpy()

    plt.suptitle("TABI Learned Gradient Estimate")
    plt.hist(tabi_learned_grad_estimates, bins=30)
    plt.axvline(x=np.mean(tabi_learned_grad_estimates), color='r')
    plt.axvline(x=2.358, color='black', linestyle='--')
    plt.suptitle(f"TABI Learned Gradient Estimate: {np.mean(tabi_learned_grad_estimates)}\n GT = {2.358}")
    plt.show()
    # </Learned TABI Guides for Grads 1d------------------------------------------------------>


if __name__ == "__main__":
    main()
