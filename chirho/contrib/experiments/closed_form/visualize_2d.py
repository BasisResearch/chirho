import torch
import chirho.contrib.experiments.closed_form as cfe
from torch import tensor as tnsr
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import numpy as np
from typing import Tuple
from chirho.contrib.compexp.typedecs import KWType


def _build_snis_opt_proposal_density_grad(
        problem: cfe.CostRiskProblem,
        theta: torch.Tensor,
        pi: int  # parameter index
):
    """
    Reconstruct the pseudo-density the SNIS guides are targeting.
    """

    targetf = cfe.build_opt_snis_grad_proposal_f(problem, theta, pi)

    def qstar_snis(s: KWType):
        pdensity = multivariate_normal(mean=np.zeros(problem.n), cov=problem.Sigma).logpdf(s['z'])
        fval = targetf(s)
        return torch.exp(torch.tensor(pdensity) + torch.log(1e-25 + fval))

    return qstar_snis


def _build_snis_opt_proposal_density_nograd(
        problem: cfe.CostRiskProblem,
        theta: torch.Tensor
):
    targetf = cfe.build_opt_snis_proposal_f(problem, theta)

    # TODO deduplicate code from _build_snis_opt_proposal_density_grad
    def qstar_snis(s: KWType):
        pdensity = multivariate_normal(mean=np.zeros(problem.n), cov=problem.Sigma).logpdf(s['z'])
        fval = targetf(s)
        return torch.exp(torch.tensor(pdensity) + torch.log(1e-25 + fval))

    return qstar_snis


def plot_f_contours_on_ax(ax, f, xlim, ylim, n=40):
    x = np.linspace(-xlim, xlim, n)
    y = np.linspace(-ylim, ylim, n)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    # TODO vectorize
    for i in range(n):
        for j in range(n):
            t = torch.tensor(np.array([X[i, j], Y[i, j]])).double()
            # Note Z is meshgrid syntax that differs from z, our syntax for rv.
            Z[i, j] = f({'z': t}).detach().numpy()
    # A filled contour.
    ax.contourf(X, Y, Z, cmap='Blues', levels=20)
    return ax


def plot_mvn_contours_on_ax(ax, loc, scale_tril, n=40, **kwargs):
    cov = scale_tril @ scale_tril.T
    # Get xlim and ylim from diagonal of cov.
    cxlim, cylim = np.sqrt(np.diag(cov)) * 4
    x = np.linspace(-cxlim, cxlim, n) + loc[0]
    y = np.linspace(-cylim, cylim, n) + loc[1]
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    Z = multivariate_normal(mean=loc, cov=cov).pdf(pos)

    ax.contour(X, Y, Z, **kwargs)


# TODO this doesn't actually animate right now.
def animate_guides_snis_grad(
        problem: cfe.CostRiskProblem,
        theta,
        guide_loc,
        guide_scale_tril,
        pi: int # parameter index
):
    # TODO unfix
    RES = 100
    XLIM = 4
    YLIM = 4

    theta = torch.tensor(theta).double().requires_grad_()

    qstar = _build_snis_opt_proposal_density_grad(problem, theta, pi)

    # with large dpi
    fig, ax = plt.subplots(dpi=300)
    ax.set_title(f"pi: {pi}")
    # TODO can help resolution by targeting the relevant bits with the xlim ylim
    plot_f_contours_on_ax(ax, qstar, XLIM, YLIM, n=RES)

    # With small linewidth
    plot_mvn_contours_on_ax(
        ax, guide_loc, guide_scale_tril, cmap='Greens', n=RES, linewidths=0.5
    )

    return fig


def animate_guides_snis_grad_from_guide_track(
        problem: cfe.CostRiskProblem, guide_track: cfe.GuideTrack, trajis=(0, -1)):
    for pi in [0, 1]:  # parameter index:
        for traji in trajis:
            theta = torch.tensor(guide_track.thetas[traji]).double().requires_grad_()
            guide_loc = list(guide_track.guide_means.values())[pi][traji]
            guide_scale_tril = list(guide_track.guide_scale_trils.values())[pi][traji]

            animate_guides_snis_grad(problem, theta, guide_loc, guide_scale_tril, pi)


def animate_guides_snis_nograd(
        problem: cfe.CostRiskProblem,
        theta,
        guide_loc,
        guide_scale_tril
):
    # TODO deduplicate code from animate_guides_snis_grad

    # TODO unfix
    RES = 100
    XLIM = 6
    YLIM = 6

    theta = torch.tensor(theta).double().requires_grad_()

    qstar = _build_snis_opt_proposal_density_nograd(problem, theta)

    # with large dpi
    fig, ax = plt.subplots(dpi=300)
    # make x and y same scale.
    ax.set_aspect('equal')
    # TODO can help resolution by targeting the relevant bits with the xlim ylim
    plot_f_contours_on_ax(ax, qstar, XLIM, YLIM, n=RES)

    # With small linewidth
    plot_mvn_contours_on_ax(
        ax, guide_loc, guide_scale_tril, cmap='Greens', n=RES, linewidths=0.5
    )

    return fig


def animate_guides_snis_nograd_from_guide_track(
        problem: cfe.CostRiskProblem, guide_track: cfe.GuideTrack, trajis=(0, -1)):
    # TODO deduplicate code from animate_guides_snis_grad_from_guide_track
    for traji in trajis:
        theta = torch.tensor(guide_track.thetas[traji]).double().requires_grad_()
        guide_loc = list(guide_track.guide_means.values())[0][traji]
        guide_scale_tril = list(guide_track.guide_scale_trils.values())[0][traji]

        fig = animate_guides_snis_nograd(problem, theta, guide_loc, guide_scale_tril)

        # TODO hack saving manually here.
        # fig.savefig(f"/Users/azane/Downloads/snistechnicallyunbiased2/{traji}_snis_nograd.png")
        # plt.close(fig)
