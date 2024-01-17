from robust_fd.squared_normal_density import (
    NormalKernel,
    PerturbableNormal,
    ExpectedDensityQuadFunctional,
    ExpectedDensityMCFunctional
)
from chirho.robust.handlers.fd_model import (
    fd_influence_fn,
    ModelWithMarginalDensity,
    FDModelFunctionalDensity
)
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn, norm
from itertools import product
from typing import List, Dict, Tuple, Optional
from scipy.stats._multivariate import _squeeze_output


# EPS = [0.01, 0.001]
# LAMBDA = [0.01, 0.001]
EPS = [0.01]
LAMBDA = [0.01]
NDIM = [1, 2]
NDATASETS = 3
NGUESS = 50
NEIF = 50


def analytic_eif(model: FDModelFunctionalDensity, points, funcval=None):
    if funcval is None:
        funcval = model.functional()
    density = model.density(points, points)

    return 2. * (density - funcval)


class MultivariateSkewnormFDModel(ModelWithMarginalDensity):

    # From https://gregorygundersen.com/blog/2020/12/29/multivariate-skew-normal/:

    def __init__(self, shape, cov, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = len(shape)
        self.shape = np.asarray(shape)
        self.mean = np.zeros(self.dim)
        self.cov = np.eye(self.dim) if cov is None else np.asarray(cov)

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def density(self, x):
        return self.pdf(x)

    def logpdf(self, x):
        x = mvn._process_quantiles(x, self.dim)
        pdf = mvn(self.mean, self.cov).logpdf(x)
        cdf = norm(0, 1).logcdf(np.dot(x, self.shape))
        return _squeeze_output(np.log(2) + pdf + cdf)

    def rvs_fast(self, size=1):
        aCa = self.shape @ self.cov @ self.shape
        delta = (1 / np.sqrt(1 + aCa)) * self.cov @ self.shape
        cov_star = np.block([[np.ones(1), delta],
                             [delta[:, None], self.cov]])
        x = mvn(np.zeros(self.dim + 1), cov_star).rvs(size)
        x0, x1 = x[:, 0], x[:, 1:]
        inds = x0 <= 0
        x1[inds] = -1 * x1[inds]
        return x1

    def forward(self, *args, **kwargs):
        # TODO whatever the pyro version of this is? If there is one just get rid of this class.
        raise NotImplementedError()


class PerturbableSkewNormal(FDModelFunctionalDensity):
    def __init__(self, shape, cov, *args, **kwargs):
        default_kernel_point = dict(x=np.zeros(len(shape)))
        super().__init__(*args, default_kernel_point=default_kernel_point, **kwargs)

        self.model = MultivariateSkewnormFDModel(shape, cov)

        self.shape = shape
        self.cov = cov


class ExpectedSkewNormalDensityQuadFunctional(
    NormalKernel,
    PerturbableSkewNormal,
    ExpectedDensityQuadFunctional,
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ExpectedNormalDensityQuadFunctional(
    NormalKernel,
    PerturbableNormal,
    ExpectedDensityQuadFunctional,
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ExpectedNormalDensityMCFunctional(
    NormalKernel,
    PerturbableNormal,
    ExpectedDensityMCFunctional,
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def skew_unit_norm_dataset_generator(ndim: int, num_datasets: int, nguess: int, neif: int):

    for _ in range(num_datasets):
        cov = np.eye(ndim)
        shape = np.random.normal(size=ndim, scale=3.)

        datadist = ExpectedSkewNormalDensityQuadFunctional(shape, cov)
        dataset = datadist.model.rvs_fast(nguess + neif)
        guess_dataset, correction_dataset = dataset[:nguess], dataset[nguess:]
        assert len(guess_dataset) == nguess
        assert len(correction_dataset) == neif
        yield guess_dataset, correction_dataset, datadist


# def main2d():
#     for guess_dataset, correction_dataset, oracle in skew_unit_norm_dataset_generator(2, NDATASETS, NGUESS):
#
#         print(f"Oracle: {oracle}")
#
#         plt.figure()
#         plt.scatter(dataset[:, 0], dataset[:, 1], alpha=0.5, s=0.2)
#         plt.show()


def main(ndim=1, plot_densities=True, plot_corrections=True, plot_fd_ana_diag=True):

    fd_cors_quad = dict()  # type: Dict[Tuple[float, float], List[float]]
    fd_cors_mc = dict()  # type: Dict[Tuple[float, float], List[float]]
    ana_cors_quad = list()
    ana_cors_mc = list()

    for guess_dataset, correction_dataset, datadist in skew_unit_norm_dataset_generator(ndim, NDATASETS, NGUESS, NEIF):

        oracle_fval = datadist.functional()
        print(f"Oracle: {oracle_fval}")

        if ndim == 1 and plot_densities:
            # Plot a density estimate.
            f1 = plt.figure()
            # Using correction dataset to see where things fall on the influence function curve.
            plt.hist(correction_dataset, bins=100, density=True, alpha=0.5)
            xx = np.linspace(correction_dataset.min(), correction_dataset.max(), 100).reshape(-1, ndim)
            yy = datadist.density(dict(x=xx), dict(x=xx))
            plt.plot(xx, yy)

        # Train a model on the first nguess points in the dataset. This is a misspecified
        #  model in that it is a normal with no skew.
        mean = torch.tensor(np.atleast_1d(np.mean(guess_dataset, axis=0))).float()
        cov = torch.tensor(np.atleast_2d(np.cov(guess_dataset, rowvar=False))).float()

        # Compute the functionals of the unperturbed models.
        fd_quad = ExpectedNormalDensityQuadFunctional(
            default_kernel_point=dict(x=torch.zeros(len(mean))),
            mean=mean,
            cov=cov
        )
        quad_guess = fd_quad.functional()
        print(f"Quad: {quad_guess}")
        fd_mc = ExpectedNormalDensityMCFunctional(
            default_kernel_point=dict(x=torch.zeros(len(mean))),
            mean=mean,
            cov=cov
        )
        mc_guess = fd_mc.functional()
        print(f"MC: {mc_guess}")

        # Compute the analytic eif on the samples after the first nguess.
        correction_points = dict(x=torch.tensor(correction_dataset).float())

        if ndim == 1 and plot_densities:
            # Plot the influence function across the linspace in the same figure.
            plt.plot(xx, analytic_eif(fd_quad, points=dict(x=xx), funcval=oracle_fval), color='blue')
            plt.plot(xx, analytic_eif(fd_mc, points=dict(x=xx), funcval=oracle_fval), color='red')

        # Quick check that the two have the same density.
        assert np.allclose(
            fd_quad.density(correction_points, correction_points),
            fd_mc.density(correction_points, correction_points)
        )

        # And compute the analytic corrections.
        ana_eif_quad = analytic_eif(fd_quad, correction_points, funcval=oracle_fval)
        ana_cor_quad = ana_eif_quad.mean()
        ana_eif_mc = analytic_eif(fd_mc, correction_points, funcval=oracle_fval)
        ana_cor_mc = ana_eif_mc.mean()

        print(f"Quad (Ana Correction): {ana_cor_quad}")
        print(f"MC (Ana Correction): {ana_cor_mc}")

        print(f"Oracle: {oracle_fval}")
        print(f"Quad (Ana Corrected): {quad_guess + ana_cor_quad}")
        print(f"MC (Ana Corrected): {mc_guess + ana_cor_mc}")

        if plot_corrections:
            f2 = plt.figure()
            # Plot the oracle value.
            plt.axhline(oracle_fval, color='black', label='Oracle', linestyle='--')
            # Plot lines from guesses to corrected values.
            plt.plot([0, 1], [quad_guess, quad_guess + ana_cor_quad], color='blue', label='Quad')
            plt.plot([0, 1], [mc_guess, mc_guess + ana_cor_mc], color='red', label='MC')
            plt.legend()

        for eps, lambda_ in product(EPS, LAMBDA):
            fd_eif_quad = fd_influence_fn(
                model=fd_quad,
                points=correction_points,
                eps=eps,
                lambda_=lambda_)()
            fd_cor_quad = np.mean(fd_eif_quad)
            fd_eif_mc = fd_influence_fn(
                model=fd_mc,
                points=correction_points,
                eps=eps,
                # Scale the nmc with epsilon so that the kernel gets seen.
                lambda_=lambda_)(nmc=(1. / eps) * 100)
            fd_cor_mc = np.mean(fd_eif_mc)

            fd_cors_quad[(eps, lambda_)] = fd_cors_quad.get((eps, lambda_), []) + [fd_cor_quad]
            fd_cors_mc[(eps, lambda_)] = fd_cors_mc.get((eps, lambda_), []) + [fd_cor_mc]

        ana_cors_quad.append(ana_cor_quad)
        ana_cors_mc.append(ana_cor_mc)

        print()

        plt.show()
        if ndim == 1 and plot_densities:
            plt.close(f1)
        if plot_corrections:
            plt.close(f2)

    def plot_diag(ax, x1, x2s, x1lab, x2labs, title):
        ax.set_title(title)
        for x2, lab in zip(x2s, x2labs):
            ax.scatter(x1, x2, alpha=0.5, label=lab)
        ax.set_xlabel(x1lab)
        ax.set_ylabel("FD Correction")
        # Draw the diagonal in figure coordinates.
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        # Make the axes the same (min/max of each xy)
        xymin = min(xmin, ymin)
        xymax = max(xmax, ymax)
        ax.set_xlim(xymin, xymax)
        ax.set_ylim(xymin, xymax)
        ax.plot([xymin, xymax], [xymin, xymax], color='black', linestyle='--')

    # Plot the finite difference diagonals.
    if plot_fd_ana_diag:
        # Prep gridplot over eps and lambda.
        f, axes = plt.subplots(len(EPS), len(LAMBDA), figsize=(30, 20))
        for (eps, lambda_), ax in zip(product(EPS, LAMBDA), axes.flatten()):
            plot_diag(
                ax=ax,
                x1=ana_cors_quad,
                x2s=[fd_cors_quad[(eps, lambda_)], fd_cors_mc[(eps, lambda_)]],
                x1lab='Analytic Correction',
                x2labs=['Quad', 'MC'],
                title=f"Quad (ndim={ndim}, eps={eps}, lambda={lambda_})"
            )
        plt.tight_layout()
        plt.show()
        plt.close(f)

    return


if __name__ == '__main__':
    main(plot_densities=True, plot_corrections=True, plot_fd_ana_diag=True)
