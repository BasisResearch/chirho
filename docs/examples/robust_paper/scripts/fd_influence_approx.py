from typing import List, Dict, Tuple
import pyro
import pyro.distributions as dist

# TODO move these into __init__.py of finite_difference_eif for single import.
from docs.examples.robust_paper.finite_difference_eif.mixins import (
    NormalKernel,
    ExpectedDensityMCFunctional,
    ExpectedDensityQuadFunctional
)
from docs.examples.robust_paper.finite_difference_eif.abstractions import (
    fd_influence_fn,
    FDModelFunctionalDensity
)
from docs.examples.robust_paper.finite_difference_eif.distributions import (
    PerturbableNormal
)
import torch
from itertools import product
from chirho.robust.ops import Point, T
from docs.examples.robust_paper.utils import rng_seed_context
import time
import numpy as np


# Couple together perturbation kernels, perturbable models and functionals.
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


def compute_fd_correction_sqd_mvn_quad(*, theta_hat: Point[T], **kwargs) -> List[Dict]:
    mean = theta_hat['mean'].detach()
    scale_tril = theta_hat['scale_tril'].detach()

    fd_coupling = ExpectedNormalDensityQuadFunctional(
        # TODO agnostic to names
        default_kernel_point=dict(x=mean),
        mean=mean,
        scale_tril=scale_tril
    )

    return compute_fd_correction(fd_coupling, **kwargs)


def compute_fd_correction_sqd_mvn_mc(*, theta_hat: Point[T], **kwargs) -> List[Dict]:
    mean = theta_hat['mean'].detach()
    scale_tril = theta_hat['scale_tril'].detach()

    fd_coupling = ExpectedNormalDensityMCFunctional(
        # TODO agnostic to names
        default_kernel_point=dict(x=mean),
        mean=mean,
        scale_tril=scale_tril
    )

    return compute_fd_correction(fd_coupling, **kwargs)


def compute_fd_correction(
        fd_coupling: FDModelFunctionalDensity,
        test_data: Point[T],
        lambdas: List[float],
        epss: List[float],
        num_samples_scaling: int,
        seed: int,
) -> List[Dict]:
    epslam = product(epss, lambdas)

    results = list()

    for eps, lambda_ in epslam:
        result = dict()

        with rng_seed_context(seed):
            st = time.time()

            # TODO HACK nmc depends on eps but only applies when fd_coupling.functional takes that argument.
            #  Better ways to abstract this.
            functional_kwargs = dict()
            if isinstance(fd_coupling, ExpectedDensityMCFunctional):
                functional_kwargs['nmc'] = int(num_samples_scaling / eps)

            pointwise = fd_influence_fn(
                fd_coupling=fd_coupling,
                points=test_data,
                eps=eps,
                lambda_=lambda_
            )(**functional_kwargs)

            result['wall_time'] = time.time() - st

        result['eps'] = eps
        result['lambda'] = lambda_
        result['pointwise'] = [
            y if isinstance(y, float) else y.item() for y in pointwise
        ]
        result['correction'] = np.mean(pointwise)

        results.append(result)

    return results


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import json

    def smoke_test():
        # Recommended values for experiments are commented out.
        fd_kwargs = dict(
            # lambdas=[0.1, 0.01, 0.001],
            # epss=[0.1, 0.01, 0.001, 0.0001],
            # num_samples_scaling=100,
            # seed=0
            lambdas=[0.001],
            epss=[0.01, 0.001],
            num_samples_scaling=10000,
            seed=0
        )

        # Runtime
        for ndim in [1, 2]:
            theta_hat = th = dict(
                mean=torch.zeros(ndim),
                scale_tril=torch.linalg.cholesky(torch.eye(ndim))
            )

            test_data = dict(
                x=dist.MultivariateNormal(loc=th['mean'], scale_tril=th['scale_tril']).sample((20,))
            )

            mc_correction = compute_fd_correction_sqd_mvn_mc(
                theta_hat=theta_hat,
                test_data=test_data,
                **fd_kwargs
            )

            quad_correction = compute_fd_correction_sqd_mvn_quad(
                theta_hat=theta_hat,
                test_data=test_data,
                **fd_kwargs
            )

            # print("MC Correction")
            # print(json.dumps(mc_correction, indent=2))
            # print("Quad Correction")
            # print(json.dumps(quad_correction, indent=2))

            # Plot the quad correction results against the MC correction results. Fix aspect ratio.
            for mc, qu in zip(mc_correction, quad_correction):
                plt.figure()
                plt.suptitle(f"D={ndim}, eps={mc['eps']}, lambda={mc['lambda']}")
                plt.plot(mc['pointwise'], qu['pointwise'], 'o')
                plt.xlabel("MC Correction")
                plt.ylabel("Quad Correction")
                # Set xlim and ylim to the same range.
                xymin = min(min(mc['pointwise']), min(qu['pointwise'])) - 0.1
                xymax = max(max(mc['pointwise']), max(qu['pointwise'])) + 0.1
                plt.xlim(xymin, xymax)
                plt.ylim(xymin, xymax)
                plt.show()

    smoke_test()
