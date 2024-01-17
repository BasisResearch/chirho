raise NotImplementedError()

from robust_fd.squared_normal_density import ExpectedNormalDensityQuad, ExpectedNormalDensityMC, _ExpectedNormalDensity
from chirho.robust.handlers.fd_model import fd_influence_fn
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm

ndim = 1
eps = 0.01
mean = torch.tensor([0.,] * ndim)
cov = torch.eye(ndim)
lambda_ = 0.001

end_quad = ExpectedNormalDensityQuad(
    mean=mean,
    cov=cov,
    default_kernel_point=dict(x=torch.tensor([0.,] * ndim)),
)

guess = end_quad.functional()
print(f"Guess: {guess}")

if ndim == 1:
    xx = np.linspace(-5, 5, 1000)
    with (end_quad.set_kernel_point(dict(x=torch.tensor([1., ] * ndim))),
          end_quad.set_lambda(lambda_),
          end_quad.set_eps(eps)):
        yy = [end_quad.density(
            {'x': torch.tensor([x])},
            {'x': torch.tensor([x])})
            for x in xx
        ]

        plt.plot(xx, yy)

    yy = [end_quad.density(
        {'x': torch.tensor([x])},
        {'x': torch.tensor([x])})
        for x in xx
    ]

    plt.plot(xx, yy)

# Sample points from a slightly more entropoic model.
# FIXME not generalized for ndim > 1
points = dict(x=torch.linspace(-3, 3, 50)[:, None])

print(f"Analytic: {((1./(3. - -3.))**2) * (3. - -3.)}")

target_quad = fd_influence_fn(
    model=end_quad,
    points=points,
    eps=eps,
    lambda_=lambda_,
)

correction_quad_eif = np.array(target_quad())

if ndim == 1:
    plt.figure()
    plt.plot(points['x'].numpy(), correction_quad_eif, label='quad eif')

correction_quad = np.mean(correction_quad_eif)

print(f"Correction (Quad): {correction_quad}")

end_mc = ExpectedNormalDensityMC(
    mean=mean,
    cov=cov,
    default_kernel_point=dict(x=torch.tensor([0.,] * ndim)),
)

target_mc = fd_influence_fn(
    model=end_mc,
    points=points,
    eps=eps,
    lambda_=lambda_,
)

correction_mc_eif = np.array(target_mc(nmc=4000))

if ndim == 1:
    plt.plot(points['x'].numpy(), correction_mc_eif, linewidth=0.3, alpha=0.8)

correction_mc = np.mean(correction_mc_eif)

print(f"Correction (MC): {correction_mc}")


def compute_analytic_eif(model: _ExpectedNormalDensity, points):
    funcval = model.functional()
    density = model.density(points, points)

    return 2. * (density - funcval)


analytic_eif = compute_analytic_eif(end_quad, points).numpy()

analytic = np.mean(analytic_eif)

print(f"Analytic: {analytic}")

print(f"Analytic Corrected: {guess - analytic}")


if ndim == 1:

    plt.suptitle(f"ndim={ndim}, eps={eps}, lambda={lambda_}")

    pxsamps = points['x'].numpy().squeeze()

    plt.plot(pxsamps, analytic_eif, label="analytic")

    # Plot the corresponding uniform and normal densities.
    plt.plot(points['x'].numpy(), [1./(3. - -3.)] * len(points['x']), color='black', label='uniform')

    # plt.plot(xx, norm.pdf(xx, loc=0, scale=1), color='green', label='normal')
    plt.plot(pxsamps, norm.pdf(pxsamps, loc=0, scale=1), color='green', label='normal')
    # Plot the correction, just quad.
    plt.plot(pxsamps, norm.pdf(pxsamps, loc=0, scale=1) - 0.1 * np.array(correction_quad_eif),
             linestyle='--', color='green', label='normal (corrected)')

    plt.legend()
    plt.show()
