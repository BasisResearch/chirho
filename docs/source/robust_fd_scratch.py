from robust_fd.squared_normal_density import ExpectedNormalDensityQuad, ExpectedNormalDensityMC
from chirho.robust.handlers.fd_model import fd_influence_fn
import numpy as np
import torch
import matplotlib.pyplot as plt

ndim = 1
eps = 0.01
mean = torch.tensor([0.,] * ndim)
cov = torch.eye(ndim)
lambda_ = 0.01

end_quad = ExpectedNormalDensityQuad(
    mean=mean,
    cov=cov,
    default_kernel_point=dict(x=torch.tensor([0.,] * ndim)),
    default_eps=eps,
    default_lambda=lambda_,
)

print(end_quad.functional())

xx = np.linspace(-5, 5, 1000)

with end_quad.set_kernel_point(dict(x=torch.tensor([1., ] * ndim))), end_quad.set_lambda(.01), end_quad.set_eps(0.1):
    yy = [end_quad.density(
        {'x': torch.tensor([x])},
        {'x': torch.tensor([x])})
        for x in xx
    ]


plt.plot(xx, yy)
plt.show()

# Sample points from a slightly more entropoic model.
# FIXME not generalized for ndim > 1
points = dict(x=torch.linspace(-3, 3, 100)[:, None])

target_quad = fd_influence_fn(
    model=end_quad,
    points=points,
    eps=0.1,
    lambda_=0.1,
)

correction_quad = target_quad()

print(correction_quad)

end_mc = ExpectedNormalDensityMC(
    mean=mean,
    cov=cov,
    default_kernel_point=dict(x=torch.tensor([0.,] * ndim)),
    default_eps=eps,
    default_lambda=lambda_,
)

target_mc = fd_influence_fn(
    model=end_mc,
    points=points,
    eps=0.1,
    lambda_=0.1,
)

correction_mc = target_mc(nmc=10000000).item()

print(correction_mc)
