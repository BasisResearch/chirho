from robust_fd.squared_normal_density import SquaredNormalDensityQuad
import numpy as np
import torch
import matplotlib.pyplot as plt

ndim = 1
eps = 0.01
mean = torch.tensor([0.,] * ndim)
cov = torch.eye(ndim)
lambda_ = 0.01

sndq = SquaredNormalDensityQuad(
    mean=mean,
    cov=cov,
    eps=eps,
    lambda_=lambda_,
)

print(sndq.functional())

xx = np.linspace(-5, 5, 1000)
yy = [sndq.density(
    {'x': torch.tensor([x])},
    {'x': torch.tensor([x])})
    for x in xx
]

plt.plot(xx, yy)
plt.show()
