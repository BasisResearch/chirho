import torch


# TODO dispatch on rho type to use different log funcs (torch, numpy, fenics)
def stable_gamma(rho, lA, gravity=9.81, n=3., logf=torch.log, expf=torch.exp):
    ncoef = (2. / (n + 2.))
    lpg = logf(rho * gravity)
    lApgn = lA + n * lpg

    return ncoef * expf(lApgn)


def gamma(rho, A, gravity=9.81, n=3.):
    ncoef = (2. / (n + 2.))

    return ncoef * A * (rho * gravity) ** n


def initial_curve(x):
    condition = (x >= -1.0) & (x <= 1.0)
    values = (1.0 - x**2 + 0.3 * torch.sin(2. * torch.pi * x) ** 2.)
    return torch.where(condition, values, torch.zeros_like(x))


def fillna(tensor):
    return torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)