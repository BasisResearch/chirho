import torch


def initial_curve(x):
    condition = (x >= -1.0) & (x <= 1.0)
    values = (1.0 - x**2 + 0.3 * torch.sin(2. * torch.pi * x) ** 2.)
    return torch.where(condition, values, torch.zeros_like(x))


def fillna(tensor):
    return torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)