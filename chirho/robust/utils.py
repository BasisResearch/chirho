import inspect
import collections
from typing import Callable, Dict
import torch


def _flatten_dict(d: Dict[str, torch.tensor]) -> torch.tensor:
    r"""
    Flatten a dictionary of tensors into a single vector.
    """
    return torch.cat([v.flatten() for k, v in d.items()])


def _unflatten_dict(
    x: torch.tensor, d: Dict[str, torch.tensor]
) -> Dict[str, torch.tensor]:
    r"""
    Unflatten a vector into a dictionary of tensors.
    """
    return collections.OrderedDict(
        zip(
            d.keys(),
            [
                v_flat.reshape(v.shape)
                for v, v_flat in zip(
                    d.values(), torch.split(x, [v.numel() for k, v in d.items()])
                )
            ],
        )
    )


def _conjugate_gradient(
    f_Ax: Callable, b: torch.tensor, cg_iters: int, residual_tol: float = 1e-10
) -> torch.tensor:
    r"""Use Conjugate Gradient iteration to solve Ax = b. Demmel p 312.

    Args:
        f_Ax (callable): A function to compute matrix vector product.
        b (torch.Tensor): Right hand side of the equation to solve.
        cg_iters (int): Number of iterations to run conjugate gradient
            algorithm.
        residual_tol (float): Tolerence for convergence.

    Returns:
        torch.Tensor: Solution x* for equation Ax = b.

    Notes: This code is copied from https://github.com/rlworkgroup/garage/blob/master/src/garage/torch/optimizers/conjugate_gradient_optimizer.py
    """
    p = b.clone()
    r = b.clone()
    x = torch.zeros_like(b)
    rdotr = torch.dot(r, r)

    for _ in range(cg_iters):
        z = f_Ax(p)
        v = rdotr / torch.dot(p, z)
        x += v * p
        r -= v * z
        newrdotr = torch.dot(r, r)
        mu = newrdotr / rdotr
        p = r + mu * p

        # Still executes loop but effectively stops update (can't break loop since we're using vmap)
        # rdotr = torch.where(rdotr < residual_tol, rdotr, newrdotr)
        # rdotr = newrdotr
        # if rdotr < residual_tol:
        #     break
    return x


def _check_correct_model_signature(model):
    r"""
    Assert that the forward method of a model takes only a single argument, `N`.

    Args:
        model (PyroModule): Pyro model.
    """
    forward_method = getattr(model, "forward", None)

    if forward_method:
        sig = inspect.signature(forward_method)
        expected_params = ["N"]

        assert (
            list(sig.parameters.keys()) == expected_params
        ), "Method signature doesn't match."
    else:
        assert False, "Method 'forward' not found."


def _check_correct_functional_signature(functional):
    r"""
    Assert that the functional takes only arguments `model`, `theta_hat`, and
    potentially an optional argument `n_monte_carlo`.

    Args:
        functional (Callable): Functional to check.
    """
    sig = inspect.signature(functional)
    expected_params = ["model", "theta_hat"]
    arg_names = list(sig.parameters.keys())
    arg_names = [arg for arg in arg_names if arg != "n_monte_carlo"]
    assert (
        arg_names == expected_params
    ), "Functional signature doesn't match. Got {}, expected {}".format(
        arg_names, expected_params
    )
