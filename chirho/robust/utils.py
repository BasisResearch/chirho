import collections
import functools
import inspect
from typing import Callable, Dict, Optional, Tuple, TypeVar

import torch

T = TypeVar("T")


@functools.singledispatch
def make_flatten_unflatten(
    v,
) -> Tuple[Callable[[T], torch.Tensor], Callable[[torch.Tensor], T]]:
    raise NotImplementedError


@make_flatten_unflatten.register(tuple)
def _make_flatten_unflatten_tuple(v: Tuple[torch.Tensor, ...]):
    sizes = [x.size() for x in v]

    def flatten(xs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        return torch.cat([x.reshape(-1) for x in xs], dim=0)

    def unflatten(x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        tensors = []
        i = 0
        for size in sizes:
            num_elements = torch.prod(torch.tensor(size))
            tensors.append(x[i : i + num_elements].view(size))
            i += num_elements
        return tuple(tensors)

    return flatten, unflatten


@make_flatten_unflatten.register(dict)
def _make_flatten_unflatten_dict(d: Dict[str, torch.Tensor]):
    def flatten(d: Dict[str, torch.Tensor]) -> torch.Tensor:
        r"""
        Flatten a dictionary of tensors into a single vector.
        """
        return torch.cat([v.flatten() for k, v in d.items()])

    def unflatten(x: torch.Tensor) -> Dict[str, torch.Tensor]:
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

    return flatten, unflatten


def conjugate_gradient_solve(f_Ax: Callable[[T], T], b: T, **kwargs) -> T:
    flatten, unflatten = make_flatten_unflatten(b)
    f_Ax_flat = lambda v: flatten(f_Ax(flatten(v)))
    b_flat = flatten(b)
    return unflatten(_flat_conjugate_gradient_solve(f_Ax_flat, b_flat, **kwargs))


def _flat_conjugate_gradient_solve(
    f_Ax: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    *,
    cg_iters: Optional[int] = None,
    residual_tol: float = 1e-10
) -> torch.Tensor:
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
    if cg_iters is None:
        cg_iters = b.numel()

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
