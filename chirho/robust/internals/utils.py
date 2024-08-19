import contextlib
from math import prod
from typing import Callable, List, Mapping, Optional, Tuple, TypeVar

import pyro
import torch
from torch.utils._pytree import (
    SUPPORTED_NODES,
    PyTree,
    TreeSpec,
    _get_node_type,
    tree_flatten,
    tree_unflatten,
)
from typing_extensions import Concatenate, ParamSpec

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
U = TypeVar("U")

ParamDict = Mapping[str, torch.Tensor]


def make_flatten_unflatten(
    v: T,
) -> Tuple[Callable[[T], torch.Tensor], Callable[[torch.Tensor], T]]:
    """
    Returns functions to flatten and unflatten an object. Used as a helper
    in :func:`chirho.robust.internals.linearize.conjugate_gradient_solve`

    :param v: some object
    :raises NotImplementedError:
    :return: flatten and unflatten functions
    :rtype: Tuple[Callable[[T], torch.Tensor], Callable[[torch.Tensor], T]]
    """
    flat_v, treespec = torch.utils._pytree.tree_flatten(v)

    def _flatten(unflat_v: T) -> torch.Tensor:
        parts, _ = torch.utils._pytree.tree_flatten(unflat_v)
        return torch.hstack([x.reshape((x.shape[0], -1)) for x in parts])

    def _unflatten(single_flat_v: torch.Tensor) -> T:
        parts = [
            v_flat.reshape(v.shape)
            for v, v_flat in zip(
                flat_v,
                torch.split(
                    single_flat_v,
                    [int(vi.numel() / flat_v[0].shape[0]) for vi in flat_v],
                    dim=1,
                ),
            )
        ]
        return torch.utils._pytree.tree_unflatten(parts, treespec)

    return _flatten, _unflatten


SPyTree = TypeVar("SPyTree", bound=PyTree)
TPyTree = TypeVar("TPyTree", bound=PyTree)
UPyTree = TypeVar("UPyTree", bound=PyTree)


def pytree_generalized_manual_revjvp(
    fn: Callable[[TPyTree], SPyTree], params: TPyTree, batched_vector: UPyTree
) -> SPyTree:
    """
    Computes the jacobian-vector product using backward differentiation for the jacobian, and then manually
    right multiplying the batched vector. This supports pytree structured inputs, outputs, and params.

    :param fn: function to compute the jacobian of
    :param params: parameters to compute the jacobian at
    :param batched_vector: batched vector to right multiply the jacobian by
    :raises ValueError: if params and batched_vector do not have the same tree structure
    :return: jacobian-vector product
    """

    # Assumptions (in terms of elements of the referenced pytrees):
    # 1. params is not batched, and represents just the inputs to the fn that we'll take the jac wrt.
    #    - params.shape == (*param_shape)
    # 2. batched_vector is the batched vector component of the jv product. It's rightside shape matches params.
    #    - batched_vector.shape == (*batch_shape, *param_shape)
    # 3. The output of the function will have some output_shape, which will cause the jacobian to have shape.
    #    - jac.shape == (*output_shape, *param_shape)
    # So the task is to infer these shapes and line everything up correctly. As a general approach, we'll flatten
    #  the inputs and output shapes in order to apply a standard batched matrix multiplication operation.
    # The output will have shape (*batch_shape, *output_shape).

    # The shaping is complicated by fact that we aren't working with tensors, but PyTrees instead, and we want to
    #  perform the same inner product wrt to the tree structure. This mainly shows up in that the jacobian will
    #  return a pytree with a "root" structure matching that of SPyTree (the return of the fn), but at each leaf
    #  of that tree, we have a pytree matching the structure of TPyTree (the params). This is the tree-structured
    #  equivalent the jac shape matching output on the left, and params on the right.

    jac_fn = torch.func.jacrev(fn)
    jac = jac_fn(params)

    flat_params, param_tspec = tree_flatten(params)

    flat_batched_vector, batched_vector_tspec = tree_flatten(batched_vector)

    if param_tspec != batched_vector_tspec:
        # This is also required by pytorch's jvp implementation.
        raise ValueError(
            "params and batched_vector must have the same tree structure. This requirement generalizes"
            " the notion that the batched_vector must be the correct shape to right multiply the "
            "jacobian."
        )

    # In order to map the param shapes together, we need to iterate through the output tree structure and map each
    #  subtree (corresponding to params) onto the params and batched_vector tree structures, which are both structured
    #  according to the parameters.
    def recurse_to_flattened_sub_tspec(
        pytree: PyTree, sub_tspec: TreeSpec, tspec: Optional[TreeSpec] = None
    ):
        # Default to passed treespec, otherwise compute here.
        _, tspec = tree_flatten(pytree) if tspec is None else (None, tspec)

        # If fn returns a tensor straight away, then the subtree will match at the root node. Check for that here.
        if tspec == sub_tspec:
            flattened, _ = tree_flatten(pytree)
            yield flattened
            return

        # Extract child trees in a node-type agnostic way.
        node_type = _get_node_type(pytree)
        flatten_fn = SUPPORTED_NODES[node_type].flatten_fn
        children_pytrees, _ = flatten_fn(pytree)
        children_tspecs = tspec.children_specs

        # Iterate through children and their specs.
        for child_pytree, child_tspec in zip(children_pytrees, children_tspecs):
            # If we've landed on the target subtree...
            if child_tspec == sub_tspec:
                child_flattened, _ = tree_flatten(child_pytree)
                yield child_flattened  # ...yield the flat child for that subtree.
            else:  # otherwise, recurse to the next level.
                yield from recurse_to_flattened_sub_tspec(
                    child_pytree, sub_tspec, tspec=child_tspec
                )

    flat_out: List[PyTree] = []

    # Recurse into the jacobian tree to find the subtree corresponding to the sub-jacobian for each
    #  individual output tensor in that tree.
    for flat_jac_output_subtree in recurse_to_flattened_sub_tspec(
        pytree=jac, sub_tspec=param_tspec
    ):
        flat_sub_out: List[torch.Tensor] = []

        # Then map that subtree (with tree structure matching that of params) onto the params and batched_vector.
        for i, (p, j, v) in enumerate(
            zip(flat_params, flat_jac_output_subtree, flat_batched_vector)
        ):
            # Infer the parameter shapes directly from passed parameters.
            og_param_shape = p.shape
            param_shape = og_param_shape if len(og_param_shape) else (1,)
            param_numel = prod(param_shape)
            og_param_ndim = len(og_param_shape)

            # Infer the batch shape by subtracting off the param shape on the right.
            og_batch_shape = v.shape[:-og_param_ndim] if og_param_ndim else v.shape
            batch_shape = og_batch_shape if len(og_batch_shape) else (1,)
            batch_ndim = len(batch_shape)

            # Infer the output shape by subtracting off the param shape from the jacobian.
            og_output_shape = j.shape[:-og_param_ndim] if og_param_ndim else j.shape
            output_shape = og_output_shape if len(og_output_shape) else (1,)
            output_numel = prod(output_shape)

            # Reshape for matmul and s.t. that the jacobian can be broadcast over the batch dims.
            j_bm = j.reshape(*(1,) * batch_ndim, output_numel, param_numel)
            v_bm = v.reshape(*batch_shape, param_numel, 1)
            jv = j_bm @ v_bm

            # Reshape result back to the original output shape, with support for empty scalar shapes.
            og_res_shape = (*og_batch_shape, *og_output_shape)
            jv = jv.reshape(*og_res_shape) if len(og_res_shape) else jv.squeeze()

            flat_sub_out.append(jv)

        # The inner product is operating over parameters and the parameter subtree that we just iterated over.
        # So stack these and sum.
        flat_out.append(torch.stack(flat_sub_out, dim=0).sum(0))

    # flat_out is now the flattened version of the tree returned by fn, with each contained tensor having the same
    #  batch dimensions (matching the batching of the batched vector).
    # TODO get out_treespec from the jacobian treespec instead, and don't have an extra forward eval of fn.
    #  Jacobian tree has this structure but its leaves have params treespec.
    out = fn(params)
    _, out_treespec = tree_flatten(out)

    return tree_unflatten(flat_out, out_treespec)


def make_functional_call(
    mod: Callable[P, T],
    ignore_params_not_requiring_grad: bool = False,
) -> Tuple[ParamDict, Callable[Concatenate[ParamDict, P], T]]:
    """
    Converts a PyTorch module into a functional call for use with
    functions in :class:`torch.func`.

    :param mod: PyTorch module
    :type mod: Callable[P, T]
    :param ignore_params_not_requiring_grad: whether treat params that don't require_grad as normal tensors.
    :type ignore_params_not_requiring_grad: bool
    :return: parameter dictionary and functional call
    :rtype: Tuple[ParamDict, Callable[Concatenate[ParamDict, P], T]]
    """
    assert isinstance(mod, torch.nn.Module)
    param_dict: ParamDict = dict(mod.named_parameters())

    if ignore_params_not_requiring_grad:
        # TODO check if any of these are FunctionalTensors, cz this breaks in that context.
        param_dict = {k: v for k, v in param_dict.items() if v.requires_grad}

    @torch.func.functionalize
    def mod_func(params: ParamDict, *args: P.args, **kwargs: P.kwargs) -> T:
        with pyro.validation_enabled(False):
            return torch.func.functional_call(mod, params, args, dict(**kwargs))

    return param_dict, mod_func


@contextlib.contextmanager
def reset_rng_state(rng_state: T):
    """
    Helper to temporarily reset the Pyro RNG state.
    """
    try:
        prev_rng_state: T = pyro.util.get_rng_state()
        yield pyro.util.set_rng_state(rng_state)
    finally:
        pyro.util.set_rng_state(prev_rng_state)
