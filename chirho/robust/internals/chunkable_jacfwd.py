"""
Modified version of torch.func.jacfwd that accepts chunk size.
"""
from torch._functorch.eager_transforms import (
    _slice_argnums,
    _construct_standard_basis_for,
    vmap,
    tree_flatten,
    Callable,
    argnums_t,
    wraps,
    tree_unflatten,
    _jvp_with_argnums,
    safe_unflatten,
)


def jacfwd(
    func: Callable,
    argnums: argnums_t = 0,
    has_aux: bool = False,
    *,
    randomness: str = "error",
    chunk_size=None
):
    """
    Wrapper around torch._functorch.jacfwd that accepts chunk size.
    See the original torch documentation for more details on the arguments.

    **Arguments:**

    - `func`: A Python function that takes one or more arguments, one of which
        must be a Tensor, and returns one or more Tensors
    - `argnums`: An integer or a tuple of integers specifying which positional
        argument(s) to differentiate with respect to.
    - `has_aux`: If `True`, `func` is assumed to return a pair where the first
        element is considered the output of the original function to be
        differentiated and the second element is auxiliary data.
    - `randomness`: A string specifying how to handle randomness in `func`.
        Valid values are “different”, “same”, “error”.
    - `chunk_size`: An integer specifying the chunk size for vmap.
    """
    @wraps(func)
    def wrapper_fn(*args, tangents=None):
        primals = args if argnums is None else _slice_argnums(args, argnums)
        flat_primals, primals_spec = tree_flatten(primals)
        flat_primals_numels = tuple(p.numel() for p in flat_primals)

        if tangents is None:  # TODO BREAKPOINT check if tangents is the same shape as standard basis. probs have to chunk it.
            flat_basis = _construct_standard_basis_for(flat_primals, flat_primals_numels)
            basis = tree_unflatten(flat_basis, primals_spec)
        else:
            basis = tangents,

        def push_jvp(basis):
            output = _jvp_with_argnums(
                func, args, basis, argnums=argnums, has_aux=has_aux
            )
            # output[0] is the output of `func(*args)`
            if has_aux:
                _, jvp_out, aux = output
                return jvp_out, aux
            _, jvp_out = output
            return jvp_out

        results = vmap(push_jvp, randomness=randomness, chunk_size=chunk_size)(basis)
        if has_aux:
            results, aux = results
            # aux is in the standard basis format, e.g. NxN matrix
            # We need to fetch the first element as original `func` output
            flat_aux, aux_spec = tree_flatten(aux)
            flat_aux = [value[0] for value in flat_aux]
            aux = tree_unflatten(flat_aux, aux_spec)

        jac_outs, spec = tree_flatten(results)
        # Most probably below output check can never raise an error
        # as jvp should test the output before
        # assert_non_empty_output(jac_outs, 'jacfwd(f, ...)(*args)')

        jac_outs_ins = tuple(
            tuple(
                safe_unflatten(jac_out_in, -1, primal.shape)
                for primal, jac_out_in in zip(
                    flat_primals,
                    jac_out.movedim(0, -1).split(flat_primals_numels, dim=-1),
                )
            )
            for jac_out in jac_outs
        )
        jac_outs_ins = tuple(
            tree_unflatten(jac_ins, primals_spec) for jac_ins in jac_outs_ins
        )

        if isinstance(argnums, int):
            jac_outs_ins = tuple(jac_ins[0] for jac_ins in jac_outs_ins)
        if has_aux:
            return tree_unflatten(jac_outs_ins, spec), aux
        return tree_unflatten(jac_outs_ins, spec)

    return wrapper_fn
