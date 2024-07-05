from chirho.robust.internals.utils import pytree_generalized_manual_revjvp
import pytest
import torch
from math import prod


_shapes = [
    tuple(),
    (1,),
    (1, 1),
    (2,),
    (2, 3)
]


def _exec_pytree_generalized_manual_revjvp(batch_shape, output_shape1, output_shape2, param_shape1, param_shape2):

    # TODO add tests of subdicts and sublists to really exercise the pytree structure.
    # TODO add permutations for single tensors params/batch_vector/outputs (i.e. not in an explicit tree structure.

    params = dict(
        params1=torch.randn(param_shape1),
        params2=torch.randn(param_shape2),
    )

    batch_vector = dict(  # this tree is mapped onto the params struture in the right multiplication w/ the jacobian.
        params1=torch.randn(batch_shape + param_shape1),
        params2=torch.randn(batch_shape + param_shape2)
    )

    weights1 = torch.randn(prod(output_shape1), prod(param_shape1))
    weights2 = torch.randn(prod(output_shape2), prod(param_shape2))

    def fn_inner(p: torch.Tensor, weights: torch.Tensor):
        # Arbitrary functino that maps param shape to output shape implicit in weights.
        p = p.flatten()
        out = weights @ p
        return out

    def fn(p):
        return dict(
            out1=fn_inner(p["params1"], weights1).reshape(output_shape1),
            out2=fn_inner(p["params2"], weights2).reshape(output_shape2)
        )

    for (k, v), output_shape in zip(fn(params).items(), (output_shape1, output_shape2)):
        assert v.shape == output_shape

    broadcasted_reverse_jvp_result = pytree_generalized_manual_revjvp(
        fn,
        params,
        batch_vector
    )

    return broadcasted_reverse_jvp_result, (fn, params, batch_vector)


@pytest.mark.parametrize("batch_shape", _shapes)
@pytest.mark.parametrize("output_shape1", _shapes)
@pytest.mark.parametrize("output_shape2", _shapes)
@pytest.mark.parametrize("param_shape1", _shapes)
@pytest.mark.parametrize("param_shape2", _shapes)
def test_smoke_pytree_generalized_manual_revjvp(batch_shape, output_shape1, output_shape2, param_shape1, param_shape2):

    broadcasted_reverse_jvp_result, _ = _exec_pytree_generalized_manual_revjvp(
        batch_shape, output_shape1, output_shape2, param_shape1, param_shape2
    )

    assert broadcasted_reverse_jvp_result["out1"].shape == batch_shape + output_shape1
    assert broadcasted_reverse_jvp_result["out2"].shape == batch_shape + output_shape2


# Standard vmap and jvp application doesn't support multiple batch dims or scalar shapes. So manually spec
#  single batch dims and remove the tuple() scalar shape via _shapes[1:]
@pytest.mark.parametrize("batch_shape", [(1,), (3,)])
@pytest.mark.parametrize("output_shape1", _shapes[1:])
@pytest.mark.parametrize("output_shape2", _shapes[1:])
@pytest.mark.parametrize("param_shape1", _shapes[1:])
@pytest.mark.parametrize("param_shape2", _shapes[1:])
def test_pytree_generalized_manual_revjvp(batch_shape, output_shape1, output_shape2, param_shape1, param_shape2):
    # TODO this test has some stochastic failures. Probably floating point mismatch due to lower precision of computing
    #  jacobian separately?

    broadcasted_reverse_jvp_result, (fn, params, batch_vector) = _exec_pytree_generalized_manual_revjvp(
        batch_shape, output_shape1, output_shape2, param_shape1, param_shape2
    )

    vmapped_forward_jvp_result = torch.vmap(
        lambda d: torch.func.jvp(
            fn,
            (params,),
            (d,),
        )[1],
        in_dims=0,
        randomness="different"
    )(batch_vector)

    assert torch.allclose(broadcasted_reverse_jvp_result["out1"], vmapped_forward_jvp_result["out1"])
    assert torch.allclose(broadcasted_reverse_jvp_result["out2"], vmapped_forward_jvp_result["out2"])
