from math import prod

import pytest
import torch

from chirho.robust.internals.utils import pytree_generalized_manual_revjvp

_shapes = [tuple(), (1,), (1, 1), (2,), (2, 3)]


def _exec_pytree_generalized_manual_revjvp(
    batch_shape, output_shape1, output_shape2, param_shape1, param_shape2
):

    # TODO add tests of subdicts and sublists to really exercise the pytree structure.
    # TODO add permutations for single tensors params/batch_vector/outputs (i.e. not in an explicit tree structure.

    params = dict(
        params1=torch.randn(param_shape1),
        params2=torch.randn(param_shape2),
    )

    batch_vector = dict(  # this tree is mapped onto the params struture in the right multiplication w/ the jacobian.
        params1=torch.randn(batch_shape + param_shape1),
        params2=torch.randn(batch_shape + param_shape2),
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
            out2=fn_inner(p["params2"], weights2).reshape(output_shape2),
        )

    for (k, v), output_shape in zip(fn(params).items(), (output_shape1, output_shape2)):
        assert v.shape == output_shape

    broadcasted_reverse_jvp_result = pytree_generalized_manual_revjvp(
        fn, params, batch_vector
    )

    return broadcasted_reverse_jvp_result, (fn, params, batch_vector)


@pytest.mark.parametrize("batch_shape", _shapes)
@pytest.mark.parametrize("output_shape1", _shapes)
@pytest.mark.parametrize("output_shape2", _shapes)
@pytest.mark.parametrize("param_shape1", _shapes)
@pytest.mark.parametrize("param_shape2", _shapes)
def test_smoke_pytree_generalized_manual_revjvp(
    batch_shape, output_shape1, output_shape2, param_shape1, param_shape2
):

    broadcasted_reverse_jvp_result, _ = _exec_pytree_generalized_manual_revjvp(
        batch_shape, output_shape1, output_shape2, param_shape1, param_shape2
    )

    assert broadcasted_reverse_jvp_result["out1"].shape == batch_shape + output_shape1
    assert broadcasted_reverse_jvp_result["out2"].shape == batch_shape + output_shape2

    assert not torch.isnan(broadcasted_reverse_jvp_result["out1"]).any()
    assert not torch.isnan(broadcasted_reverse_jvp_result["out2"]).any()


# Standard vmap and jvp application doesn't support multiple batch dims or scalar shapes. So manually spec
#  single batch dims and remove the tuple() scalar shape via _shapes[1:]
@pytest.mark.parametrize("batch_shape", [(1,), (3,)])
@pytest.mark.parametrize("output_shape1", _shapes[1:])
@pytest.mark.parametrize("output_shape2", _shapes[1:])
@pytest.mark.parametrize("param_shape1", _shapes[1:])
@pytest.mark.parametrize("param_shape2", _shapes[1:])
def test_pytree_generalized_manual_revjvp(
    batch_shape, output_shape1, output_shape2, param_shape1, param_shape2
):

    broadcasted_reverse_jvp_result, (fn, params, batch_vector) = (
        _exec_pytree_generalized_manual_revjvp(
            batch_shape, output_shape1, output_shape2, param_shape1, param_shape2
        )
    )

    vmapped_forward_jvp_result = torch.vmap(
        lambda d: torch.func.jvp(
            fn,
            (params,),
            (d,),
        )[1],
        in_dims=0,
        randomness="different",
    )(batch_vector)

    # When using standard precision, this test has some stochastic failures (around 1/3000) that pass on rerun.
    # This is probably due to floating point mismatch induced by lower precision of separate jacobian computation
    #  and manual matmul?
    assert torch.allclose(
        broadcasted_reverse_jvp_result["out1"],
        vmapped_forward_jvp_result["out1"],
        atol=1e-5,
    )
    assert torch.allclose(
        broadcasted_reverse_jvp_result["out2"],
        vmapped_forward_jvp_result["out2"],
        atol=1e-5,
    )


def test_memory_pytree_generalized_manual_revjvp():
    # vmap over jvp can not handle 1000 batch x 1000 params (10s of gigabytes used).
    batch_shape = (10000,)
    output_shape1 = (2,)
    output_shape2 = (2,)
    params_shape1 = (10000,)
    params_shape2 = (10000,)
    # Also works with these, but runtime is too long for CI. Runs locally at a little over 7GB.
    # params_shape1 = (100000,)
    # params_shape2 = (100000,)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        profile_memory=True,
        with_stack=False,
    ) as prof:

        broadcasted_reverse_jvp_result, _ = _exec_pytree_generalized_manual_revjvp(
            batch_shape, output_shape1, output_shape2, params_shape1, params_shape2
        )

    assert broadcasted_reverse_jvp_result["out1"].shape == batch_shape + output_shape1
    assert broadcasted_reverse_jvp_result["out2"].shape == batch_shape + output_shape2

    assert not torch.isnan(broadcasted_reverse_jvp_result["out1"]).any()
    assert not torch.isnan(broadcasted_reverse_jvp_result["out2"]).any()

    # Summing up the self CPU memory usage
    total_memory_allocated = sum(
        [item.self_cpu_memory_usage for item in prof.key_averages()]
    )
    total_gb_allocated = total_memory_allocated / (1024**3)

    # Locally, this runs at slightly over 1.0 GB.
    assert (
        total_gb_allocated < 3.0
    ), f"Memory usage was {total_gb_allocated} GB, which is too high."
