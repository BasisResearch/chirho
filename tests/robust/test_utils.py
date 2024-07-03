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


@pytest.mark.parametrize("batch_shape", _shapes)
@pytest.mark.parametrize("output_shape1", _shapes)
@pytest.mark.parametrize("output_shape2", _shapes)  # some redundant tests here  in lower triangle of test case prod
@pytest.mark.parametrize("param_shape1", _shapes)
@pytest.mark.parametrize("param_shape2", _shapes)  # some redundant tests here  in lower triangle of test case prod
def test_pytree_generalized_manual_revjvp(batch_shape, output_shape1, output_shape2, param_shape1, param_shape2):

    params = dict(
        params1=torch.randn(param_shape1),
        params2=torch.randn(param_shape2)
    )

    batch_vector = dict(
        batch_vector1=torch.randn(batch_shape + param_shape1),
        batch_vector2=torch.randn(batch_shape + param_shape2)
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

    assert broadcasted_reverse_jvp_result["out1"].shape == batch_shape + output_shape1
    assert broadcasted_reverse_jvp_result["out2"].shape == batch_shape + output_shape2

    # skip test.
    # pytest.skip("Not implemented yet.")
