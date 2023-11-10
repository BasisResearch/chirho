import pytest
import torch
import chirho.contrib.experiments.closed_form as cfe
from torch import tensor as tnsr


# Define the test cases (q, c, n) for both 1D and 2D scenarios
test_cases = [
    (0.1, 11.49544, 1),
    (0.4, 7.486817, 1),
    (0.8, 16.4037516, 1),
    (0.5, 104.367511, 1),
    (0.1, 38.126061902, 2),
    (0.4, 14.006552, 2),
    (0.8, 24.605627, 2),
    (0.5, 180.76983, 2)
]


@pytest.mark.parametrize("q, c, n", test_cases)
def test_r_c_inversion(q, c, n):
    q, c, n = tnsr([q]), tnsr([c]), n

    rstar = cfe.compute_ana_rstar(q, c, n)
    c_prime = cfe.compute_ana_c(q, rstar, n)

    print(rstar, q, c, n)

    assert torch.isclose(c, c_prime)


@pytest.mark.parametrize("q, c, n", test_cases)
def test_ana_sol_num_sol_simple_num_sol_full_agreement(q, c, n):

    q, c, n = tnsr([q]), tnsr([c]), n

    # Set up Q and Sigma according to assumptions made for the simple case.
    Q = q * torch.eye(n)
    Sigma = torch.eye(n)

    Sigma = cfe.rescale_cov_to_unit_mass(Sigma)

    def neg_ana_obj_simple(r):
        return -cfe.simple_ana_obj(r=r, q=q, c=c, n=n)

    def neg_ana_obj_full(r):
        theta = torch.randn(n)
        theta /= torch.linalg.norm(theta)
        return -cfe.full_ana_obj(theta * r, Q=Q, Sigma=Sigma, c=c)

    # First make sure that the two functions agree across a range of r values.
    r_values = torch.linspace(1e-3, 10, 1000)
    simple_obj_values = tnsr([neg_ana_obj_simple(r) for r in r_values])
    full_obj_values = tnsr([neg_ana_obj_full(r) for r in r_values])
    assert torch.isclose(tnsr(simple_obj_values), tnsr(full_obj_values)).all()

    # Get the analytic solution for this q, c, n.
    rstar = cfe.compute_ana_rstar(q, c, n)

    # Evaluate it in each of the objective functions, make sure they are close.
    simple_obj_value = neg_ana_obj_simple(rstar)
    full_obj_value = neg_ana_obj_full(rstar)
    assert torch.isclose(simple_obj_value, full_obj_value)

    # And make sure that these values are less than or equal to the min of the linspace eval above.
    assert simple_obj_value <= simple_obj_values.min()
    assert full_obj_value <= full_obj_values.min()
