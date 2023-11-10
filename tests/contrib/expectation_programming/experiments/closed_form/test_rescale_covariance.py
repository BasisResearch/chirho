import pytest
import torch
from chirho.contrib.experiments.closed_form import rescale_cov_to_unit_mass

# Produced through this ChatGPT session, modified a bit:
#  https://chat.openai.com/share/06106ef0-5a13-4c22-8e7f-5914b48af36f


# Define a set of valid covariance matrices for different dimensions
@pytest.mark.parametrize("covariance_matrix", [
    torch.tensor([[2, 0.5], [0.5, 1]]),
    torch.tensor([[1, 0.1, 0.2], [0.1, 2, 0.3], [0.2, 0.3, 3]]),
    torch.tensor([[4, 0.1, 0.2, 0.3], [0.1, 3, 0.4, 0.5], [0.2, 0.4, 2, 0.6], [0.3, 0.5, 0.6, 1]])
])
def test_rescale_cov_to_unit_mass(covariance_matrix):
    """
    Test that the rescale_cov_to_unit_mass function correctly rescales a covariance matrix
    such that the normalizing constant of the resulting Gaussian is equal to that of the unit Gaussian.
    """
    # TODO note that this doesn't actually test that it's just a rescaling. Also could make sure we can recover the
    #  original with a scalar multiplication.

    d = covariance_matrix.shape[0]
    pi = torch.tensor(torch.pi)

    # Assert that the original determinant is not equal to 1. This just assures we aren't getting false positives.
    original_det = torch.linalg.det(covariance_matrix)
    assert not torch.isclose(original_det, torch.tensor(1.0))

    # Calculate the normalizing constant for the unit Gaussian of dimension d
    normalizing_constant_unit = (2 * pi) ** (d / 2)

    # Calculate the normalizing constant for the given covariance matrix, again checking for false positives.
    normalizing_constant_original = (2 * pi) ** (d / 2) * original_det ** 0.5
    assert not torch.isclose(normalizing_constant_original, normalizing_constant_unit)

    # Rescale the covariance matrix using the function
    rescaled_cov_matrix = rescale_cov_to_unit_mass(covariance_matrix)

    # Calculate the determinant of the rescaled covariance matrix
    rescaled_det = torch.linalg.det(rescaled_cov_matrix)

    # Calculate the normalizing constant for the rescaled covariance matrix
    normalizing_constant_rescaled = (2 * pi) ** (d / 2) * rescaled_det ** 0.5

    # Assert that the normalizing constant of the rescaled covariance is equal to that of the unit Gaussian
    assert torch.isclose(normalizing_constant_rescaled, normalizing_constant_unit)
