import torch


def rescale_cov_to_unit_mass(covariance_matrix):
    """
    Produced through this ChatGPT session, modified a bit:
     https://chat.openai.com/share/06106ef0-5a13-4c22-8e7f-5914b48af36f

    Rescale a covariance matrix for a zero-mean multivariate Gaussian distribution
    such that the normalizing constant matches that of a unit Gaussian.

    Parameters:
    covariance_matrix (torch.tensor): A square covariance matrix.

    Returns:
    torch.tensor: The rescaled covariance matrix.
    """

    cm = covariance_matrix

    # Ensure the matrix is a valid covariance matrix: square, symmetric, positive semi-definite
    if cm.shape[0] != cm.shape[1] or not torch.allclose(cm, cm.T):
        raise ValueError("The input must be a square, symmetric matrix.")

    # Compute the determinant of the covariance matrix
    det_covariance = torch.linalg.det(covariance_matrix)

    # Compute the scaling factor such that the determinant of the scaled matrix is 1
    scale_factor = det_covariance ** (-1 / len(covariance_matrix))

    # Scale the covariance matrix
    rescaled_covariance = scale_factor * covariance_matrix

    return rescaled_covariance
