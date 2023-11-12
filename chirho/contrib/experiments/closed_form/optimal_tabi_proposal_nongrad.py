import torch


def optimal_tabi_proposal_nongrad(theta, Q, Sigma):
    # Just the product of two gaussians (where one has zero mean).
    Sigma_star = torch.linalg.inv(torch.linalg.inv(Q) + torch.linalg.inv(Sigma))
    mu_star = Sigma_star @ (torch.linalg.inv(Q) @ theta)

    return mu_star, Sigma_star
