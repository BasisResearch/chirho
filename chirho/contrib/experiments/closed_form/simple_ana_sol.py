import torch


# Define a function to compute the analytic solution for r given q, c, and n.
def compute_ana_rstar(q, c, n):
    """
    Derived through this ChatGPT session:
      https://chat.openai.com/share/a5ca8fa1-8866-4640-be7d-391ba491e13b
    See .objectives.simple_ana_obj for details. This is the analytical solution r that maximizes that objective.
    """

    # This is the constant k from the determinant term
    k = torch.sqrt((1 / (1/q + 1))**n)
    # Compute the analytic solution for r
    r = torch.sqrt(-2 * (q + 1) * torch.log((2 * (q + 1)) / (c * k)))

    return r


def compute_ana_c(q, rstar, n):
    """
    A sort of inverse of the above. This specifies an r and then returns a risk scaling factor such that the maximizing
     solution r is the specified r.
    See this desmos plot for details/verification: https://www.desmos.com/calculator/5noaryuluz
    Also this ChatGPT session:
       https://chat.openai.com/share/a5ca8fa1-8866-4640-be7d-391ba491e13b
    """

    q2p1 = 2 * (q + 1)
    exponential_term = - rstar ** 2 / q2p1
    k = torch.sqrt((1 / (1 / q + 1)) ** n)
    return q2p1 / (k * torch.exp(exponential_term))
