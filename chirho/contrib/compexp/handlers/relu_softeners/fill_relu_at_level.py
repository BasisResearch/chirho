import pyro
import torch
from torch import tensor as tt


class FillReluAtLevel(pyro.poutine.messenger.Messenger):
    """
    Softens a relu by "filling" in the sharp corner of the relu with
     a quadratic function. This was derived by rotating a quadratic
     filling of a rescaled |x| such that it corresponds directly
     to an angled filling of the relu.
    See https://www.desmos.com/calculator/sp6n3eizt0 for a visualization
     of the quadratic filling.
    See https://chat.openai.com/share/d47c9731-e7aa-4110-817c-57b4d435c11d for
     the chatgpt assisted derivation of the rotated quadratic.
    """

    def __init__(self, beta):
        self.beta = beta

        if torch.isclose(self.beta, tt(0.)) or torch.lt(self.beta, tt(0.)):
            raise ValueError("Beta must be greater than zero.")

        # Re-useable constants.
        self._a = torch.pi / tt(8.0) - torch.arctan(tt(1.0))
        self._tan_pi_8 = torch.tan(torch.pi / tt(8.0))
        self._cos_a = torch.cos(self._a)
        self._sin_a = torch.sin(self._a)
        self._sqrt_2 = torch.sqrt(tt(2.0))
        self._sqrt_2_minus_sqrt_2 = torch.sqrt(tt(2.0) - self._sqrt_2)
        self._4_minus_3_sqrt_2 = tt(4.0) - tt(3.0) * self._sqrt_2

    def _symmetric_absx(self, x):
        return self._tan_pi_8 * torch.abs(x)

    def _pyro_srelu(self, msg) -> None:
        x = msg["args"][0]

        b = self.beta

        # Negative boundary of quadratic piece.
        nb = -b * self._cos_a + self._symmetric_absx(b) * self._sin_a

        # Positive boundary of quadratic piece.
        pb = b * self._cos_a + self._symmetric_absx(b) * self._sin_a

        # Define mask for the conditions
        mask_lt_nb = x < nb
        mask_between_nb_pb = (x >= nb) & (x <= pb)
        mask_gt_pb = x > pb

        # Calculate the two main components of the solution only for the valid range
        term1 = torch.zeros_like(x)
        term2 = torch.zeros_like(x)
        term1[mask_between_nb_pb] = 4 * torch.sqrt(
            b * (2 * b - x[mask_between_nb_pb] * self._sqrt_2_minus_sqrt_2)) * torch.sqrt(
            -1 + self._sqrt_2) / self._4_minus_3_sqrt_2
        term2[mask_between_nb_pb] = torch.sqrt(self._sqrt_2 + 2) * (
            2 * b - self._sqrt_2 * x[mask_between_nb_pb] * self._sqrt_2_minus_sqrt_2 +
            x[mask_between_nb_pb] * self._sqrt_2_minus_sqrt_2) / self._4_minus_3_sqrt_2

        # Compute the result based on the masks
        result = torch.zeros_like(x)
        result[mask_lt_nb] = tt(0.0)
        result[mask_between_nb_pb] = term1[mask_between_nb_pb] - term2[mask_between_nb_pb]
        result[mask_gt_pb] = x[mask_gt_pb]

        msg["value"] = result


if __name__ == "__main__":
    from chirho.contrib.compexp.ops import srelu
    import matplotlib.pyplot as plt

    xx = torch.linspace(-10., 10., 1000)
    with FillReluAtLevel(beta=tt(5.0)):
        yy = srelu(xx)

    plt.plot(xx, torch.relu(xx))
    plt.plot(xx, yy, linestyle='--')
    plt.show()
