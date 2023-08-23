import pyro
import torch
from torch import tensor as tt


class FillReluAtLevelExp(pyro.poutine.messenger.Messenger):

    def __init__(self, beta):
        self.beta = beta

        self._c = torch.exp(tt(-1.))

    # @torch.compile
    def _srelu_compiled(self, x):
        """
        An experimental temperature based relaxation of the relu that maintains a non-negative
        range.
        See https://www.desmos.com/calculator/vnjgtf4f6l for more details.
        """

        c = self._c

        x = x / self.beta

        inflection_point = -torch.log(c) / c
        bounded_x = torch.where(x < inflection_point, x, tt(0.0))
        lt_part = torch.exp(c * bounded_x)
        gt_part = x + (1. / c) * (torch.log(c) + 1)
        y = torch.where(x < inflection_point, lt_part, gt_part)

        assert not torch.any(torch.isnan(y))

        return self.beta * y

    def _pyro_srelu(self, msg) -> None:
        msg["value"] = self._srelu_compiled(msg["args"][0])


if __name__ == "__main__":
    from chirho.contrib.compexp.ops import srelu
    import matplotlib.pyplot as plt

    xx = torch.linspace(-50., 30., 1000)
    px = torch.nn.Parameter(tt(0.0))
    with FillReluAtLevelExp(beta=tt(5.0)):
        yy = srelu(xx)
        yn = srelu(xx[0])
        yq = srelu(xx[500])
        yp = srelu(xx[-1])

        py = srelu(px)

    print(yn, yq, yp)

    with FillReluAtLevelExp(beta=tt(5.0)):
        grad_x = []
        for x_ in xx:
            px.data = x_
            py = srelu(px)
            grad_x.append(torch.autograd.grad(py, px)[0].item())

    # Get fig and ax.
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    tax = ax.twinx()
    tax.plot(xx, grad_x, linestyle='-.', label='d/dx srelu(beta=5.0)')

    ax.plot(xx, srelu(xx), linestyle='-', label='relu')
    ax.plot(xx, yy, linestyle='--', label='srelu(beta=5.0)')

    # Make a legend with labels from both ax and tax.
    lines, labels = ax.get_legend_handles_labels()
    tlines, tlabels = tax.get_legend_handles_labels()
    tax.legend(lines + tlines, labels + tlabels, loc='upper left')

    plt.show()
