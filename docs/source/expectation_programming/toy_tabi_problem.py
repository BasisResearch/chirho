import torch
import pyro
import pyro.distributions as dist
from collections import OrderedDict
from torch import tensor as tt


def h(d, c):
    # Convert the decision parameter to a convex space with defined maximum (1+c) and minimum (c).
    # This is the root form of a student-t distribution, giving us longer tails and a better ramp up for
    #  the decision parameter gradients.
    return (1. + d**2.)**-1. + c


def cost_part(d, x, c, z: bool):
    # This builds out a part of the cost function. It's a normal positioned in the tails to the degree defined by
    #  c. If z is true, this is the negative part of the cost function in the left tail, otherwise it's the positive
    #  part of the cost function in the right tail. Each part is normal-distribution shaped with a mean of h(d, c)
    #  and h(d, c) - 3 * c, respectively. The variance is fixed at .2**2.
    # numerator = -(x - h(d, c) + 3 * float(z) * c)**2.
    # denominator = torch.tensor(.2**2.)
    # return .5 * torch.exp(numerator / denominator)

    mean = h(d, c) - 3 * float(z) * c
    std = .2
    return torch.exp(dist.Normal(mean, std).log_prob(x))


def cost(d: torch.Tensor, x: torch.Tensor, c: torch.Tensor, **kwargs) -> torch.Tensor:
    # The additional kwargs just takes stochastics and decision parameters that don't actually play into this
    #  cost function. This is required because all decision parameters and all stochastics are unpacked here
    #  as keyword arguments to the cost function, and there are auxiliary stochastics we don't care about.
    return cost_part(d, x, c, False) - cost_part(d, x, c, True)


def q_optimal_normal_guide_mean_var(d, c, z: bool):
    # This is the product of the standard normal and the cost normal pdfs.
    # See e.g. https://www.johndcook.com/blog/2012/10/29/product-of-normal-pdfs/ for more details.
    m = h(d, c) - 3 * float(z) * c
    vp = tt(.2**2.)
    vn = tt(.2**-2)

    new_mean = (vn * m) / (vn + 1)
    new_var = vp / (1 + vp)

    return new_mean, torch.sqrt(new_var)


MODEL_DIST = dist.Normal(0., 1.)


def model():
    # The model is simply a unit normal.
    return OrderedDict(x=pyro.sample("x", MODEL_DIST))


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    xx = torch.linspace(-5., 5., 300)

    c_ = tt(1.5)
    d1_ = tt(2.)
    d2_ = tt(0.)
    d3_ = tt(0.5)

    # Plot the decision parameter re-mapping.
    plt.figure()
    plt.suptitle('Decision Parameter Re-Mapping')
    plt.plot(xx, h(xx, c_))

    # Plot the cost function for the two decision parameters.
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    plt.suptitle('Cost by Model on Stochastics')

    # Plot the cost function for the two decision parameters. Put in the different subplots.
    ax1.plot(xx, cost(d1_, xx, c_), color='red', label=f"d={d1_}")
    ax2.plot(xx, cost(d2_, xx, c_), color='red', label=f"d={d2_}")
    ax3.plot(xx, cost(d3_, xx, c_), color='red', label=f"d={d3_}")

    # Plot the model on top of the cost function.
    ax1.plot(xx, torch.exp(MODEL_DIST.log_prob(xx)), color='blue')
    ax2.plot(xx, torch.exp(MODEL_DIST.log_prob(xx)), color='blue')
    ax3.plot(xx, torch.exp(MODEL_DIST.log_prob(xx)), color='blue')

    # Plot the optimal guides for the two parts of the cost function. These are normal distributions with means and
    #  variances calculated from the product of the model pdf and the cost function "pdf".
    for d_, ax in zip([d1_, d2_, d3_], [ax1, ax2, ax3]):
        gpm, gms = q_optimal_normal_guide_mean_var(d_, c_, False)
        gnm, gns = q_optimal_normal_guide_mean_var(d_, c_, True)
        ax.plot(xx, torch.exp(-((xx - gpm)/gms)**tt(2.))*.4, color='orange', label='Optimal Guide')
        ax.plot(xx, torch.exp(-((xx - gnm)/gns)**tt(2.))*.4, color='orange')

    ax1.legend()
    ax2.legend()
    ax3.legend()

    # In a different plot, show the unnormalized positive component (using torch.relu) of the cost function
    #  multiplied by the model pdf. Also show the negative component (this can be achieved by using torch.relu(-x)).
    # Just do this for d3.
    fig1, ax = plt.subplots()
    tax = ax.twinx()
    tax.plot(xx, torch.exp(MODEL_DIST.log_prob(xx)), color='blue', label=f"d={d3_}", linestyle='--')
    ax.plot(xx, torch.exp(MODEL_DIST.log_prob(xx)) * torch.relu(1e2*cost(d3_, xx, c_)), color='green', label=f"d={d3_}")
    ax.plot(xx, torch.exp(MODEL_DIST.log_prob(xx)) * torch.relu(-1e2*cost(d3_, xx, c_)), color='red', label=f"d={d3_}")

    # Plot just the cost function with a thin line.
    ax.plot(xx, cost(d3_, xx, c_), color='black', linestyle='--', linewidth=0.4)

    # Now plot the properly scaled normal distributions that map to the normalizations of the curves plotted above.
    gpm, gms = q_optimal_normal_guide_mean_var(d3_, c_, False)
    gnm, gns = q_optimal_normal_guide_mean_var(d3_, c_, True)

    ax.plot(xx, torch.exp(dist.Normal(gpm, gms).log_prob(xx)), color='orange', linestyle='--')
    ax.plot(xx, torch.exp(dist.Normal(gnm, gns).log_prob(xx)), color='orange', linestyle='--')

    # In a different figure, show the ratio of the model-scaled positive and negative components of the cost function
    #  with respect to the properly normalized optimal guides. Show this across the same xx.
    fig2, ax = plt.subplots()
    num_pos = torch.exp(MODEL_DIST.log_prob(xx)) * torch.relu(1e2 * cost(d3_, xx, c_))
    den_pos = torch.exp(dist.Normal(gpm, gms).log_prob(xx))
    plt.plot(xx, num_pos / den_pos, color='green', label='Positive Component')

    num_neg = torch.exp(MODEL_DIST.log_prob(xx)) * torch.relu(-1e2 * cost(d3_, xx, c_))
    den_neg = torch.exp(dist.Normal(gnm, gns).log_prob(xx))
    plt.plot(xx, num_neg / den_neg, color='red', label='Negative Component')

    plt.show()
