# HACK when running osx-64 conda env on M2 mac. Side effects could be an issue:
# https://stackoverflow.com/a/53014308
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from torch import tensor as tt
import src.fenics_overloaded as fe
import torch_fenics

from src.halfar_ice.analytical import halfar_ice_analytical, t0f
from src.halfar_ice.utils import fillna
from src.halfar_ice.numeric_nonlinear import HalfarIceNonLinearPolar
from src.halfar_ice.utils import stable_gamma


def anamain():
    h0 = 1.
    r0 = 1.

    xx = torch.linspace(0., 2., 512)

    sols = []

    t = 1.
    tstep = 1.0
    for i in range(1000):
        h = halfar_ice_analytical(xx, t, h0, r0)
        sols.append(h)
        t += tstep

    fig, ax = plt.subplots(dpi=500, figsize=(4, 2))
    plot_sols(xx, sols, ax, skip=5)

    # Sum across each solution to check if mass is being conserved.
    # FIXME for some reason this was sigsegv-ing :/
    # ice_sum = torch.stack(sols).sum(axis=-1)
    ice_sum = [(sol * xx.abs()).sum() / len(sol) for sol in sols]
    print(min(ice_sum), max(ice_sum), max(ice_sum) - min(ice_sum), len(xx))
    plt.figure()
    plt.plot(ice_sum)


def plot_sols(xx, sols, ax, skip=5, color1='purple', color2='orange', thickness1=1.0, thickness2=0.1):

    ax.plot(xx, sols[0].detach(), color=color1, linestyle='--', label="Initial Condition", linewidth=thickness1)
    for t_sols in sols[1:-1:skip]:
        ax.plot(xx, t_sols.detach(), color=color2, alpha=0.5, linewidth=thickness2)
    ax.plot(xx, sols[-1].detach(), color=color1, label="Final Solution", linewidth=thickness1)


def main():

    halfar_ice = HalfarIceNonLinearPolar(511)

    xx = torch.linspace(-2., 2., halfar_ice.n_elements + 1)

    ice_density = tt(910.).double()
    log_flow_rate = torch.log(tt(1e-16)).double()

    # # A tricky initial glacier.
    # u_init = initial_curve(xx)\
    #     [None, :].double().clone().detach().requires_grad_(True)  # Make this a leaf tensor.

    # Init with analytically known progression in time.
    h0, r0 = 1., 1.
    gamma = stable_gamma(rho=ice_density, lA=log_flow_rate)
    t0 = t0f(r0, h0, gamma)
    u_init = halfar_ice_analytical(
        r=xx, t=t0, h0=h0, r0=r0, gamma=gamma)[None, :] \
        .double().clone().detach().requires_grad_(True)

    u_last_t = u_init

    plt.show()

    sols = [u_last_t[0]]

    tstep = tt(1000).double()[None, None]
    end_t = t0
    for i in range(300):
        if len(sols) > 1:
            change = torch.linalg.norm(sols[-1] - sols[-2])
            change = change.item()
        else:
            change = torch.inf

        print(f"\rIteration {i:05d}; Div {change:6.6f}", end='')

        u_last_t = halfar_ice(tstep, u_last_t, *torch.atleast_2d(ice_density, log_flow_rate))

        end_t += tstep.item()

        sols.append(u_last_t[0])
    print()

    # FIXME these plots are reversed across x for some reason. Run heat_conduction_simple.py and compare.
    #  Also see that initial condition is reversed.
    # sols = sols[:, :, ::-1]

    fig, ax = plt.subplots(dpi=500, figsize=(4, 2))
    plot_sols(xx, sols, ax, skip=5)

    analytical_solution_maybe = halfar_ice_analytical(xx, t=end_t, h0=h0, r0=r0, gamma=gamma)
    plt.plot(xx, analytical_solution_maybe, color='white', linestyle='--',
             linewidth=0.5, label="Analytical Equilibrium (Maybe)")
    plt.gca().set_aspect('equal')
    plt.gca().set_facecolor('lightgray')
    plt.gcf().set_facecolor('lightgray')

    plt.tight_layout()

    plt.show()

    print(f"Glacier Mass Before", sols[0].sum().item())
    print(f"Glacier Mass After (Num)", sols[-1].sum().item())
    print(f"Glacier Mass After (Ana)", analytical_solution_maybe.sum().item())

    print(f"Glacier Mass Adj Before", (xx.abs() * sols[0]).sum().item())
    print(f"Glacier Mass Adj After (Ana)", (xx.abs() * analytical_solution_maybe).sum().item())

    # Now, make sure we get proper gradients. Get the gradient of the final solution just
    #  to the left of the center with respect to the initial condition.
    pio_idx = 230
    poi = sols[-1][pio_idx]
    poi.backward()
    if u_init.grad is None:
        print("No gradient")
    else:
        plt.figure()
        plt.plot(xx, u_init.grad[0].detach().numpy())
        # Plot a vertical line on the point of interest.
        plt.axvline(xx[pio_idx].item(), color='red', linestyle='--')
        # And plot thin initial and final conditions on the right axis.
        tax = plt.twinx()
        tax.plot(xx, u_init[0].detach().numpy(), color='gray', linestyle='--')
        tax.plot(xx, sols[-1].detach().numpy(), color='gray', linestyle='--')


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()
    # anamain()

    plt.show()
