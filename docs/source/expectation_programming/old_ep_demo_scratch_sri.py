from sri_tabi_problem import ReuseableSimulation, LOCKDOWN_TYPE, LockdownType
from torch import tensor as tt

import pyro
import torch
from pyro.infer.autoguide import AutoMultivariateNormal
import pyro.distributions as dist
import matplotlib.pyplot as plt

from enum import Enum

from chirho.dynamical.ops import State, Trajectory

from collections import OrderedDict

from typing import (
    Optional,
    Union,
)

import numpy as np

import old_ep_demo_scratch as stor

from sklearn.neighbors import KernelDensity

PT = torch.nn.Parameter
TT = torch.Tensor
UPTT = Union[PT, TT]

if LOCKDOWN_TYPE == LockdownType.NONCONT_STATE:
    DEFAULT_DPARAMS = DDP = OrderedDict(
        # # Optimized.
        # lockdown_trigger=tt(0.176),
        # lockdown_lift_trigger=tt(0.457),
        # lockdown_strength=tt(0.656)

        # lockdown_trigger=tt(0.23),
        # lockdown_lift_trigger=tt(0.45),
        # lockdown_strength=tt(0.69)

        lockdown_trigger=torch.nn.Parameter(tt(0.1)),
        lockdown_lift_trigger=torch.nn.Parameter(tt(0.32)),
        lockdown_strength=torch.nn.Parameter(tt(0.7))
    )
elif LOCKDOWN_TYPE == LockdownType.CONT_PLATEAU:
    DEFAULT_DPARAMS = DDP = OrderedDict(
        # Optimal-ish for continouous plateau setup.
        lockdown_trigger=tt(0.03),
        lockdown_lift_trigger=tt(0.49),
        lockdown_strength=tt(0.61)

        # # Sub-optimal but decent init for continuous plateau setup.
        # lockdown_trigger=tt(0.08),
        # lockdown_lift_trigger=tt(0.6),
        # lockdown_strength=tt(0.5)
    )
elif LOCKDOWN_TYPE == LockdownType.NONCONT_TIME:
    DEFAULT_DPARAMS = DDP = OrderedDict(
        lockdown_trigger=tt(0.8),
        lockdown_lift_trigger=tt(8.0),
        lockdown_strength=tt(0.5)
    )

DEFAULT_INIT_STATE = DIS = State(S=tt(0.99), I=tt(0.01), R=tt(0.0), L=tt(0.0), l=tt(0.0), O=tt(0.0))

DEFAULT_STOCHASTICS = DST = OrderedDict(
    beta=tt(2.),
    gamma=tt(.4),
    capacity=tt(0.01),
    hospitalization_rate=tt(0.05)
)

DEFAULT_TIMES = DT = torch.linspace(0., 20., 100)

if LOCKDOWN_TYPE == LockdownType.NONCONT_STATE:
    OEXPO = 1.
    OSCALING = 2e2
elif LOCKDOWN_TYPE == LockdownType.NONCONT_TIME:
    OEXPO = 1.
    OSCALING = 2e2
elif LOCKDOWN_TYPE == LockdownType.CONT_PLATEAU:
    OSCALING = 1.3e2
    OEXPO = 1.5


def o_transform(o: torch.Tensor) -> torch.Tensor:
    return ((1. + o) ** OEXPO - 1.) * OSCALING


def copy_odict(odict: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((k, tt(v.item())) for k, v in odict.items())


def f_traj(
        dparams: stor.KWTypeNNParams,
        stochastics: stor.KWType,
        rs: Optional[ReuseableSimulation] = None) -> Trajectory[torch.Tensor]:

    # If dparams or stochastics are missing keys defined in the defaults, fill them in with copies of the default
    #  values. Also wrap lockdown triggers in their appropriate state objects.
    dparams = copy_odict(DEFAULT_DPARAMS) | dparams
    stochastics = copy_odict(DEFAULT_STOCHASTICS) | stochastics

    if rs is None:
        rs = ReuseableSimulation()

    for k, v in stochastics.items():
        # All stochastics need to be positive, so just do it here and not worry about it in the guide.
        stochastics[k] = torch.abs(v)

    for k, v in dparams.items():
        # All dparams need to be positive and non-zero, so put them through a relu with eps added.
        dparams[k] = torch.relu(v) + 1e-3

    traj = rs(
        **dparams,
        init_state=DEFAULT_INIT_STATE,
        **stochastics,
        times=DEFAULT_TIMES)

    return traj


def f_combined(dparams: stor.KWTypeNNParams, stochastics: stor.KWType) -> stor.KWType:
    traj = f_traj(dparams, stochastics)

    total_hospital_overrun = traj[-1].O
    total_lockdown_unpleasantness = traj[-1].L

    # For now, just combine these into a single cost. In future we want to minimize lockdown with constraints
    #  on total hospital overrun.
    return OrderedDict(cost=o_transform(total_hospital_overrun) + total_lockdown_unpleasantness)


def f_o_only(
        dparams: stor.KWTypeNNParams,
        stochastics: stor.KWType,
        # TODO lo1dop6k This avoids redundancy in the cost and constraint calls, but isn't used right now.
        rs: Optional[ReuseableSimulation] = None) -> stor.KWType:
    traj = f_traj(dparams, stochastics, rs=rs)

    total_hospital_overrun = traj[-1].O

    return OrderedDict(cost=o_transform(total_hospital_overrun))


def f_l_only(
        dparams: stor.KWTypeNNParams,
        stochastics: stor.KWType,
        # TODO lo1dop6k See tag above.
        rs: Optional[ReuseableSimulation] = None) -> stor.KWType:
    traj = f_traj(dparams, stochastics, rs=rs)

    total_lockdown_unpleasantness = traj[-1].L

    return OrderedDict(cost=total_lockdown_unpleasantness)


def plot_basic(dparams=None, stochastics=None):

    if dparams is None:
        dparams = OrderedDict()
    if stochastics is None:
        stochastics = OrderedDict()

    traj = f_traj(dparams, stochastics)

    fig, (ax1, ax3, ax2) = plt.subplots(3, 1, figsize=(7, 10))
    tax3 = ax3.twinx()
    ax2.axhline(DST['capacity'] * (1. / DST['hospitalization_rate']), color='k', linestyle='--')
    ax3.axhline(DST['capacity'] * (1. / DST['hospitalization_rate']), color='k', linestyle='--',
                label='Healthcare Capacity')
    ax2.plot(DT, traj.S, label='S', color='blue')
    ax2.plot(DT, traj.I, label='I', color='red')
    ax3.plot(DT, traj.I, label='I', color='red')
    ax2.plot(DT, traj.R, label='R', color='green')
    ax1.plot(DT, traj.L, label='Aggregate Lockdown', color='orange')
    ax1.plot(DT, traj.l, label='Lockdown', color='orange', linestyle='--')
    tax3.plot(DT, traj.O, label='Aggregate Overrun', color='k')
    ax1.legend()
    ax2.legend()
    ax3.legend()
    tax3.legend()

    ax2.set_xlabel('Time')
    ax1.set_ylabel('Lockdown Strength')
    ax2.set_ylabel('Proportion of Population')
    ax3.set_ylabel('Proportion of Population')
    tax3.set_ylabel('Aggregate Overrun')

    plt.tight_layout()

    plt.show()

    return


def plot_cost_vs_parameter(parameter_name, center: bool = False):

    if center:
        c = DDP[parameter_name]
        parameter_values = torch.linspace(torch.relu(c - 0.05), c + 0.05, 25)
    else:
        # Define the range of parameter values to consider
        parameter_values = torch.linspace(0.01, 1.0, 100)

    # Initialize empty lists to store the corresponding cost values
    cost_values = []
    o_only_values = []
    l_only_values = []

    # Loop over the parameter values
    for parameter_value in parameter_values:
        # Calculate the cost for this parameter value
        o_only = f_o_only(OrderedDict({parameter_name: parameter_value}), OrderedDict())['cost']
        l_only = f_l_only(OrderedDict({parameter_name: parameter_value}), OrderedDict())['cost']
        cost = f_combined(OrderedDict({parameter_name: parameter_value}), OrderedDict())['cost']

        o_only_values.append(o_only.item())
        l_only_values.append(l_only.item())
        cost_values.append(cost.item())

    # Create the plot
    fig, ax = plt.subplots(1, 1)
    ax.plot(parameter_values, cost_values)
    ax.plot(parameter_values, o_only_values)
    ax.plot(parameter_values, l_only_values)
    ax.set_xlabel(parameter_name.capitalize())
    ax.set_ylabel('Cost')
    ax.legend(['Combined', 'Overrun Only', 'Lockdown Only'])
    ax.grid(True)

    plt.show()


def plot_cost_vs_parameters(parameter_name1, parameter_name2):
    # Define the range of parameter values to consider
    parameter_values1 = torch.linspace(0.01, 1.0, 10)  # Adjust the number of points as needed
    parameter_values2 = torch.linspace(0.01, 1.0, 10)  # Adjust the number of points as needed

    # Create a meshgrid of parameter values
    parameter_grid1, parameter_grid2 = torch.meshgrid(parameter_values1, parameter_values2)

    # Initialize empty tensors to store the corresponding cost values
    cost_values = torch.zeros_like(parameter_grid1)
    o_only_values = torch.zeros_like(parameter_grid1)
    l_only_values = torch.zeros_like(parameter_grid1)

    # Loop over the parameter values
    for i in range(parameter_values1.shape[0]):
        for j in range(parameter_values2.shape[0]):
            # Calculate the cost for this pair of parameter values
            o_only = f_o_only(OrderedDict({parameter_name1: parameter_grid1[i, j], parameter_name2: parameter_grid2[i, j]}), OrderedDict())['cost']
            l_only = f_l_only(OrderedDict({parameter_name1: parameter_grid1[i, j], parameter_name2: parameter_grid2[i, j]}), OrderedDict())['cost']
            cost = f_combined(OrderedDict({parameter_name1: parameter_grid1[i, j], parameter_name2: parameter_grid2[i, j]}), OrderedDict())['cost']

            o_only_values[i, j] = o_only.item()
            l_only_values[i, j] = l_only.item()
            cost_values[i, j] = cost.item()

    # Create the contour plots
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    axs[0].contourf(parameter_grid1.numpy(), parameter_grid2.numpy(), o_only_values.numpy(), cmap='viridis')
    axs[0].set_xlabel(parameter_name1.capitalize())
    axs[0].set_ylabel(parameter_name2.capitalize())
    axs[0].set_title('Overrun Cost')
    fig.colorbar(ax=axs[0], mappable=axs[0].collections[0], label='Cost')

    axs[1].contourf(parameter_grid1.numpy(), parameter_grid2.numpy(), l_only_values.numpy(), cmap='viridis')
    axs[1].set_xlabel(parameter_name1.capitalize())
    axs[1].set_ylabel(parameter_name2.capitalize())
    axs[1].set_title('Lockdown Cost')
    fig.colorbar(ax=axs[1], mappable=axs[1].collections[0], label='Cost')

    axs[2].contourf(parameter_grid1.numpy(), parameter_grid2.numpy(), cost_values.numpy(), cmap='viridis')
    axs[2].set_xlabel(parameter_name1.capitalize())
    axs[2].set_ylabel(parameter_name2.capitalize())
    axs[2].set_title('Combined Cost')
    fig.colorbar(ax=axs[2], mappable=axs[2].collections[0], label='Cost')

    plt.tight_layout()
    plt.show()


def _NNM_vectorized(f_, N, M, X, Y):
    out = np.empty((N, N, M))
    it = np.nditer(out, flags=['multi_index'])
    while not it.finished:
        out[*it.multi_index[:-1], :] = f_(X[it.multi_index[:-1]], Y[it.multi_index[:-1]])
        it.iternext()

    return out


def plot_cost_likelihood_convolution_for_stochastics(
        stochastic_name1: str, stochastic_name2: str,
        p: stor.ModelType, f_: stor.ExpectigrandType, n=1000):

    samples = []
    for _ in range(n):  # Generate 1000 samples
        sample = p()
        samples.append((sample[stochastic_name1].item(), sample[stochastic_name2].item()))

    samples = np.array(samples)

    kde: KernelDensity = KernelDensity(kernel='gaussian', bandwidth=0.02).fit(samples)

    def density(s1, s2) -> np.ndarray:
        return np.array([np.exp(kde.score_samples([[s1, s2]]))])

    def cost(s1, s2) -> np.ndarray:
        vals = f_(OrderedDict({stochastic_name1: tt(s1), stochastic_name2: tt(s2)})).values()
        return np.array(tuple(v.item() for v in vals))

    def cust_colorbar(ax, arr):
        ticks = np.linspace(0., 1.0, 10)
        cbar = fig.colorbar(ax=ax, mappable=ax.collections[0], ticks=ticks)
        arrmax = arr.max()
        arrmin = arr.min()
        tick_labels = ticks * (arrmax - arrmin) + arrmin
        cbar.ax.set_yticklabels(['{:.1f}'.format(tick) for tick in tick_labels])

    resolution = 15

    s1ls = np.linspace(0.00, 1.0, resolution)
    s2ls = np.linspace(0.00, 1.0, resolution)

    s1ls = s1ls * (samples[:, 0].max() - samples[:, 0].min()) + samples[:, 0].min()
    s2ls = s2ls * (samples[:, 1].max() - samples[:, 1].min()) + samples[:, 1].min()

    X, Y = np.meshgrid(s1ls, s2ls)

    # Make a subplot for the density and one for each component of the cost.
    cost_component_names = f_(OrderedDict({stochastic_name1: tt(s1ls[0]), stochastic_name2: tt(s2ls[0])})).keys()
    num_cost_components = len(cost_component_names)

    fig, axs = plt.subplots(num_cost_components + 1, 3, figsize=(6*num_cost_components, 18))

    # Plot the density in the top row.
    density_array = _NNM_vectorized(f_=density, N=resolution, M=1, X=X, Y=Y)
    for col in range(3):
        axs[0][col].contourf(X, Y, density_array[..., 0], cmap='viridis')
        cust_colorbar(axs[0][col], density_array[..., 0])

    # Compute the cost array. We have to do this manually because there
    cost_array = _NNM_vectorized(f_=cost, N=resolution, M=num_cost_components, X=X, Y=Y)

    assert cost_array.shape == (resolution, resolution, num_cost_components)

    # Plot the cost components in the left column.
    for i, cost_component_name in enumerate(cost_component_names):
        axs[i+1][0].contourf(X, Y, cost_array[:, :, i], cmap='viridis')
        axs[i+1][0].set_title(cost_component_name.capitalize())
        cust_colorbar(axs[i+1][0], cost_array[:, :, i])

    # This maybe doesn't give the right shape.
    cost_density_convolution_array = cost_array * density_array
    assert cost_density_convolution_array.shape == (resolution, resolution, num_cost_components)

    # Plot the positive part of the convolved cost components in the middle column.

    def plot_part(arr, col):
        for i, cost_component_name in enumerate(cost_component_names):

            axs[i+1][col].contourf(X, Y, arr[:, :, i], cmap='viridis')
            axs[i+1][col].set_title(cost_component_name + (' +' if col == 1 else ' -'))
            cust_colorbar(axs[i+1][col], arr[:, :, i])

            # And draw crosshairs at the original density mean, for comparison.
            axs[i+1][col].axvline(x=samples[:, 0].mean(), color='white', linestyle='--')
            axs[i+1][col].axhline(y=samples[:, 1].mean(), color='white', linestyle='--')

    plot_part(np.maximum(cost_density_convolution_array, 0.0), 1)
    plot_part(-np.minimum(cost_density_convolution_array, 0.0), 2)

    plt.tight_layout()

    plt.show()


def pyro_prior_over_sirlo_params():
    beta = pyro.sample(
        "beta", dist.Normal(2., 0.3))
    gamma = pyro.sample(
        "gamma", dist.Normal(.4, .06))
    capacity = pyro.sample(
        "capacity", dist.Normal(0.01, 0.003))
    hospitalization_rate = pyro.sample(
        "hospitalization_rate", dist.Normal(0.05, 0.015))
    return OrderedDict(
        beta=beta,
        gamma=gamma,
        capacity=capacity,
        hospitalization_rate=hospitalization_rate
    )


def pyro_prior_over_sirlo_params_2d():
    beta = pyro.sample(
        "beta", dist.Normal(2., 0.1))
    gamma = pyro.sample(
        "gamma", dist.Normal(.4, .02))
    return OrderedDict(
        beta=beta,
        gamma=gamma
    )


def _grad_debugging():
    dparams = OrderedDict(
        lockdown_trigger=torch.nn.Parameter(tt(DDP['lockdown_trigger'].item())),
        lockdown_lift_trigger=torch.nn.Parameter(tt(DDP['lockdown_lift_trigger'].item())),
        lockdown_strength=torch.nn.Parameter(tt(DDP['lockdown_strength'].item()))
    )

    overrun = f_o_only(dparams, OrderedDict())
    lockdown = f_l_only(dparams, OrderedDict())
    combined = f_combined(dparams, OrderedDict())

    traj = f_traj(dparams, OrderedDict())

    # Make sure the gradient of the lockdown trigger end state wrt the param is one.
    dltdlt = torch.autograd.grad(
        outputs=(traj.lockdown_trigger[-1],),
        inputs=tuple(dparams.values()),
        create_graph=True)
    assert torch.isclose(dltdlt[0], tt(1.0))
    assert torch.isclose(dltdlt[1], tt(0.0))
    assert torch.isclose(dltdlt[2], tt(0.0))

    # Make sure the gradient of the lockdown lift trigger end state wrt the param is zero.
    dlftdlft = torch.autograd.grad(
        outputs=(traj.lockdown_lift_trigger[-1],),
        inputs=tuple(dparams.values()),
        create_graph=True)
    assert torch.isclose(dlftdlft[0], tt(0.0), atol=1e-4)
    assert torch.isclose(dlftdlft[1], tt(1.0))
    assert torch.isclose(dlftdlft[2], tt(0.0))

    if LOCKDOWN_TYPE != LockdownType.CONT_PLATEAU:
        # Make sure the gradient of the lockdown strength wrt itself is one.
        dlstdlst = torch.autograd.grad(
            outputs=(traj.l[30],),  # get strength when lockdown is active.
            inputs=tuple(dparams.values()),
            create_graph=True)
        assert torch.isclose(dlstdlst[0], tt(0.0))
        assert torch.isclose(dlstdlst[1], tt(0.0))
        assert torch.isclose(dlstdlst[2], tt(1.0))

    dCd = torch.autograd.grad(
        outputs=(combined["cost"],),
        inputs=tuple(dparams.values()),
        # inputs=(dparams['lockdown_strength'],),
        # inputs=(dparams['lockdown_trigger'],),
        # inputs=(dparams['lockdown_lift_trigger'],),
        create_graph=True)

    assert not torch.isclose(dCd[0], tt(0.0), atol=1e-4)
    assert not torch.isclose(dCd[1], tt(0.0), atol=1e-4)
    assert not torch.isclose(dCd[2], tt(0.0), atol=1e-4)


class ConstraintType(Enum):
    JOINT = 1
    MEAN = 2


def optimize_decision_2d_latent(constraint_type: ConstraintType):

    # Beta and gamma are the only latents.
    p_ = pyro_prior_over_sirlo_params_2d

    dparams = OrderedDict(
        lockdown_trigger=torch.nn.Parameter(tt(0.1)),
        lockdown_lift_trigger=torch.nn.Parameter(tt(0.3)),
        lockdown_strength=torch.nn.Parameter(tt(0.7))
    )

    q_plus_guide = AutoMultivariateNormal(
        model=p_,
    )
    q_minus_guide = AutoMultivariateNormal(
        model=p_
    )
    q_den_guide = AutoMultivariateNormal(
        model=p_
    )

    te = stor.TABIExpectation(
        q_plus=q_plus_guide,
        q_minus=q_minus_guide,
        q_den=q_den_guide,
        num_samples=1
    )

    def abort_guide_grads_(*parameters: torch.nn.Parameter):
        # These gradients also blow up, but clipping them causes weird non-convergence. Just aborting
        #  the gradient update seems to work.
        if torch.any(torch.tensor([torch.any(torch.abs(param.grad) > 400.) for param in parameters])):
            for param in parameters:
                param.grad = torch.zeros_like(param.grad)

    dh = stor.DecisionOptimizerHandler(
        dparams=dparams,
        lr=1e-1,
        proposal_update_lr=1e-4,
        # TODO restore this once the constraints can get dedicated proposals.
        proposal_update_steps=10 if constraint_type == ConstraintType.JOINT else 0,
        proposal_adjust_grads_=abort_guide_grads_
    )

    mc = stor.MeanConstraintHandler(
        g=f_o_only,
        tau=1.,
        threshold=0.001
    ) if constraint_type == ConstraintType.MEAN else pyro.poutine.messenger.Messenger()

    def terminal_condition(_, i: int) -> bool:
        return i > 1000  # TODO more sophisticated convergence criterion.

    betagammas_q_plus_progression = []
    betagammas_q_minus_progression = []
    betagammas_q_den_progression = []
    lockdown_strength_progression = []
    lockdown_trigger_progression = []
    lockdown_lift_trigger_progression = []

    def save_progressions():
        np.savez(
            f"/Users/azane/Desktop/sirlo_logs/sirlo_opt.npz",
            lockdown_strength_progression=np.array(lockdown_strength_progression),
            lockdown_trigger_progression=np.array(lockdown_trigger_progression),
            lockdown_lift_trigger_progression=np.array(lockdown_lift_trigger_progression),
            beta_q_plus_progression=np.array([bg['beta'].detach().item() for bg in betagammas_q_plus_progression]),
            gamma_q_plus_progression=np.array([bg['gamma'].detach().item() for bg in betagammas_q_plus_progression]),
            beta_q_minus_progression=np.array([bg['beta'].detach().item() for bg in betagammas_q_minus_progression]),
            gamma_q_minus_progression=np.array([bg['gamma'].detach().item() for bg in betagammas_q_minus_progression]),
            beta_q_den_progression=np.array([bg['beta'].detach().item() for bg in betagammas_q_den_progression]),
            gamma_q_den_progression=np.array([bg['gamma'].detach().item() for bg in betagammas_q_den_progression]),
        )

    class OptimizeProposalCallback:

        init_optimize_proposal_iterations = 0

        def __call__(self):
            self.init_optimize_proposal_iterations += 1
            print(f"Optimizing proposal {self.init_optimize_proposal_iterations}.")

            betagammas_q_plus_progression.append(q_plus_guide.forward())
            betagammas_q_minus_progression.append(q_minus_guide.forward())
            betagammas_q_den_progression.append(q_den_guide.forward())

            save_progressions()

    class OptimizeDecisionCallback:
        optimize_decision_iterations = 0

        def __call__(self):
            self.optimize_decision_iterations += 1
            print(f"Optimizing decision {self.optimize_decision_iterations}.")

            lockdown_strength_progression.append(dh.dparams['lockdown_strength'].detach().item())
            lockdown_trigger_progression.append(dh.dparams['lockdown_trigger'].detach().item())
            lockdown_lift_trigger_progression.append(dh.dparams['lockdown_lift_trigger'].detach().item())

            betagammas_q_plus_progression.append(q_plus_guide.forward())
            betagammas_q_minus_progression.append(q_minus_guide.forward())
            betagammas_q_den_progression.append(q_den_guide.forward())

            save_progressions()

    with te, dh, mc:

        # # This changes what optimize_decision sees as the stochastic program to be the product of a conditioned
        # #  inference procedure.
        # with pyro.condition(sir_noisy_data):

        # TODO enable for explicit constraints can get dedicated proposals.
        if constraint_type == ConstraintType.JOINT:
            # Initial optimization of proposals.
            stor.optimize_proposal(
                p=p_,
                f=f_combined,
                n_steps=10,
                lr=1e-4,
                adjust_grads_=dh.proposal_adjust_grads_,
                callback=OptimizeProposalCallback()
            )

        optimal_decision = stor.optimize_decision(
            p=p_,
            f=f_combined if constraint_type == ConstraintType.JOINT else f_l_only,
            terminal_condition=terminal_condition,
            adjust_grads=lambda g: OrderedDict([(k, torch.clip(g[k], -1./100., 1./100.)) for k in g.keys()]),
            callback=OptimizeDecisionCallback(),
        )

    return  # just here to put a breakpoint on


if __name__ == '__main__':

    # plot_basic()

    plot_cost_vs_parameter('lockdown_strength')
    plot_cost_vs_parameter('lockdown_trigger')
    plot_cost_vs_parameter('lockdown_lift_trigger')
    # plot_cost_vs_parameters('lockdown_strength', 'lockdown_trigger')
    # plot_cost_vs_parameters('lockdown_strength', 'lockdown_lift_trigger')
    # plot_cost_vs_parameters('lockdown_trigger', 'lockdown_lift_trigger')

    # optimize_decision_2d_latent(constraint_type=ConstraintType.MEAN)
    # plot_basic(OrderedDict(lockdown_strength=tt(.663)))

    # _grad_debugging()

    # plot_cost_likelihood_convolution_for_stochastics(
    #     'beta', 'gamma', pyro_prior_over_sirlo_params_2d, f_=stor.build_expectigrand_gradient(DDP, f))
