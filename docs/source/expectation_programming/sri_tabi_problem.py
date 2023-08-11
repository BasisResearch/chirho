import pyro
import torch
import pyro.distributions as dist
import numpy as np

from chirho.dynamical.handlers import (
    DynamicIntervention,
    SimulatorEventLoop,
    simulate,
    ODEDynamics,
)

from chirho.dynamical.ops import State, Trajectory
from enum import Enum
from collections import OrderedDict
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity

from chirho.contrib.compexp.typedecs import ModelType, KWType
import chirho.contrib.compexp as ep

from typing import List, Optional

TT = torch.Tensor
tt = torch.tensor


# Largely for debugging...
class LockdownType(Enum):
    CONT_PLATEAU = 1
    NONCONT_STATE = 2
    NONCONT_TIME = 3


LOCKDOWN_TYPE = LockdownType.NONCONT_STATE


class SimpleSIRDynamics(ODEDynamics):
    def __init__(self, beta, gamma):
        super().__init__()
        self.beta = beta
        self.gamma = gamma

    def diff(self, dX: State[torch.Tensor], X: State[torch.Tensor]):
        dX.S = -self.beta * X.S * X.I
        dX.I = self.beta * X.S * X.I - self.gamma * X.I
        dX.R = self.gamma * X.I

    def observation(self, X: State[torch.Tensor]):

        I_obs = pyro.sample(f"I_obs", dist.Poisson(X.I))  # noisy number of infected actually observed
        R_obs = pyro.sample(f"R_obs", dist.Poisson(X.R))  # noisy number of recovered actually observed

        return {
            f"I_obs": I_obs,
            f"R_obs": R_obs,
        }


class SimpleSIRDynamicsLockdown(SimpleSIRDynamics):
    def __init__(self, beta0, gamma):
        super().__init__(torch.zeros_like(gamma), gamma)
        self.beta0 = beta0

    def diff(self, dX: State[torch.Tensor], X: State[torch.Tensor]):

        # Lockdown strength is a piecewise constant function affected by dynamic interventions, so dX is 0.
        dX.l = torch.tensor(0.0)

        if LOCKDOWN_TYPE == LOCKDOWN_TYPE.CONT_PLATEAU:
            # Pretend the trigger is also in terms of the number recovered.
            recovered_trigger = X.lockdown_trigger
            # Pretend the lift trigger is in terms of how many additional recovered need to trigger.
            recovered_lift_trigger = recovered_trigger + X.lockdown_lift_trigger
            # Definep dX.l as a continuous plateau.
            plateau_u = (recovered_lift_trigger + recovered_trigger) / tt(2.)
            plateau_s = (recovered_lift_trigger - recovered_trigger) / tt(2.)
            dXL_override = X.static_lockdown_strength * torch.exp(-((X.R - plateau_u) / plateau_s)**tt(10))

            dX.L = dXL_override
            self.beta = (1 - dXL_override) * self.beta0
        else:
            # Time-varing beta parametrized by lockdown strength l_t
            self.beta = (1 - X.l) * self.beta0

            # Accrual of lockdown time.
            dX.L = X.l

        # Constant event parameters have to be in the state in order for torchdiffeq to give derivs.
        dX.lockdown_trigger = tt(0.0)
        dX.lockdown_lift_trigger = tt(0.0)
        dX.static_lockdown_strength = tt(0.0)

        # Call the base SIR class diff method
        super().diff(dX, X)


class SimpleSIRDynamicsLockdownCapacityOverrun(SimpleSIRDynamicsLockdown):
    def __init__(self, beta0, gamma, capacity, hospitalization_rate):
        super().__init__(beta0, gamma)

        self.capacity = capacity
        self.hospitalization_rate = hospitalization_rate

    def diff(self, dX: State[torch.Tensor], X: State[torch.Tensor]):
        # If the number of infected individuals needing hospitalization exceeds the capacity, accrue that difference
        #  in the overrun factor.
        dX.O = torch.relu(X.I * self.hospitalization_rate - self.capacity)

        super().diff(dX, X)


def initiate_lockdown(t: torch.tensor, state: State[torch.tensor]):
    target_recovered = state.lockdown_lift_trigger
    target_infected = state.lockdown_trigger

    if LOCKDOWN_TYPE == LockdownType.NONCONT_STATE:
        # To enact the policy, require that the lift trigger hasn't effectively already fired.
        # Do this by saying that the infected count is treated as zero if the lift trigger had fired.
        # Let's say the lockdown is approaching its firing state (infected count is increasing), then this value
        #  will start going to zero, but if the lift trigger had fired, then it will jump up to the non-zero
        #  target state again and the lockdown will never fire.
        return target_infected - state.I * torch.greater_equal(target_recovered, state.R).type(state.R.dtype)
    elif LOCKDOWN_TYPE == LockdownType.NONCONT_TIME:
        return target_infected - t
    elif LOCKDOWN_TYPE == LockdownType.CONT_PLATEAU:
        # Disabled entirely for continuous plateau.
        return tt(1.)


def lift_lockdown(t: torch.tensor, state: State[torch.tensor]):
    target_recovered = state.lockdown_lift_trigger

    if LOCKDOWN_TYPE == LockdownType.NONCONT_STATE:
        # To lift the policy, require that the recovered count exceeds a certain level, and that a lockdown of
        #  non-zero strength is in place.
        return target_recovered - state.R + torch.isclose(state.l, torch.tensor(0.0)).type(state.l.dtype)
    elif LOCKDOWN_TYPE == LockdownType.NONCONT_TIME:
        return target_recovered - t
    elif LOCKDOWN_TYPE == LockdownType.CONT_PLATEAU:
        # Disabled entirely for continuous plateau.
        return tt(1.)


class ReuseableSimulation:
    """
    This reusable simulation can be used in both the cost and failure magnitude functions so the same
     simulation doesn't have to be run twice.
    """
    # TODO can something like this be accomplished with functools partial?
    def __init__(self):
        self.result = None

    @staticmethod
    def constrain_params(
            lockdown_trigger, lockdown_lift_trigger, lockdown_strength,
            beta, gamma, capacity, hospitalization_rate, **kwargs):

        dparams = dict(
            lockdown_trigger=lockdown_trigger,
            lockdown_lift_trigger=lockdown_lift_trigger,
            lockdown_strength=lockdown_strength
        )
        stochastics = dict(
            beta=beta,
            gamma=gamma,
            capacity=capacity,
            hospitalization_rate=hospitalization_rate
        )

        for k, v in stochastics.items():
            # All stochastics need to be positive, so just do it here and not worry about it in the guide.
            stochastics[k] = torch.abs(v)

        for k, v in dparams.items():
            # All dparams need to be positive and non-zero, so put them through a relu with eps added.
            dparams[k] = torch.relu(v) + 1e-3

        return dparams, stochastics

    @staticmethod
    def _inner_call(lockdown_trigger, lockdown_lift_trigger, lockdown_strength,
                    init_state, beta, gamma, capacity, hospitalization_rate, times, **kwargs):

        # Make a new state object so we can add the lockdown trigger constants to the state without modifying the
        #  original. This is required because torchdiffeq requires that even constant event parameters must be in the
        #  state in order to take gradients with respect to them.
        new_init_state = State()
        for k in init_state.keys:
            setattr(new_init_state, k, getattr(init_state, k))
        setattr(new_init_state, "lockdown_trigger", lockdown_trigger)
        setattr(new_init_state, "lockdown_lift_trigger", lockdown_lift_trigger)
        setattr(new_init_state, "static_lockdown_strength", lockdown_strength)

        if torch.isclose(lockdown_trigger, torch.tensor(0.0)) or torch.less(lockdown_trigger, torch.tensor(0.0)):
            raise ValueError("Lockdown trigger must be greater than zero.")

        if torch.isclose(lockdown_lift_trigger, torch.tensor(0.0)) or torch.less(lockdown_lift_trigger, torch.tensor(0.0)):
            raise ValueError("Lockdown lift trigger must be greater than zero.")

        sir = SimpleSIRDynamicsLockdownCapacityOverrun(beta, gamma, capacity, hospitalization_rate)
        with SimulatorEventLoop():
            with DynamicIntervention(event_f=initiate_lockdown,
                                     intervention=State(l=lockdown_strength),
                                     var_order=new_init_state.var_order, max_applications=1, ):
                with DynamicIntervention(event_f=lift_lockdown,
                                         intervention=State(l=torch.tensor(0.0)), var_order=new_init_state.var_order,
                                         max_applications=1):
                    tspan = times
                    if not torch.isclose(tspan[0], torch.tensor(0.0)):
                        tspan = torch.cat([torch.tensor([0.]), tspan])

                    soln = simulate(sir, new_init_state, tspan)

        return soln

    def __call__(self, *args, **kwargs):
        if self.result is None:
            self.result = self._inner_call(*args, **kwargs)
        else:
            # TODO assert that the arguments are the same?
            pass

        return self.result


def cost(dparams, stochastics, end_time, reuseable_sim) -> torch.Tensor:
    sim_results = reuseable_sim(dparams, stochastics, end_time)

    # The cost of the lockdown policy is the total lockdown time times the severity. See lockdown diff.
    return sim_results[-1].L


def failure_magnitude(dparams, stochastics, end_time, reuseable_sim) -> torch.Tensor:
    sim_results = reuseable_sim(dparams, stochastics, end_time)

    # The failure magnitude of lockdown policy is the total time spent by infected individuals who needed
    #  hospitalization but could not get it.
    return sim_results[-1].O


def get_traj(
        dparams: KWType,
        stochastics: KWType,
        timespan: TT,
        init_state: State,
        rs: Optional[ReuseableSimulation] = None):

    if rs is None:
        rs = ReuseableSimulation()

    d, s = rs.constrain_params(**dparams, **stochastics)
    return rs(**d, **s, init_state=init_state, times=timespan)


# noinspection PyPep8Naming
def plot_basic(dparams: KWType, stochastics: List[KWType], timespan: TT, init_state: State, ci=0.90):

    d = dparams
    t = timespan
    x0 = init_state

    Ss = []
    Is = []
    Rs = []
    Ls = []
    Os = []
    ls = []
    capacities = []
    hospitalization_rates = []

    for s in stochastics:
        traj = get_traj(d, s, t, x0)
        Ss.append(traj.S.detach().numpy())
        Is.append(traj.I.detach().numpy())
        Rs.append(traj.R.detach().numpy())
        Ls.append(traj.L.detach().numpy())
        Os.append(traj.O.detach().numpy())
        ls.append(traj.l.detach().numpy())
        capacities.append(s['capacity'].item())
        hospitalization_rates.append(s['hospitalization_rate'].item())

    Ss = np.array(Ss)
    Is = np.array(Is)
    Rs = np.array(Rs)
    Ls = np.array(Ls)
    Os = np.array(Os)
    ls = np.array(ls)
    t = t.detach().numpy()
    capacities = np.array(capacities)
    hospitalization_rates = np.array(hospitalization_rates)

    fig, (ax2, ax3, ax1) = plt.subplots(3, 1, figsize=(7, 10))
    tax3 = ax3.twinx()

    def plot_elementwise_mean_and_error(ax, y, mean_kwargs, error_kwargs):
        # Plot mean:
        ax.plot(t, y.mean(axis=0), **mean_kwargs)
        # Plot the confidence range specified by the ci argument passed by the user.
        top = np.quantile(y, (1. + ci) / 2., axis=0)
        bot = np.quantile(y, (1. - ci) / 2., axis=0)
        ax.fill_between(t, top, bot, **error_kwargs)

    def plot_axhline_error(ax, y, mean_kwargs, error_kwargs):
        # Extend y, which is a 1d vector, to repeat for the length of t along a new axis.
        y = np.repeat(y[:, np.newaxis], t.shape[0], axis=1)
        plot_elementwise_mean_and_error(ax, y, mean_kwargs, error_kwargs)

    plot_axhline_error(ax2, capacities * (1. / hospitalization_rates),
                       mean_kwargs=dict(color='k', linestyle='--', label='Healthcare Capacity'),
                       error_kwargs=dict(color='k', alpha=0.2))
    plot_axhline_error(ax3, capacities * (1. / hospitalization_rates),
                       mean_kwargs=dict(color='k', linestyle='--', label='Healthcare Capacity'),
                       error_kwargs=dict(color='k', alpha=0.2))

    plot_elementwise_mean_and_error(
        ax2, Ss, mean_kwargs=dict(color='blue', label='Susceptible'),
        error_kwargs=dict(color='blue', alpha=0.2))
    plot_elementwise_mean_and_error(
        ax2, Is, mean_kwargs=dict(color='red', label='Infected'),
        error_kwargs=dict(color='red', alpha=0.2))
    plot_elementwise_mean_and_error(
        ax3, Is, mean_kwargs=dict(color='red', label='Infected'),
        error_kwargs=dict(color='red', alpha=0.2))
    plot_elementwise_mean_and_error(
        ax2, Rs, mean_kwargs=dict(color='green', label='Recovered'),
        error_kwargs=dict(color='green', alpha=0.2))
    plot_elementwise_mean_and_error(
        ax1, Ls, mean_kwargs=dict(color='orange', label='Lockdown Cost'),
        error_kwargs=dict(color='orange', alpha=0.2))
    plot_elementwise_mean_and_error(
        ax1, ls, mean_kwargs=dict(color='orange', label='Lockdown', linestyle='--'),
        error_kwargs=dict(color='orange', alpha=0.2))
    plot_elementwise_mean_and_error(
        tax3, Os, mean_kwargs=dict(color='k', label='Aggregate Overrun'),
        error_kwargs=dict(color='k', alpha=0.2))

    ax1.legend()
    ax2.legend()
    # Make a combined legend for ax3 and tax3.
    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = tax3.get_legend_handles_labels()
    ax3.legend(lines + lines2, labels + labels2, loc=0)

    ax1.set_xlabel('Years')
    ax1.set_ylabel('Lockdown Strength')
    ax2.set_ylabel('Proportion of Population')
    ax2.set_ylim(0, 1)
    ax3.set_ylabel('Proportion of Population')
    ax3.set_ylim(0, 1)
    tax3.set_ylabel('Aggregate Overrun')

    plt.tight_layout()

    plt.show()


def _NNM_vectorized(f_, N, M, X, Y):
    out = np.empty((N, N, M))
    it = np.nditer(out, flags=['multi_index'])
    while not it.finished:
        out[*it.multi_index[:-1], :] = f_(X[it.multi_index[:-1]], Y[it.multi_index[:-1]])
        it.iternext()

    return out


# noinspection DuplicatedCode
def plot_cost_likelihood_convolution_for_stochastics(
        stochastic_name1: str, stochastic_name2: str,
        p: ModelType, ce: ep.ComposedExpectation, n=500,
        resolution=5, bandwidth=0.1):

    ve = ValueError("Can only plot convolutions for stacked compositions.")

    if not repr(ce).startswith('stack'):
        raise ve

    for c in ce.children:
        if not isinstance(c, ep.ExpectationAtom):
            raise ve

    atoms: List[ep.ExpectationAtom] = [c for c in ce.children]

    samples = []
    for _ in range(n):  # Generate samples from the model.
        sample = p()
        samples.append((sample[stochastic_name1].item(), sample[stochastic_name2].item()))

    samples = np.array(samples)

    kde: KernelDensity = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(samples)

    def density(s1, s2) -> np.ndarray:
        return np.array([np.exp(kde.score_samples([[s1, s2]]))])

    def cost(s1, s2) -> np.ndarray:
        stochastics: KWType = OrderedDict({stochastic_name1: tt(s1), stochastic_name2: tt(s2)})
        vals = [p.f(stochastics) for p in atoms]
        return np.array(tuple(v.item() for v in vals))

    def cust_colorbar(ax, arr):
        ticks = np.linspace(0., 1.0, 10)
        cbar = fig.colorbar(ax=ax, mappable=ax.collections[0], ticks=ticks)
        arrmax = arr.max()
        arrmin = arr.min()
        tick_labels = ticks * (arrmax - arrmin) + arrmin
        cbar.ax.set_yticklabels(['{:.1f}'.format(tick) for tick in tick_labels])

    s1ls = np.linspace(0.00, 1.0, resolution)
    s2ls = np.linspace(0.00, 1.0, resolution)

    s1ls = s1ls * (samples[:, 0].max() - samples[:, 0].min()) + samples[:, 0].min()
    s2ls = s2ls * (samples[:, 1].max() - samples[:, 1].min()) + samples[:, 1].min()

    # noinspection PyPep8Naming
    X, Y = np.meshgrid(s1ls, s2ls)

    # Make a subplot for the density and one for each component of the cost.
    cost_component_names = [c.name for c in atoms]
    num_cost_components = len(cost_component_names)

    fig, axs = plt.subplots(num_cost_components + 1, 3, figsize=(6*num_cost_components, 18))

    # Plot the density in the top row.
    density_array = _NNM_vectorized(f_=density, N=resolution, M=1, X=X, Y=Y)
    for col_ in range(3):
        axs[0][col_].contourf(X, Y, density_array[..., 0], cmap='viridis')
        cust_colorbar(axs[0][col_], density_array[..., 0])

    # Compute the cost array. We have to do this manually because there
    cost_array = _NNM_vectorized(f_=cost, N=resolution, M=num_cost_components, X=X, Y=Y)

    assert cost_array.shape == (resolution, resolution, num_cost_components)

    # Plot the cost components in the left column.
    for i_, cost_component_name_ in enumerate(cost_component_names):
        axs[i_+1][0].contourf(X, Y, cost_array[:, :, i_], cmap='viridis')
        axs[i_+1][0].set_title(cost_component_name_.capitalize())
        cust_colorbar(axs[i_+1][0], cost_array[:, :, i_])

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
