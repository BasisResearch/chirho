import pyro
import torch
import pyro.distributions as dist

from chirho.dynamical.handlers import (
    DynamicIntervention,
    SimulatorEventLoop,
    simulate,
    ODEDynamics,
)

from chirho.dynamical.ops import State
from torch import tensor as tt
from enum import Enum


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
    # TODO can something like this be accomplished with functools partial?
    def __init__(self):
        self.result = None

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
