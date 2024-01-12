import logging

import pytest
import torch

from chirho.counterfactual.handlers import (
    MultiWorldCounterfactual,
    TwinWorldCounterfactual,
)
from chirho.dynamical.handlers import DynamicIntervention, LogTrajectory
from chirho.dynamical.handlers.solver import TorchDiffEq
from chirho.dynamical.ops import State, simulate
from chirho.indexed.ops import IndexSet, gather, indices_of, union

from .dynamical_fixtures import UnifiedFixtureDynamics

logger = logging.getLogger(__name__)

# Points at which to measure the state of the system.
start_time = torch.tensor(0.0)
end_time = torch.tensor(10.0)
logging_times = torch.linspace(start_time + 1, end_time - 2, 10)

# Initial state of the system.
init_state = dict(S=torch.tensor(50.0), I=torch.tensor(3.0), R=torch.tensor(0.0))

# State at which the dynamic intervention will trigger.
trigger_state1 = dict(R=torch.tensor(30.0))
trigger_state2 = dict(R=torch.tensor(50.0))

# State we'll switch to when the dynamic intervention triggers.
intervene_state1 = dict(S=torch.tensor(50.0))
intervene_state2 = dict(S=torch.tensor(30.0))


def get_state_reached_event_f(target_state: State[torch.tensor], event_dim: int = 0):
    def event_f(t: torch.tensor, state: State[torch.tensor]):
        actual, target = state["R"], target_state["R"]
        cf_indices = IndexSet(
            **{
                k: {1}
                for k in union(
                    indices_of(actual, event_dim=event_dim),
                    indices_of(target, event_dim=event_dim),
                ).keys()
            }
        )
        event_var = gather(actual - target, cf_indices, event_dim=event_dim)
        return event_var

    return event_f


@pytest.mark.parametrize("model", [UnifiedFixtureDynamics()])
@pytest.mark.parametrize("init_state", [init_state])
@pytest.mark.parametrize("start_time", [start_time])
@pytest.mark.parametrize("end_time", [end_time])
@pytest.mark.parametrize("logging_times", [logging_times])
@pytest.mark.parametrize(
    "trigger_states",
    [(trigger_state1, trigger_state2), (trigger_state2, trigger_state1)],
)
@pytest.mark.parametrize(
    "intervene_states",
    [(intervene_state1, intervene_state2), (intervene_state2, intervene_state1)],
)
def test_nested_dynamic_intervention_causes_change(
    model,
    init_state,
    start_time,
    end_time,
    logging_times,
    trigger_states,
    intervene_states,
):
    ts1, ts2 = trigger_states
    is1, is2 = intervene_states
    with LogTrajectory(
        times=logging_times,
    ) as dt:
        with TorchDiffEq():
            with DynamicIntervention(
                event_fn=get_state_reached_event_f(ts1),
                intervention=is1,
            ):
                with DynamicIntervention(
                    event_fn=get_state_reached_event_f(ts2),
                    intervention=is2,
                ):
                    simulate(model, init_state, start_time, end_time)

    preint_total = init_state["S"] + init_state["I"] + init_state["R"]

    # Each intervention just adds a certain amount of susceptible people after the recovered count exceeds some amount

    trajectory = dt.trajectory

    postint_mask1 = trajectory["R"] > ts1["R"]
    postint_mask2 = trajectory["R"] > ts2["R"]
    preint_mask = ~(postint_mask1 | postint_mask2)

    # TODO support dim != -1
    name_to_dim = {"__time": -1}
    preint_idx = IndexSet(
        __time=set(i for i in range(len(preint_mask)) if preint_mask[i])
    )

    # Make sure all points before the intervention maintain the same total population.
    preint_traj = gather(trajectory, preint_idx, name_to_dim=name_to_dim)
    assert torch.allclose(
        preint_total, preint_traj["S"] + preint_traj["I"] + preint_traj["R"]
    )

    # Make sure all points after the first intervention, but before the second, include the added population of that
    #  first intervention.
    postfirst_int_mask, postsec_int_mask = (
        (postint_mask1, postint_mask2)
        if ts1["R"] < ts2["R"]
        else (postint_mask2, postint_mask1)
    )
    firstis, secondis = (is1, is2) if ts1["R"] < ts2["R"] else (is2, is1)

    postfirst_int_presec_int_mask = postfirst_int_mask & ~postsec_int_mask

    assert torch.any(postfirst_int_presec_int_mask) or torch.any(
        postsec_int_mask
    ), "trivial test case"

    postfirst_int_presec_int_idx = IndexSet(
        __time=set(
            i
            for i in range(len(postfirst_int_presec_int_mask))
            if postfirst_int_presec_int_mask[i]
        )
    )

    postfirst_int_presec_int_traj = gather(
        trajectory, postfirst_int_presec_int_idx, name_to_dim=name_to_dim
    )
    assert torch.all(
        postfirst_int_presec_int_traj["S"]
        + postfirst_int_presec_int_traj["I"]
        + postfirst_int_presec_int_traj["R"]
        > (preint_total + firstis["S"]) * 0.95
    )

    postsec_int_idx = IndexSet(
        __time=set(i for i in range(len(postsec_int_mask)) if postsec_int_mask[i])
    )

    postsec_int_traj = gather(trajectory, postsec_int_idx, name_to_dim=name_to_dim)
    assert torch.all(
        postsec_int_traj["S"] + postsec_int_traj["I"] + postsec_int_traj["R"]
        > (preint_total + firstis["S"] + secondis["S"]) * 0.95
    )


@pytest.mark.parametrize("model", [UnifiedFixtureDynamics()])
@pytest.mark.parametrize("init_state", [init_state])
@pytest.mark.parametrize("start_time", [start_time])
@pytest.mark.parametrize("end_time", [end_time])
@pytest.mark.parametrize("logging_times", [logging_times])
@pytest.mark.parametrize("trigger_state", [trigger_state1])
@pytest.mark.parametrize("intervene_state", [intervene_state1])
def test_dynamic_intervention_causes_change(
    model,
    init_state,
    start_time,
    end_time,
    logging_times,
    trigger_state,
    intervene_state,
):
    with LogTrajectory(
        times=logging_times,
    ) as dt:
        with TorchDiffEq():
            with DynamicIntervention(
                event_fn=get_state_reached_event_f(trigger_state),
                intervention=intervene_state,
            ):
                simulate(model, init_state, start_time, end_time)

    preint_total = init_state["S"] + init_state["I"] + init_state["R"]

    trajectory = dt.trajectory

    # The intervention just "adds" (sets) 50 "people" to the susceptible population.
    #  It happens that the susceptible population is roughly 0 at the intervention point,
    #  so this serves to make sure the intervention actually causes that population influx.

    postint_mask = trajectory["R"] > trigger_state["R"]

    # TODO support dim != -1
    name_to_dim = {"__time": -1}

    preint_idx = IndexSet(
        __time=set(i for i in range(len(postint_mask)) if not postint_mask[i])
    )
    postint_idx = IndexSet(
        __time=set(i for i in range(len(postint_mask)) if postint_mask[i])
    )

    postint_traj = gather(trajectory, postint_idx, name_to_dim=name_to_dim)
    preint_traj = gather(trajectory, preint_idx, name_to_dim=name_to_dim)

    # Make sure all points before the intervention maintain the same total population.
    assert torch.allclose(
        preint_total, preint_traj["S"] + preint_traj["I"] + preint_traj["R"]
    )

    # Make sure all points after the intervention include the added population.
    # noinspection PyTypeChecker
    assert torch.all(
        postint_traj["S"] + postint_traj["I"] + postint_traj["R"]
        > (preint_total + intervene_state["S"]) * 0.95
    )


@pytest.mark.parametrize("model", [UnifiedFixtureDynamics()])
@pytest.mark.parametrize("init_state", [init_state])
@pytest.mark.parametrize("start_time", [start_time])
@pytest.mark.parametrize("end_time", [end_time])
@pytest.mark.parametrize("logging_times", [logging_times])
@pytest.mark.parametrize(
    "trigger_states",
    [(trigger_state1, trigger_state2), (trigger_state2, trigger_state1)],
)
@pytest.mark.parametrize(
    "intervene_states",
    [(intervene_state1, intervene_state2), (intervene_state2, intervene_state1)],
)
def test_split_twinworld_dynamic_intervention(
    model,
    init_state,
    start_time,
    end_time,
    logging_times,
    trigger_states,
    intervene_states,
):
    ts1, ts2 = trigger_states
    is1, is2 = intervene_states

    # Simulate with the intervention and ensure that the result differs from the observational execution.
    with LogTrajectory(
        times=logging_times,
    ) as dt:
        with TorchDiffEq():
            with DynamicIntervention(
                event_fn=get_state_reached_event_f(ts1),
                intervention=is1,
            ):
                with DynamicIntervention(
                    event_fn=get_state_reached_event_f(ts2),
                    intervention=is2,
                ):
                    with TwinWorldCounterfactual() as cf:
                        cf_state = simulate(
                            model,
                            init_state,
                            start_time,
                            end_time,
                        )

    with cf:
        cf_trajectory = dt.trajectory
        for k in cf_trajectory.keys():
            # TODO: Figure out why event_dim=1 is not needed with cf_state but is with cf_trajectory.
            assert cf.default_name in indices_of(cf_state[k])
            assert cf.default_name in indices_of(cf_trajectory[k], event_dim=1)


@pytest.mark.parametrize("model", [UnifiedFixtureDynamics()])
@pytest.mark.parametrize("init_state", [init_state])
@pytest.mark.parametrize("start_time", [start_time])
@pytest.mark.parametrize("end_time", [end_time])
@pytest.mark.parametrize(
    "trigger_states",
    [(trigger_state1, trigger_state2), (trigger_state2, trigger_state1)],
)
@pytest.mark.parametrize(
    "intervene_states",
    [(intervene_state1, intervene_state2), (intervene_state2, intervene_state1)],
)
def test_split_multiworld_dynamic_intervention(
    model, init_state, start_time, end_time, trigger_states, intervene_states
):
    ts1, ts2 = trigger_states
    is1, is2 = intervene_states

    # Simulate with the intervention and ensure that the result differs from the observational execution.
    with LogTrajectory(
        times=logging_times,
    ) as dt:
        with TorchDiffEq():
            with DynamicIntervention(
                event_fn=get_state_reached_event_f(ts1),
                intervention=is1,
            ):
                with DynamicIntervention(
                    event_fn=get_state_reached_event_f(ts2),
                    intervention=is2,
                ):
                    with MultiWorldCounterfactual() as cf:
                        cf_state = simulate(
                            model,
                            init_state,
                            start_time,
                            end_time,
                        )

    with cf:
        cf_trajectory = dt.trajectory
        for k in cf_trajectory.keys():
            # TODO: Figure out why event_dim=1 is not needed with cf_state but is with cf_trajectory.
            assert cf.default_name in indices_of(cf_state[k])
            assert cf.default_name in indices_of(cf_trajectory[k], event_dim=1)


@pytest.mark.parametrize("model", [UnifiedFixtureDynamics()])
@pytest.mark.parametrize("init_state", [init_state])
@pytest.mark.parametrize("start_time", [start_time])
@pytest.mark.parametrize("end_time", [end_time])
@pytest.mark.parametrize(
    "trigger_states",
    [(trigger_state1, trigger_state2), (trigger_state2, trigger_state1)],
)
@pytest.mark.parametrize(
    "intervene_states",
    [(intervene_state1, intervene_state2), (intervene_state2, intervene_state1)],
)
def test_split_twinworld_dynamic_matches_output(
    model, init_state, start_time, end_time, trigger_states, intervene_states
):
    ts1, ts2 = trigger_states
    is1, is2 = intervene_states

    with TorchDiffEq():
        with DynamicIntervention(
            event_fn=get_state_reached_event_f(ts1),
            intervention=is1,
        ):
            with DynamicIntervention(
                event_fn=get_state_reached_event_f(ts2),
                intervention=is2,
            ):
                with TwinWorldCounterfactual() as cf:
                    cf_result = simulate(model, init_state, start_time, end_time)

    with TorchDiffEq():
        with DynamicIntervention(
            event_fn=get_state_reached_event_f(ts1),
            intervention=is1,
        ):
            with DynamicIntervention(
                event_fn=get_state_reached_event_f(ts2),
                intervention=is2,
            ):
                cf_expected = simulate(model, init_state, start_time, end_time)

    with TorchDiffEq():
        factual_expected = simulate(model, init_state, start_time, end_time)

    with cf:
        factual_indices = IndexSet(
            **{k: {0} for k in indices_of(cf_result, event_dim=0).keys()}
        )

        cf_indices = IndexSet(
            **{k: {1} for k in indices_of(cf_result, event_dim=0).keys()}
        )

        cf_actual = gather(cf_result, cf_indices, event_dim=0)
        factual_actual = gather(cf_result, factual_indices, event_dim=0)

        assert not set(indices_of(cf_actual, event_dim=0))
        assert not set(indices_of(factual_actual, event_dim=0))

    assert cf_result.keys() == cf_actual.keys() == cf_expected.keys()
    assert cf_result.keys() == factual_actual.keys() == factual_expected.keys()

    for k in cf_result.keys():
        assert torch.allclose(
            cf_actual[k], cf_expected[k], atol=1e-3, rtol=0
        ), f"Trajectories differ in state result of variable {k}, but should be identical."

    for k in cf_result.keys():
        assert torch.allclose(
            factual_actual[k],
            factual_expected[k],
            atol=1e-3,
            rtol=0,
        ), f"Trajectories differ in state result of variable {k}, but should be identical."


def test_grad_of_dynamic_intervention_event_f_params():
    def model(X: State[torch.Tensor]):
        dX = dict()
        dX["x"] = torch.tensor(1.0)
        dX["z"] = X["dz"]
        dX["dz"] = torch.tensor(0.0)  # also a constant, this gets set by interventions.
        dX["param"] = torch.tensor(
            0.0
        )  # this is a constant event function parameter, so no change.
        return dX

    param = torch.nn.Parameter(torch.tensor(5.0))
    # Param has to be part of the state in order to take gradients with respect to it.
    s0 = dict(
        x=torch.tensor(0.0), z=torch.tensor(0.0), dz=torch.tensor(0.0), param=param
    )

    dynamic_intervention = DynamicIntervention(
        event_fn=lambda t, s: t - s["param"],
        intervention=dict(dz=torch.tensor(1.0)),
    )

    # noinspection DuplicatedCode
    with TorchDiffEq():
        with dynamic_intervention:
            result = simulate(model, s0, start_time, end_time)

    (dxdparam,) = torch.autograd.grad(
        outputs=(result["x"],), inputs=(param,), create_graph=True
    )
    assert torch.isclose(dxdparam, torch.tensor(0.0), atol=1e-5)

    # Z begins accruing dz=1 at t=param, so dzdparam should be -1.0.
    (dzdparam,) = torch.autograd.grad(
        outputs=(result["z"],), inputs=(param,), create_graph=True
    )
    assert torch.isclose(dzdparam, torch.tensor(-1.0), atol=1e-5)


def test_grad_of_event_f_params_torchdiffeq_only():
    # This tests functionality tests in test_grad_of_dynamic_intervention_event_f_params
    # See "NOTE: parameters for the event function must be in the state itself to obtain gradients."
    # In the torchdiffeq readme:
    #  https://github.com/rtqichen/torchdiffeq/blob/master/README.md#differentiable-event-handling

    import torchdiffeq

    param = torch.nn.Parameter(torch.tensor(5.0))

    dx = torch.tensor(1.0)
    dz = torch.tensor(0.0)
    dparam = torch.tensor(
        0.0
    )  # this is a constant event function parameter, so no change.
    ds = (dx, dz, dparam)

    t0 = torch.tensor(0.0)
    x0, z0, param0 = torch.tensor(0.0), torch.tensor(0.0), param
    s0 = (x0, z0, param0)  # x, z, param

    t_at_split, s_at_split = torchdiffeq.odeint_event(
        lambda t, s: ds,
        s0,
        t0,
        # Terminate when the final element of the state vector (the parameter) is equal to the time. i.e. terminate
        #  at t=param.
        event_fn=lambda t, s: t - s[-1],
    )

    assert torch.isclose(t_at_split, param)

    x_at_split, z_at_split, param_at_split = tuple(v[-1] for v in s_at_split)
    (dxdparam,) = torch.autograd.grad(
        outputs=(x_at_split,), inputs=(param,), create_graph=True
    )

    assert torch.isreal(dxdparam)
    assert torch.isclose(dxdparam, torch.tensor(1.0))

    dz = torch.tensor(1.0)

    t_at_end, s_at_end = torchdiffeq.odeint_event(
        lambda t, s: (dx, dz, torch.tensor(0.0)),
        (x_at_split, z_at_split, param_at_split),
        t_at_split,
        event_fn=lambda t, s: t - torch.tensor(10.0),  # Terminate at a constant t=10.
    )

    x_at_end, z_at_end, param_at_end = tuple(v[-1] for v in s_at_end)
    (dxdparam,) = torch.autograd.grad(
        outputs=(x_at_end,), inputs=(param,), create_graph=True
    )

    assert torch.isclose(dxdparam, torch.tensor(0.0), atol=1e-5)

    (dzdparam,) = torch.autograd.grad(
        outputs=(z_at_end,), inputs=(param,), create_graph=True
    )

    assert torch.isclose(dzdparam, torch.tensor(-1.0))

    # Run a second time without the event function, but with the t_at_end terminating the tspan.
    s_at_end2 = torchdiffeq.odeint(
        func=lambda t, s: (dx, dz, torch.tensor(0.0)),
        y0=(x_at_split, z_at_split, param_at_split),
        t=torch.cat((t_at_split[None], t_at_end[None])),
        # t=torch.tensor((t_at_split[None], t_at_end[None])),  <-- This is what breaks the gradient propagation.
    )

    x_at_end2, z_at_end2, param_at_end2 = tuple(v[-1] for v in s_at_end2)

    (dxdparam2,) = torch.autograd.grad(
        outputs=(x_at_end2,), inputs=(param,), create_graph=True
    )

    assert torch.isclose(dxdparam, dxdparam2)

    (dzdparam2,) = torch.autograd.grad(
        outputs=(z_at_end2,), inputs=(param,), create_graph=True
    )

    assert torch.isclose(dzdparam, dzdparam2)
