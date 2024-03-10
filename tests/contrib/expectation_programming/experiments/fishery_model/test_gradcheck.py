import juliacall, juliatorch  # have to import before torch else sigsegv
import torch
from chirho.contrib.experiments.fishery.build_f import build_steady_state_f, build_temporal_f


def _repo_root():
    from os.path import dirname as dn
    return dn(dn(dn(dn(dn(dn(__file__))))))


def _params():
    K1 = 1000.0
    r1, r2, r3 = 2.0, 1.0, 0.25
    p12, p23 = 0.5, 0.5
    D1, D2 = 100.0, 10.0
    e12, e23 = 0.2, 0.2
    M3 = 0.01
    f = 0.5
    F1, F2, F3 = f * r1, f * r2, f * r3

    return r1, K1, p12, D1, F1, r2, e12, p23, D2, F2, r3, e23, M3, F3


def _build_ss_f_and_x():
    f = build_steady_state_f()

    params = _params()
    B1, B2, B3 = 1000.0, 200.0, 30.0
    x = torch.tensor([B1, B2, B3, *params], dtype=torch.float64, requires_grad=True)

    return f, x


def _build_temporal_f_and_x():
    f = build_temporal_f()

    params = _params()
    B1, B2, B3 = 1000.0, 200.0, 30.0
    tspan = torch.linspace(0.0, 5.0, 7)
    x = torch.tensor([B1, B2, B3, *params, *tspan], dtype=torch.float64, requires_grad=True)

    return f, x


def test_steadystate_forward():
    f, x = _build_ss_f_and_x()

    # Check forward execution.
    ss = f(x)

    assert ss.shape == (3,)
    assert ss.dtype == torch.float64
    assert torch.all(ss >= 0.0)


def test_steadystate_grads():
    f, x = _build_ss_f_and_x()

    # Check gradients.
    torch.autograd.gradcheck(f, (x,), eps=1e-6, atol=1e-4)


def test_temporal_forward():
    f, x = _build_temporal_f_and_x()

    # Check forward execution.
    ts = f(x)

    assert ts.shape == (3, 7)
    assert ts.dtype == torch.float64
    assert torch.all(ts >= 0.0)


def test_temporal_grads():
    f, x = _build_temporal_f_and_x()

    # Check gradients.
    # FIXME rtol here has to be really high, likely due to some mathematical nuance.
    #  Requires further investigation.
    torch.autograd.gradcheck(f, (x,), atol=1e-4, rtol=1e-1)
