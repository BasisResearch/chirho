from juliatorch import JuliaFunction
import juliacall, torch
from chirho.contrib.experiments.fishery.build_f import build_f

jl = juliacall.Main.seval

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


def _build_f_and_x():
    f = build_f()

    params = _params()
    B1, B2, B3 = 1000.0, 200.0, 30.0
    x = torch.tensor([B1, B2, B3, *params], dtype=torch.float64, requires_grad=True)

    return f, x


def test_forward():
    f, x = _build_f_and_x()

    # Check forward execution.
    ss = f(x)

    assert ss.shape == (3,)
    assert ss.dtype == torch.float64
    assert torch.all(ss >= 0.0)


def test_grads():
    f, x = _build_f_and_x()

    # Check gradients.
    torch.autograd.gradcheck(f, (x,), eps=1e-6, atol=1e-4)
