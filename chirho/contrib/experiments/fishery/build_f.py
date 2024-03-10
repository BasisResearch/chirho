from os.path import (
    dirname as dn,
    join as jn
)
import juliacall, juliatorch
import torch

jl = juliacall.Main.seval


def _include_pth():
    return jn(dn(__file__), "fishery_model.jl")


def build_steady_state_f():
    jl(f"include(\"{_include_pth()}\")")
    julia_f = jl("pure_ss")

    def f(x: torch.Tensor):
        return juliatorch.JuliaFunction.apply(
            lambda x_: julia_f(x_),
            x
        )

    return f


def build_temporal_f():
    jl(f"include(\"{_include_pth()}\")")
    julia_f = jl("pure_t")

    def f(x: torch.Tensor):
        return juliatorch.JuliaFunction.apply(
            lambda x_: julia_f(x_),
            x
        )

    return f
