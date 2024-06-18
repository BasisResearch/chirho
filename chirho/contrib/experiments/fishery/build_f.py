from os.path import (
    dirname as dn,
    join as jn
)
import juliacall, juliatorch
import torch

jl = juliacall.Main.seval


def _include_pth():
    pth = jn(dn(__file__), "fishery_model.jl")
    # print("Including", pth)
    return pth


def build_get_call_apply_julia_f(julia_f_ptr, build_julia_f):
    def get_call_apply_julia_f(x):
        if julia_f_ptr[0] is None:
            julia_f_ptr[0] = build_julia_f(x.detach().numpy())

        julia_f = julia_f_ptr[0]

        return juliatorch.JuliaFunction.apply(
            lambda x_: julia_f(x_),
            x
        )

    return get_call_apply_julia_f


def build_steady_state_f():
    jl(f"include(\"{_include_pth()}\")")
    julia_f = jl("pure_ss")

    def f(x: torch.Tensor):
        return juliatorch.JuliaFunction.apply(
            lambda x_: julia_f(x_),
            x
        )

    return f


def build_steady_state_fast_f():
    jl(f"include(\"{_include_pth()}\")")
    julia_build_f = jl("build_fast_pure_ss")

    julia_f_ptr = [None]

    return build_get_call_apply_julia_f(julia_f_ptr, julia_build_f)


def build_temporal_f():
    jl(f"include(\"{_include_pth()}\")")
    julia_f = jl("pure_t")

    def f(x: torch.Tensor):
        return juliatorch.JuliaFunction.apply(
            lambda x_: julia_f(x_),
            x
        )

    return f


def build_temporal_f_fast():
    jl(f"include(\"{_include_pth()}\")")
    julia_build_f = jl("build_fast_pure_t")

    julia_f_ptr = [None]
    
    return build_get_call_apply_julia_f(julia_f_ptr, julia_build_f)
