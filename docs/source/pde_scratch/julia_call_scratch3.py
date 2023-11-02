from juliatorch import JuliaFunction
import juliacall, torch
import os
import matplotlib.pyplot as plt

jl = juliacall.Main.seval

# Read the contents of the file at into a string.
PREP_FOR_JLF_PTH = "/Users/azane/GitRepo/causal_pyro/docs/source/pde_scratch/src/halfar_ice/ice_hackathon.jl"

# Include the above file in the julia session.
jl(f'include("{PREP_FOR_JLF_PTH}")')

f = jl("""
function f()
    tₑ = 300 * 1000
    prob = ODEProblem(fₘ, u₀, (0, tₑ), constants_and_parameters)
    @info("Solving")
    soln = solve(prob, Tsit5())
    @show soln.retcode
    @info("Done")
    return soln.u
end
""")

soln = f()

# TODO parameterize the above in terms of constants and parameters:
#  1. stress_rho, stress_A, tₑ
#  2. gradcheck wrt rho and A
#  3. the mesh s' and initial condition u_0
