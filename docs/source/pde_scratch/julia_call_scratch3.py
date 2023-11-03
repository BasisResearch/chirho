from juliatorch import JuliaFunction
import juliacall, torch
from torch import tensor as tnsr
import sys
sys.path.insert(0, "/Users/azane/GitRepo/causal_pyro/docs/source/pde_scratch/src/halfar_ice")
from analytical import stable_gamma, halfar_ice_analytical
import matplotlib.pyplot as plt

jl = juliacall.Main.seval

# Read the contents of the file at into a string.
PREP_FOR_JLF_PTH = "/Users/azane/GitRepo/causal_pyro/docs/source/pde_scratch/src/halfar_ice/ice_hackathon.jl"

# Include the above file in the julia session.
jl(f'include("{PREP_FOR_JLF_PTH}")')

f = jl("""
function f(concat_tensor)
  # split out torch tensor of each variable
  flow_rate = concat_tensor[1]
  ice_density = concat_tensor[2]
  u_init_arr = concat_tensor[3:end]

  n = 3
  ρ = ice_density
  g = 9.8101
  A = fill(flow_rate, ne(s)) # `fill(a,b)` creates an b-shaped array of entirely a values
  tₑ = 5e13

  u₀ = ComponentArray(dynamics_h = u_init_arr)
  
  constants_and_parameters = (
    n = n,
    stress_ρ = ρ,
    stress_g = g,
    stress_A = A)
  prob = ODEProblem(fₘ, u₀, (0, tₑ), constants_and_parameters)
  @info("Solving")
  soln = solve(prob, Tsit5())
  @info("Done")

  # Convert to matrix for juliatorch post-processing.
  return reduce(hcat, [Vector(yn.dynamics_h) for yn in soln.u])
end
""")

ice_density = tnsr(910.).double()
flow_rate = tnsr(1e-16).double()

u_init = halfar_ice_analytical(
    h0=tnsr(1.),
    r0=tnsr(1.),
    gamma=stable_gamma(rho=ice_density, lA=flow_rate),
    t=tnsr(0),
    r=torch.linspace(-1.5, 1.5, 100)
).double()


def torch_f(x):
    return JuliaFunction.apply(f, x)


x_ = torch.cat((flow_rate[None], ice_density[None], u_init), dim=0)
x_ = x_.detach().clone().requires_grad_(True)
y = torch_f(x_)

plt.figure()
[plt.plot(sol_at_t) for sol_at_t in y.T.detach()]
plt.show()

print()

# TODO parameterize the above in terms of constants and parameters:
#  [/] stress_rho, stress_A, tₑ
#  [/] put the parameters into a single tensor on python side, then unpack them accordingly julia side.
#  [ ] gradcheck wrt rho and A
#  [ ] the mesh s' and initial condition u_0
