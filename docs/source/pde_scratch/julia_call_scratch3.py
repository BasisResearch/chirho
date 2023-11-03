from juliatorch import JuliaFunction
import juliacall, torch
from torch import tensor as tnsr
import sys
sys.path.insert(0, "/Users/azane/GitRepo/causal_pyro/docs/source/pde_scratch/src/halfar_ice")
from analytical import stable_gamma, halfar_ice_analytical
import matplotlib.pyplot as plt
import pyro
import pyro.distributions as dist
import numpy as np

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
  tₑ = 1e11

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

true_ice_density = tnsr(910.).double()
true_log_flow_rate = tnsr(1e-16).double().log()


def prior():
    log_flow_rate = pyro.sample("log_flow_rate", dist.Uniform(1e-20, 1e-15)).double().log()
    ice_density = pyro.sample("ice_density", dist.Uniform(840, 930)).double()

    return log_flow_rate, ice_density


xx = torch.linspace(-2., 2., 512)


u_init = halfar_ice_analytical(
    h0=tnsr(1.),
    r0=tnsr(1.),
    gamma=stable_gamma(lA=true_log_flow_rate, rho=true_ice_density),
    t=tnsr(0),
    r=xx
).double()


def true_depth_model(log_flow_rate, ice_density):

    x = torch.cat((log_flow_rate.exp()[None], ice_density[None], u_init), dim=0)
    return JuliaFunction.apply(f, x).T


def model():
    log_flow_rate, ice_density = prior()
    depth = true_depth_model(log_flow_rate, ice_density)

    # pyro.sample("depth", dist.Normal(depth, 1e-3), obs=tnsr(0.))

    return depth


def get_pointwise_between(preds):
    ret = preds.quantile(0.5, dim=0), preds.quantile(0.95, dim=0), preds.quantile(0.05, dim=0)

    if isinstance(preds, torch.Tensor):
        return tuple(x.detach().numpy() for x in ret)
    return ret


def plot_preds(prior_preds):

    # Get the pointwise mean, upper and lower quartiles for each predictive distribution.
    prior_median, prior_upper, prior_lower = get_pointwise_between(prior_preds)

    # Plot the prior and posterior predictive distributions.
    # fig, ax = plt.subplots(dpi=500, figsize=(8, 2.8))

    ax.plot(xx, prior_median, color='blue', linewidth=0.7, label="Prior Predictive", alpha=0.5)
    ax.fill_between(xx, prior_lower, prior_upper, color='blue', alpha=0.1)

    ax.set_ylabel("Height")
    ax.set_xlabel("Radius")
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.legend(facecolor='lightgray', fontsize=7)


# prior_samples = torch.tensor([prior() for _ in range(5000)])

fig, ax = plt.subplots(dpi=500, figsize=(8, 2.8))
# Make equal aspect ratio.
ax.set_aspect('equal')
plt.plot(xx, u_init, color='purple', linestyle='--')

prior_preds = torch.stack([model()[-1] for _ in range(200)])

plot_preds(prior_preds)

plt.show()

print()

# TODO parameterize the above in terms of constants and parameters:
#  [/] stress_rho, stress_A, tₑ
#  [/] put the parameters into a single tensor on python side, then unpack them accordingly julia side.
#  [ ] gradcheck wrt rho and A
#  [ ] the mesh s' and initial condition u_0
