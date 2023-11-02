from juliatorch import JuliaFunction

import juliacall, torch

jl = juliacall.Main.seval

jl('import Pkg')
jl('Pkg.add("DifferentialEquations")')
jl('Pkg.add("PythonCall")')
jl('using DifferentialEquations')

f = jl("""
function f(u0)
    ode_f(u, p, t) = -u
    tspan = (0.0, 1.0)
    prob = ODEProblem(ode_f, u0, tspan)
    sol = DifferentialEquations.solve(prob)
    return sol.u[end]
end""")

print(f(1))
# 0.36787959342751697
print(f(2))
# 0.7357591870280833
print(f(2)/f(1))
# 2.0000000004703966

x = torch.randn(3, 3, dtype=torch.double, requires_grad=True)

print(JuliaFunction.apply(f, x) / x)
# tensor([[0.3679, 0.3679, 0.3679],
#         [0.3679, 0.3679, 0.3679],
#         [0.3679, 0.3679, 0.3679]], dtype=torch.float64, grad_fn=<DivBackward0>)

from torch.autograd import gradcheck
py_f = lambda x: f(x)
print(gradcheck(JuliaFunction.apply, (py_f, x), eps=1e-6, atol=1e-4))
# True (wow, I honestly didn't expect that to work. Up to now
#       I'd only been using trivial Julia functions but it worked
#       on a full differential equation solver on the first try)
