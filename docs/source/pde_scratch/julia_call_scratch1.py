from juliatorch import JuliaFunction
import juliacall, torch
from torch.autograd import gradcheck

f = juliacall.Main.seval("f(x) = exp.(-x .^ 2)")
py_f = lambda x: f(x)
x = torch.randn(3, 3, dtype=torch.double, requires_grad=True)
JuliaFunction.apply(f, x)
print("gradcheck", gradcheck(JuliaFunction.apply, (py_f, x), eps=1e-6, atol=1e-4))
