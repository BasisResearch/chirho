# HACK when running osx-64 conda env on M2 mac. Side effects could be an issue:
# https://stackoverflow.com/a/53014308
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import fenics_overloaded as fe
import torch_fenics

from torch import tensor as tt


def initial_curve(x):
    condition = (x >= -1.0) & (x <= 1.0)
    values = 1.0 - x**2 + 0.3 * torch.sin(2. * torch.pi * x) ** 2.
    return torch.where(condition, values, torch.zeros_like(x))


def fillna(tensor):
    return torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)


class HalfarIce(torch_fenics.FEniCSModule):

    def __init__(self, n_elements=64):

        super().__init__()

        self.n_elements = n_elements

        # Mesh on which to build finite element function space.
        self.mesh = fe.IntervalMesh(n_elements, -2., 2.)

        self.V = fe.FunctionSpace(self.mesh, 'Lagrange', 1)

        # See https://fenicsproject.org/pub/tutorial/html/._ftut1020.html for details.
        self.W = fe.VectorFunctionSpace(self.V.mesh(), 'P', self.V.ufl_element().degree())

        # Homogeneous Dirichlet boundary condition.
        self.bc = fe.DirichletBC(
            self.V,
            # Boundary condition of zero.
            fe.Constant(0.0),
            'on_boundary'
        )

        self.u_trial = fe.TrialFunction(self.V)
        self.v_test = fe.TestFunction(self.V)

    def solve(self, tstep, u_last_t, u_last_k):

        utr = self.u_trial
        vte = self.v_test
        ult = u_last_t
        ulk = u_last_k

        cont_grad_ulk = fe.project(fe.grad(ulk), self.W)

        weak_form_residuum = (
            utr * vte * fe.dx
            - ult * vte * fe.dx
            + tstep * fe.dot(
                # So the middle bit is usually grad(utr) * fe.sqrt(grad(utr) * grad(utr) + 1e-6) ** (n-1),
                #  but that can be simplified for the typical n=3. Same with utr ** (n+2)
                # TODO I'm not sure if, in general multi-dim case, that's an elementwise ^3 or if dotted.
                #  For single dim doesn't matter though.
                # fe.elem_pow(fe.grad(utr), fe.as_vector([3.])) * utr ** 5,
                fe.grad(utr) * (ulk ** 5) * fe.dot(cont_grad_ulk, cont_grad_ulk),
                fe.grad(vte)
            ) * fe.dx
        )

        lhs = fe.lhs(weak_form_residuum)
        rhs = fe.rhs(weak_form_residuum)

        u_sol = fe.Function(self.V)

        # Solve from initial condition out to tstep in time.
        fe.solve(
            lhs == rhs,
            u_sol,
            self.bc
        )

        return u_sol

    def input_templates(self):
        # tstep, u_last
        return fe.Constant(0), fe.Function(self.V), fe.Function(self.V)


def main():

    halfar_ice = HalfarIce(512)

    xx = torch.linspace(-2., 2., halfar_ice.n_elements + 1)
    u_init = initial_curve(xx)[None, :].double()\
        .clone().detach().requires_grad_(True)  # Make this a leaf tensor.

    u_last_t = u_init

    plt.show()

    sols = [u_last_t[0]]

    tstep = tt(.0002).double()[None, None]
    for i in range(500):
        if len(sols) > 1:
            outer_div = torch.linalg.norm(sols[-1] - sols[-2])
            if outer_div < 1e-3:
                break
            outer_div = outer_div.item()
        else:
            outer_div = torch.inf

        print(f"\rIteration {i:05d}; Div {outer_div:6.6f}", end='')

        u_last_k = u_last_t
        min_div = torch.inf
        for k in range(30):
            u_last_k_ = halfar_ice(tstep, u_last_t, u_last_k)
            div = torch.linalg.norm(u_last_k - u_last_k_)
            # FIXME HACK b/c the divergence goes down and then explodes for some reason. Still debugging.
            if div > min_div:
                break
            min_div = div
            u_last_k = u_last_k_
        u_last_t = u_last_k
        sols.append(u_last_t[0])
    print()

    # FIXME these plots are reversed across x for some reason. Run heat_conduction_simple.py and compare.
    #  Also see that initial condition is reversed.
    # sols = sols[:, :, ::-1]

    plt.figure(dpi=500, figsize=(8, 2))
    plt.plot(xx, sols[0].detach(), color='purple', linestyle='--', label="Initial Condition")
    for t_sols in sols[1:-1:5]:
        plt.plot(xx, t_sols.detach(), linewidth=0.1, color='orange', alpha=0.5)
    plt.plot(xx, sols[-1].detach(), color='purple', label="Final Solution")

    # remaining_heat = sols.sum(axis=-1)
    #
    # for di in range(len(diffusivities)):
    #     print(f"Grad at termination with diffusivity {di}:")
    #     remaining_heat[di, -1].backward()
    #     print(diffusivities.grad)
    #     print(times.grad)
    #
    #     # Zero out the grads.
    #     diffusivities.grad.zero_()
    #     times.grad.zero_()

    peak = sols[-1].max().detach()
    analytical_solution_maybe = fillna(peak * (1. - torch.abs(xx))**.5)
    plt.plot(xx, analytical_solution_maybe, color='white', linestyle='--',
             linewidth=0.5, label="Analytical Equilibrium (Maybe)")
    plt.gca().set_aspect('equal')
    plt.gca().set_facecolor('lightgray')
    plt.gcf().set_facecolor('lightgray')

    plt.tight_layout()

    plt.show()

    print(f"Glacier Mass Before", sols[0].sum().item())
    print(f"Glacier Mass After (Num)", sols[-1].sum().item())
    print(f"Glacier Mass After (Ana)", analytical_solution_maybe.sum().item())

    return


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()

    plt.show()
