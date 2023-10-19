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
    values = (1.0 - x**2 + 0.3 * torch.sin(2. * torch.pi * x) ** 2.)
    return torch.where(condition, values, torch.zeros_like(x))


def fillna(tensor):
    return torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)


def _t0(r0, h0, gamma=1.):
    return (1. / (18. * gamma)) * ((7. / 4.) ** 3.) * (r0 ** 4. / h0 ** 7.)


def halfar_ice_analytical(r, t, h0, r0, gamma=1.):
    t0 = _t0(r0, h0, gamma)

    hterm = (h0 * (t0 / t) ** (1. / 9.))
    rterm = (1. - ((t0 / t) ** (1. / 18.) * (r / r0)) ** (4. / 3.)) ** (3. / 7.)

    hhalf = hterm * rterm
    rn = len(hhalf)
    hhalf[:rn // 2] = torch.flip(hhalf[rn // 2:], dims=(0,))

    return fillna(hhalf)


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
                # So the middle bit is usually grad(utr) * fe.sqrt(grad(utr).dot(grad(utr)) + 1e-6) ** (n-1),
                #  but that can be simplified for the typical n=3. Same with utr ** (n+2)
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


def anamain():
    h0 = 1.
    r0 = 1.

    xx = torch.linspace(-2., 2., 512)

    sols = []

    t = 1.
    tstep = 1.0
    for i in range(1000):
        h = halfar_ice_analytical(xx, t, h0, r0)
        sols.append(h)
        t += tstep

    fig, ax = plt.subplots(dpi=500, figsize=(4, 2))
    plot_sols(xx, sols, ax, skip=5)

    # Sum across each solution to check if mass is being conserved.
    # FIXME for some reason this was sigsegv-ing :/
    # ice_sum = torch.stack(sols).sum(axis=-1)
    ice_sum = [sol.sum() for sol in sols]
    plt.figure()
    plt.plot(ice_sum)


def plot_sols(xx, sols, ax, skip=5, color1='purple', color2='orange', thickness1=1.0, thickness2=0.1):

    ax.plot(xx, sols[0].detach(), color=color1, linestyle='--', label="Initial Condition", linewidth=thickness1)
    for t_sols in sols[1:-1:skip]:
        ax.plot(xx, t_sols.detach(), color=color2, alpha=0.5, linewidth=thickness2)
    ax.plot(xx, sols[-1].detach(), color=color1, label="Final Solution", linewidth=thickness1)


def main():

    halfar_ice = HalfarIce(511)

    xx = torch.linspace(-2., 2., halfar_ice.n_elements + 1)

    # # A tricky initial glacier.
    # u_init = initial_curve(xx)\
    #     [None, :].double().clone().detach().requires_grad_(True)  # Make this a leaf tensor.

    # Init with analytically known progression in time.
    h0, r0 = 1., 1.
    u_init = halfar_ice_analytical(r=xx, t=1., h0=h0, r0=r0)\
        [None, :].double().clone().detach().requires_grad_(True)

    u_last_t = u_init

    plt.show()

    sols = [u_last_t[0]]

    tstep = tt(0.005).double()[None, None]
    end_t = 1.
    for i in range(1000):
        if len(sols) > 1:
            outer_div = torch.linalg.norm(sols[-1] - sols[-2])
            if outer_div < 1e-6:
                break  # break out if the glacier has reached equilibrium.
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
        end_t += tstep.item()
        sols.append(u_last_t[0])
    print()

    # FIXME these plots are reversed across x for some reason. Run heat_conduction_simple.py and compare.
    #  Also see that initial condition is reversed.
    # sols = sols[:, :, ::-1]

    fig, ax = plt.subplots(dpi=500, figsize=(4, 2))
    plot_sols(xx, sols, ax, skip=5)

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

    analytical_solution_maybe = halfar_ice_analytical(xx, t=end_t, h0=h0, r0=r0)
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
    # anamain()

    plt.show()
