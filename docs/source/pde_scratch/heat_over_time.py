# HACK when running osx-64 conda env on M2 mac. Side effects could be an issue:
# https://stackoverflow.com/a/53014308
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import fenics_overloaded as fe
import torch_fenics

# A combination of two sources:
# https://github.com/Ceyron/machine-learning-and-simulation/blob/main/english/fenics/heat_conduction_simple.py
# https://github.com/barkm/torch-fenics/blob/master/examples/poisson.py


class HeatOverTime(torch_fenics.FEniCSModule):

    def __init__(self, n_elements=32):

        super().__init__()

        self.n_elements = n_elements

        # Mesh on which to build finite element function space.
        self.mesh = fe.UnitIntervalMesh(n_elements)

        # Linear lagrangian finite element function.
        self.V = fe.FunctionSpace(self.mesh, 'Lagrange', 1)

        # Homogeneous Dirichlet boundary condition.
        self.bc = fe.DirichletBC(
            self.V,
            # Boundary condition of zero.
            fe.Constant(0.0),
            'on_boundary'
        )

        # The initial condition, u(t=0, x) = sin(pi * x)
        self.initial_condition = fe.Expression(
            "sin(2.0 * 3.141 * x[0] * x[0])",
            degree=1
        )

        self.u_init = fe.interpolate(self.initial_condition, self.V)

        self.u_trial = fe.TrialFunction(self.V)
        self.v_test = fe.TestFunction(self.V)

    def solve(self, diffusivity, tstep):

        utr = self.u_trial
        vte = self.v_test
        uin = self.u_init

        weak_form_residuum = (
            utr * vte * fe.dx
            - uin * vte * fe.dx
            + tstep * diffusivity * fe.dot(
                fe.grad(utr), fe.grad(vte)
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
        # diffusivity, tstep
        return fe.Constant(0), fe.Constant(0)


def main():

    diffusivities = torch.linspace(0.1, 1.0, 3, requires_grad=True, dtype=torch.float64)
    times = torch.linspace(0.0, 1.0, 5, requires_grad=True, dtype=torch.float64)

    all_d_t_combos = torch.cartesian_prod(diffusivities, times)
    ds = all_d_t_combos[:, 0]
    ts = all_d_t_combos[:, 1]

    heat_over_time = HeatOverTime(n_elements=32)

    print("test scalar", heat_over_time(diffusivities[None, 0, None], times[None, -1, None]))

    sols = heat_over_time(ds[:, None], ts[:, None]).reshape(3, 5, heat_over_time.n_elements + 1)

    # FIXME these plots are reversed across x for some reason. Run heat_conduction_simple.py and compare.
    #  Also see that initial condition is reversed.
    # sols = sols[:, :, ::-1]

    for diff_sols in sols:
        plt.figure()
        for t_sols in diff_sols:
            plt.plot(t_sols.detach().numpy())

    remaining_heat = sols.sum(axis=-1)

    for di in range(len(diffusivities)):
        print(f"Grad at termination with diffusivity {di}:")
        remaining_heat[di, -1].backward()
        print(diffusivities.grad)
        print(times.grad)

        # Zero out the grads.
        diffusivities.grad.zero_()
        times.grad.zero_()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()

    plt.show()
