import torch
from .. import fenics_overloaded as fe
import torch_fenics


class HalfarIceLinear(torch_fenics.FEniCSModule):

    def __init__(self, n_elements=64):

        super().__init__()

        self.n_elements = n_elements

        # Mesh on which to build finite element function space.
        self.mesh = fe.IntervalMesh(n_elements, -2., 2.)

        self.V = fe.FunctionSpace(self.mesh, 'Lagrange', 1)

        # See https://fenicsproject.org/pub/tutorial/html/._ftut1020.html for details.
        self.W = fe.VectorFunctionSpace(self.V.mesh(), 'P', self.V.ufl_element().degree())

        # self.x = fe.SpatialCoordinate(self.mesh)

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
        problem = lhs == rhs

        u_sol = fe.Function(self.V)

        # Solve from initial condition out to tstep in time.
        fe.solve(
            problem,
            u_sol,
            self.bc
        )

        return u_sol

    def input_templates(self):
        # tstep, u_last
        return fe.Constant(0), fe.Function(self.V), fe.Function(self.V)
