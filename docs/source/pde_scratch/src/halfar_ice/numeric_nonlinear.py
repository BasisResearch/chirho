import torch
from .. import fenics_overloaded as fe
import torch_fenics
from .utils import stable_gamma


class HalfarIceNonLinearPolar(torch_fenics.FEniCSModule):

    def __init__(self, n_elements=64, solver_parameters: dict = None):

        super().__init__()

        self.solver_parameters = solver_parameters or dict()

        self.n_elements = n_elements

        # Mesh on which to build finite element function space.
        self.mesh = fe.IntervalMesh(n_elements, -2., 2.)

        self.V = fe.FunctionSpace(self.mesh, 'Lagrange', 1)

        self.x = fe.SpatialCoordinate(self.mesh)

        # Homogeneous Dirichlet boundary condition.
        self.bc = fe.DirichletBC(
            self.V,
            # Boundary condition of zero.
            fe.Constant(0.0),
            'on_boundary'
        )

        self.v_test = fe.TestFunction(self.V)

    def solve(self, tstep, u_last_t, ice_density, log_flow_rate):

        # TODO figure out why this has to be declared inside the solve for gradients to propagate.
        u = fe.Function(self.V)
        vte = self.v_test
        ult = u_last_t

        du = fe.grad(u)

        r = self.x[0]

        G = stable_gamma(
            rho=ice_density,
            lA=log_flow_rate,
            logf=fe.ln,
            expf=fe.exp
        )

        weak_form_residuum = (
            u * vte * r * fe.dx
            - ult * vte * r * fe.dx
            + tstep * r * G * fe.dot(
                # So the middle bit is usually grad(utr) * fe.sqrt(grad(utr).dot(grad(utr)) + 1e-6) ** (n-1),
                #  but that can be simplified for the typical n=3. Same with utr ** (n+2)
                fe.grad(u) * (u ** 5) * fe.dot(du, du),
                fe.grad(vte)
            ) * fe.dx
        )

        # Solve from initial condition out to tstep in time.
        fe.solve(
            weak_form_residuum == 0,
            u,
            self.bc,
            solver_parameters=self.solver_parameters
        )

        return u

    def input_templates(self):
        # tstep, u_last_t, ice_density, log_flow_rate
        return fe.Constant(0), fe.Function(self.V), fe.Constant(0), fe.Constant(0)
