import torch
from .. import fenics_overloaded as fe
import torch_fenics
from .utils import stable_gamma


class HalfarIceNonLinearPolarImpT(torch_fenics.FEniCSModule):

    Ti = 0
    Ri = 1

    def __init__(self, n_elements=64, solver_parameters: dict = None):

        super().__init__()

        self.solver_parameters = solver_parameters or dict()

        self.n_elements = n_elements

        # Instead of a 1D mesh, we'll use a 2D mesh so we can have implicit time.
        self.mesh = fe.UnitSquareMesh(n_elements, n_elements)

        self.V = fe.FunctionSpace(self.mesh, 'Lagrange', 1)

        self.Vb = fe.FunctionSpace(fe.UnitIntervalMesh(n_elements), 'Lagrange', 1)

        self.x = fe.SpatialCoordinate(self.mesh)

        # Instead of the above, put a boundary condition on the spacial edges (second dimension of the mesh).
        self.bc_r = fe.DirichletBC(
            self.V.sub(self.Ri),
            # Boundary condition of zero.
            fe.Constant(0.0),
            'on_boundary'
        )

        self.v_test = fe.TestFunction(self.V)

    def solve(self, u_init, ice_density, log_flow_rate):

        V = self.V
        Ti, Ri = self.Ti, self.Ri
        # TODO figure out why this has to be declared inside the solve for gradients to propagate.
        u = fe.Function(self.V)
        vte = self.v_test

        # Set a boundary condition only along the 0
        def t0_boundary(x, on_boundary):
            return on_boundary and fe.near(x[Ti], 0)

        bc_t0 = fe.DirichletBC(
            self.V.sub(0),
            u_init,
            t0_boundary
        )

        du = fe.grad(u)

        r = self.x[Ri]

        G = stable_gamma(
            rho=ice_density,
            lA=log_flow_rate,
            logf=fe.ln,
            expf=fe.exp
        )

        weak_form_residuum = (
            vte * r * fe.grad(u)[Ti] * V.dx(Ri)
            + r * G * fe.dot(
                # So the middle bit is usually grad(utr) * fe.sqrt(grad(utr).dot(grad(utr)) + 1e-6) ** (n-1),
                #  but that can be simplified for the typical n=3. Same with utr ** (n+2)
                fe.grad(u)[Ri] * (u ** 5) * (du[Ri] ** 2),
                fe.grad(vte)[Ri]
            ) * V.dx(Ri)
        )

        fe.solve(
            weak_form_residuum == 0,
            u,
            bcs=[self.bc_r, bc_t0],
            solver_parameters=self.solver_parameters
        )

        return u

    def input_templates(self):
        # u_init, ice_density, log_flow_rate
        return fe.Function(self.Vb), fe.Constant(0), fe.Constant(0)
