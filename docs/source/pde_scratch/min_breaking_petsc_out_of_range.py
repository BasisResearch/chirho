import fenics as fe
import matplotlib.pyplot as plt

if __name__ == "__main__":
    n_elements = 32
    mesh = fe.UnitIntervalMesh(n_elements)

    # Define a Function Space
    V = fe.FunctionSpace(
        mesh,
        "Lagrange",
        1
    )

    # The value of the solution on the boundary
    u_d = fe.Constant(0.0)

    # A function to return whether we are on the boundary
    def boundary_boolean_function(x, on_boundary):
        return on_boundary

    # The homogeneous Dirichlet Boundary Condition
    boundary_condition = fe.DirichletBC(
        V,
        u_d,
        boundary_boolean_function,
    )

    # The initial condition, u(t=0, x) = sin(2 * pi * x ** 2.)
    initial_condition = fe.Expression(
        "sin(2.0 * 3.141 * x[0] * x[0])",
        degree=1
    )

    # Discretize the initial condition
    u_old = fe.interpolate(
        initial_condition,
        V
    )

    # The time stepping of the implicit Euler discretization (=dt)
    time_step_length = 0.1
    diffusivity = fe.Constant(.5)

    du = fe.TrialFunction(V)
    u_ = fe.Function(V)
    v = fe.TestFunction(V)

    weak_form_residuum = (
        u_ * v * fe.dx
        - u_old * v * fe.dx
        + time_step_length * diffusivity * fe.dot(
            fe.grad(u_), fe.grad(v)
        ) * fe.dx
    )

    # J = fe.derivative(weak_form_residuum, u_, du)

    # # The function we will be solving for at each point in time
    # u_solution = fe.Function(V)


    # problem = fe.NonlinearVariationalProblem(
    #     weak_form_residuum, u_solution#, J=J
    # )
    #
    # fe.NonlinearVariationalSolver(problem).solve()



    # Finite Element Assembly, BC imprint & solving the linear system
    fe.solve(
        weak_form_residuum == 0,
        u_,
        boundary_condition,
    )
