import chirho.contrib.experiments.closed_form as cfe
import pyro.distributions as dist
from torch import tensor as tnsr
import numpy as np
from scipy.optimize import minimize
import chirho.contrib.compexp as ep
import pyro
from collections import OrderedDict


def opt_ana_with_scipy(problem: cfe.CostRiskProblem):
    if problem.ana_opt_traj is not None:
        return problem.ana_opt_traj

    traj = [problem.theta0.detach().numpy()]

    def callback(xk):
        traj.append(xk)

    def numpy_loss(theta: np.ndarray):
        return problem.ana_loss(tnsr(theta).double()).numpy()

    def numpy_grad(theta: np.ndarray):
        return problem.ana_loss_grad(tnsr(theta).double()).numpy()

    theta0 = problem.theta0.numpy()
    res = minimize(numpy_loss, theta0, method='BFGS', callback=callback, jac=numpy_grad, tol=1e-6)

    assert np.allclose(traj[-1], res.x)

    problem.ana_opt_traj = np.array(traj)

    return problem.ana_opt_traj


def opt_opt_tabi_with_scipy(problem: cfe.CostRiskProblem):
    """
    This uses TABI with an optimal guide to get exact estimates of the objective, acts as a baseline test.
    """
    traj = [problem.theta0.detach().numpy()]

    def callback(xk):
        traj.append(xk)

    def numpy_loss(theta: np.ndarray):
        theta = tnsr(theta).double()

        mu_star, Sigma_star = cfe.optimal_tabi_proposal_nongrad(theta, problem.Q, problem.Sigma)

        # When used with guides, expectations require the dict return. Is a TODO FIXME d78107gkl.
        def opt_guide():
            return OrderedDict(z=pyro.sample('z', dist.MultivariateNormal(mu_star, Sigma_star)))

        def model():
            return OrderedDict(z=problem.model())

        risk = ep.E(
            f=lambda s: cfe.risk_curve(theta, problem.Q, s['z']).squeeze(),
            name='risk',
            guide=opt_guide
        )
        risk._is_positive_everywhere = True

        cost = ep.C(theta @ theta, requires_grad=True) + ep.C(problem.c) * risk

        eh = ep.ImportanceSamplingExpectationHandler(num_samples=1)
        eh.register_guides(cost, model, auto_guide=None)

        with eh:
            return cost(model).detach().clone().numpy()

    theta0 = problem.theta0.numpy()
    res = minimize(numpy_loss, theta0, method='BFGS', callback=callback, tol=1e-6)

    assert np.allclose(traj[-1], res.x)

    return np.array(traj)
