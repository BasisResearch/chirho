from .abstractions import ModelWithMarginalDensity, FDModelFunctionalDensity
from scipy.stats import multivariate_normal, gaussian_kde
import pyro
import pyro.distributions as dist
from chirho.robust.ops import Point, T
from typing import Dict, Callable
import torch
import numpy as np


class MultivariateNormalwDensity(ModelWithMarginalDensity):

    def __init__(self, mean, scale_tril, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mean = mean
        self.scale_tril = scale_tril

        # Convert scale_tril to a covariance matrix.
        self.cov = scale_tril @ scale_tril.T

    def density(self, x):
        return multivariate_normal.pdf(x, mean=self.mean, cov=self.cov)

    def forward(self):
        return pyro.sample("x", dist.MultivariateNormal(self.mean, scale_tril=self.scale_tril))


class PerturbableNormal(FDModelFunctionalDensity):

    def __init__(self, *args, mean, scale_tril, **kwargs):
        super().__init__(*args, **kwargs)

        self.ndims = mean.shape[-1]
        self.model = MultivariateNormalwDensity(
            mean=mean,
            scale_tril=scale_tril
        )


class KDEOnPoints(ModelWithMarginalDensity):

    def __init__(self, points: Point[T], datadim: int = 0, **kdekwargs):
        super().__init__()

        if "weights" in kdekwargs:
            # TODO just need to hang onto them explicitly and index into them when conditioning?
            raise NotImplementedError("Weights are not supported yet.")

        self.points = points
        self.vars = list(points.keys())
        self.datadim = datadim
        self.num_points = len(points[self.vars[self.datadim]])
        # Sanity check.
        for var in self.vars:
            assert len(points[var]) == self.num_points

        self._vardims = {
            # Get the size of the tensors not including the data dimension.
            var: int(np.product(points[var].shape)/points[var].shape[self.datadim])
            for var in self.vars
        }

        self._kdekwargs = kdekwargs
        self._kde = None

    def density(self, **points):

        for var in self.vars:
            assert var in points

        num_points = points[self.vars[0]].shape[self.datadim]
        flat_points = self.flatten_points_np(points=points, num_points=num_points)
        assert flat_points.shape == (num_points, int(np.sum(list(self._vardims.values()))))

        return self.kde.pdf(flat_points.T)

    def forward(self, num_samples: int = 1):

        flat_points = self.kde.resample(size=num_samples).T

        # TODO abstract out to utility method.
        unflat_points = dict()
        last_col = 0
        for var in self.vars:
            # TODO vfo1kl not currently saving the shapes so this just returns a max 2d tensor for now, even if
            #  the original shape was higher dimensional.
            unflat_points[var] = torch.tensor(flat_points[:, last_col:last_col + self._vardims[var]])
            assert unflat_points[var].shape == (num_samples, self._vardims[var])
            last_col += self._vardims[var]

        return unflat_points

    @property
    def kde(self):
        if self._kde is None:
            try:
                self._kde = gaussian_kde(self.flatten_points_np().T, **self._kdekwargs)
            except np.linalg.LinAlgError as e:
                raise np.linalg.LinAlgError(f"Did you condition on an identity and then not marginalize it out? {e}")
        return self._kde

    def flatten_points_np(self, points=None, num_points=None):
        if points is None:
            points = self.points
        if num_points is None:
            num_points = self.num_points

        flat_points = np.empty((num_points, int(np.sum(list(self._vardims.values())))), dtype=float)
        last_col = 0
        for var in self.vars:
            assert points[var].shape[self.datadim] == num_points
            flat_points[:, last_col:last_col + self._vardims[var]] = \
                points[var].detach().numpy().reshape(num_points, -1)
            last_col += self._vardims[var]

        return flat_points

    def _iter_points(self):
        # TODO discard after vectorization
        for i in range(self.num_points):
            yield {var: self.points[var][i] for var in self.vars}

    def condition(self, cond: Callable) -> "KDEOnPoints":
        # TODO vectorize
        # Limit the dataset to points that are consistent with the cond callable.
        new_points = dict()
        for point in self._iter_points():
            if cond(point):
                for var in self.vars:
                    new_points.setdefault(var, []).append(point[var])

        for var in self.vars:
            new_points[var] = torch.stack(new_points[var])

        if self.datadim != 0:
            # TODO transpose stacks above.
            raise NotImplementedError("Only datadim=0 is supported for now.")

        return KDEOnPoints(new_points, datadim=self.datadim, **self._kdekwargs)

    def marginalize(self, *keepvars: str) -> "KDEOnPoints":
        new_points = {var: self.points[var] for var in self.vars if var in keepvars}
        return KDEOnPoints(new_points, datadim=self.datadim, **self._kdekwargs)
