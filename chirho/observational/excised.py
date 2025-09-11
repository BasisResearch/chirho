import math
from numbers import Number

import torch
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import _standard_normal, broadcast_all
from pyro.distributions.torch_distribution import TorchDistribution, TorchDistributionMixin
import pyro.distributions as dist


class Normal(ExponentialFamily):
    r"""
    Creates a normal (also called Gaussian) distribution parameterized by
    :attr:`loc` and :attr:`scale`.

    Example::

        >>> m = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # normally distributed with loc=0 and scale=1
        tensor([ 0.1046])

    Args:
        loc (float or Tensor): mean of the distribution (often referred to as mu)
        scale (float or Tensor): standard deviation of the distribution
            (often referred to as sigma)
    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.loc

    @property
    def stddev(self):
        return self.scale

    @property
    def variance(self):
        return self.stddev.pow(2)

    def __init__(self, loc, scale, validate_args=None):
        self.loc, self.scale = broadcast_all(loc, scale)
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(Normal, self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Normal, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        super(Normal, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            return torch.normal(self.loc.expand(shape), self.scale.expand(shape))

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + eps * self.scale

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # compute the variance
        var = (self.scale ** 2)
        log_scale = math.log(self.scale) if isinstance(self.scale, Number) else self.scale.log()
        return -((value - self.loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return 0.5 * (1 + torch.erf((value - self.loc) * self.scale.reciprocal() / math.sqrt(2)))

    def icdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return self.loc + self.scale * torch.erfinv(2 * value - 1) * math.sqrt(2)

    def entropy(self):
        return 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(self.scale)

    @property
    def _natural_params(self):
        return (self.loc / self.scale.pow(2), -0.5 * self.scale.pow(2).reciprocal())

    def _log_normalizer(self, x, y):
        return -0.25 * x.pow(2) / y + 0.5 * torch.log(-math.pi / y)




class Normal(torch.distributions.Normal, TorchDistributionMixin):
    pass




class ExcisedNormal(TorchDistribution):
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}  
    support = constraints.real  # we don't want to use intervals here, they might differ between factual points
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.loc

    @property
    def stddev(self):
        return self.scale

    @property
    def variance(self):
        return self.stddev.pow(2)
    
    @property
    def intervals(self):
        return self._intervals

    @property
    def base_normal(self):
        return self._base_normal

    @property
    def self_base_uniform(self):
        return self._base_uniform

    @property
    def interval_masses(self):
        return self._interval_masses

    @property
    def lcdfs(self):
        return self._lcdfs

    @property
    def removed_pr_mass(self):
        return self._removed_pr_mass

    @property
    def normalization_constant(self):
        return self._normalization_constant

    def __init__(self, loc, scale, intervals, validate_args=None):

        lows, highs = zip(*intervals)   # each is a tuple of tensors/scalars
        
        self.loc, self.scale, *all_edges = broadcast_all(loc, scale, *lows, *highs)
        
        n = len(lows)
        lows  = all_edges[:n]
        highs = all_edges[n:]
        self._intervals = list(zip(lows, highs))

        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(ExcisedNormal, self).__init__(batch_shape, validate_args=validate_args)


        self._base_normal = dist.Normal(self.loc, self.scale, validate_args=validate_args)

        self._base_uniform = dist.Uniform(torch.zeros_like(self.loc), torch.ones_like(self.loc))

        # these do not vary and do not depend on sample shape, can be pre-computed
        self._interval_masses = []
        self._lcdfs = []
        self._removed_pr_mass = torch.zeros_like(self.loc)

        for low, high in self.intervals:
            lower_cdf = self.base_normal.cdf(low)
            upper_cdf = self.base_normal.cdf(high)
            interval_mass = upper_cdf - lower_cdf
            self._interval_masses.append(interval_mass)
            self._lcdfs.append(lower_cdf)
            self._removed_pr_mass += interval_mass

        self._normalization_constant = torch.tensor(1.) - self._removed_pr_mass



    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(ExcisedNormal, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)

        new._intervals = [(low.expand(batch_shape), high.expand(batch_shape))
                      for low, high in self._intervals]
        new._base_normal = dist.Normal(new.loc, new.scale, validate_args=False)
        new._base_uniform = dist.Uniform(torch.zeros_like(new.loc), torch.ones_like(new.loc))

        new._interval_masses = [im.expand(batch_shape) for im in self.interval_masses]
        new._lcdfs = [lcdf.expand(batch_shape) for lcdf in self.lcdfs]

        new._removed_pr_mass = self.removed_pr_mass.expand(batch_shape)
        new._normalization_constant = self.normalization_constant.expand(batch_shape)

        super(ExcisedNormal, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new


    # TODO modify to account for excised intervals
    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
     
        normalization_constant_expanded = self.normalization_constant.expand(shape)

        with torch.no_grad():
            uniform_sample = self._base_uniform.sample(sample_shape = sample_shape).to(self.loc.device)

        v = uniform_sample * normalization_constant_expanded

        for l_cdf, mass in zip(self.lcdfs, self.interval_masses):
            v = torch.where(v >= l_cdf, v + mass, v)

        x_new = self.base_normal.icdf(v)

        return x_new 

    # TODO modify to account for excised intervals
    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + eps * self.scale



    def log_prob(self, value):

        shape = value.shape

        mask = torch.zeros(shape, dtype=torch.bool, device=self.loc.device)

        for interval in self.intervals:
            low, high = interval
            mask = mask | ((value >= low) & (value <= high))

        normalization_constant_expanded = self.normalization_constant.expand(shape)

        lp = self.base_normal.log_prob(value) - torch.log(normalization_constant_expanded)
        
        return torch.where(mask, torch.tensor(-float("inf"), device=self.loc.device), lp)
            
    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        
        base_cdf = self.base_normal.cdf(value)

        for l_cdf, mass in zip(self.lcdfs, self.interval_masses):
            cdf = torch.where(base_cdf >= l_cdf, base_cdf - ( self.base_normal.cdf(value) - l_cdf), base_cdf)

        return cdf / self.normalization_constant



        # if self._validate_args:
        #     self._validate_sample(value)
        # # compute the variance
        # var = (self.scale ** 2)
        # log_scale = math.log(self.scale) if isinstance(self.scale, Number) else self.scale.log()
        # return -((value - self.loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))





# simple tests for now

true_mean = torch.tensor([[[1.0]] , [[3.0]]])
true_stddev = torch.tensor([[[2.]] , [[3.]]])


interval_flat = [(torch.tensor(-1.0), torch.tensor(1.))]
intervals_flat = [(-2.0, -1.0), (1.0, 2.0)]
interval_shaped = [(true_mean - true_stddev, true_mean + true_stddev)]
intervals_shaped = [(true_mean - 2 * true_stddev, true_mean - true_stddev),
                    (true_mean + true_stddev, true_mean + 2 * true_stddev)]



interval_types = {
    "flat": interval_flat,
    "flat_intervals": intervals_flat,
    "shaped": interval_shaped,
    "shaped_intervals": intervals_shaped
}

new_batch_shape = (2, 1, 3)


for key, interval in interval_types.items():
    excised_normal = ExcisedNormal(true_mean, true_stddev, interval)

    excised_normal_expanded = excised_normal.expand(new_batch_shape)

    print(f"{key}")
    interval_counter = 0

    print("Original intervals:") 
    for low, high in excised_normal.intervals:
        print(f"Interval {interval_counter} edge shapes: [{low.shape}, {high.shape}]")
        print(f"flattened_interval {interval_counter}: [{low.flatten()}, {high.flatten()}]")
        interval_counter += 1

    print("Expanded intervals:")
    for i, (low, high) in enumerate(excised_normal_expanded._intervals):
        print(f" Interval {i} edge shapes: [{low.shape}, {high.shape}]")
        print(f" Flattened: [{low.flatten()}, {high.flatten()}]")

    print(f"Original mean shape: {excised_normal.mean.shape}")
    print(f"Expanded mean shape: {excised_normal_expanded.mean.shape}")
    print(f"Original stddev shape: {excised_normal.stddev.shape}")
    print(f"Expanded stddev shape: {excised_normal_expanded.stddev.shape}")
    print(f"Variance shape: {excised_normal_expanded.variance.shape}")
    print(f"Support: {excised_normal_expanded.support}")
    #print(f"Log prob of 0: {excised_normal.log_prob(torch.tensor(0.0))}\n")

    if key == "shaped":
        assert all([torch.allclose(lcdf, torch.tensor(.1587), atol = 1e-4) for lcdf in excised_normal.lcdfs])
        excised_normal_expanded.interval_masses
        assert all([torch.allclose(lcdf, torch.tensor(.1587), atol = 1e-4) for lcdf in excised_normal_expanded.lcdfs])

        assert all([torch.allclose(im, torch.tensor(.6827), atol = 1e-4) for im in excised_normal.interval_masses])
        assert all([torch.allclose(im, torch.tensor(.6827), atol = 1e-4) for im in excised_normal_expanded.interval_masses])


    assert excised_normal.mean.shape == true_mean.shape, f"{key}: mean shape mismatch on instantiation"
    assert excised_normal.stddev.shape == true_stddev.shape, f"{key}: stddev shape mismatch on instantiation"
    assert excised_normal.variance.shape == true_stddev.shape, f"{key}: variance shape mismatch on instantiation"

    for low, high in excised_normal._intervals:
        assert low.shape == true_mean.shape, f"{key}: low edge shape mismatch on instantiation"
        assert high.shape == true_mean.shape, f"{key}: high edge shape mismatch on instantiation"

    assert excised_normal.loc.shape == true_mean.shape, f"{key}: loc shape mismatch on instantiation"
    assert excised_normal.scale.shape == true_stddev.shape, f"{key}: scale shape mismatch on instantiation"
    assert excised_normal._base_normal.batch_shape == true_mean.shape, f"{key}: base normal batch shape mismatch on instantiation"

 
    assert excised_normal_expanded.mean.shape == torch.Size(new_batch_shape), f"{key}: mean shape mismatch on expand"
    assert excised_normal_expanded.stddev.shape == torch.Size(new_batch_shape), f"{key}: stddev shape mismatch on expand"
    assert excised_normal_expanded.variance.shape == torch.Size(new_batch_shape), f"{key}: variance shape mismatch on expand"

    for low, high in excised_normal_expanded._intervals:
        assert low.shape == torch.Size(new_batch_shape), f"{key}: low edge shape mismatch on expand"
        assert high.shape == torch.Size(new_batch_shape), f"{key}: high edge shape mismatch on expand"

    assert excised_normal_expanded.loc.shape == torch.Size(new_batch_shape), f"{key}: loc shape mismatch on expand"
    assert excised_normal_expanded.scale.shape == torch.Size(new_batch_shape), f"{key}: scale shape mismatch on expand"
    assert excised_normal_expanded._base_normal.batch_shape == torch.Size(new_batch_shape), f"{key}: base normal batch shape mismatch on expand"

    assert len(excised_normal.interval_masses) == len(excised_normal.intervals), f"{key}: interval masses length mismatch on instantiation"
    assert len(excised_normal.lcdfs) == len(excised_normal.intervals), f"{key}: lcdfs length mismatch on instantiation"
    assert len(excised_normal_expanded.interval_masses) == len(excised_normal_expanded.intervals), f"{key}: interval masses length mismatch on expand"
    assert len(excised_normal_expanded.lcdfs) == len(excised_normal_expanded.intervals), f"{key}: lcdfs length mismatch on expand"



    sample = excised_normal.sample(sample_shape = (400,))
    sample_expanded = excised_normal_expanded.sample(sample_shape = (400,))

    assert sample.shape == (400,2,1,1)
    assert sample_expanded.shape == (400,2,1,3)

    for interval in excised_normal.intervals:
        low, high = interval
        assert torch.all((sample <= low) | (sample >= high))

    for interval in excised_normal_expanded.intervals:
        low, high = interval
        assert torch.all((sample_expanded <= low) | (sample_expanded >= high))

    candidates = (torch.rand(sample.shape) - .5) * 4
    candidates_expanded = (torch.rand(sample_expanded.shape) - .5) * 4


    log_probs = excised_normal.log_prob(candidates)
    log_probs_expanded = excised_normal_expanded.log_prob(candidates_expanded)

    mask = torch.zeros(candidates.shape, dtype=torch.bool, device=candidates.device)
    mask_expanded = torch.zeros(candidates_expanded.shape, dtype=torch.bool, device=candidates_expanded.device)

    for interval in excised_normal.intervals:
        low, high = interval
        mask = mask | ((candidates >= low) & (candidates <= high))
    
    for interval in excised_normal_expanded.intervals:
        low, high = interval
        mask_expanded = mask_expanded | ((candidates_expanded >= low) & (candidates_expanded <= high))

    assert torch.all(torch.where(mask, log_probs == -float("inf"), True))
    assert torch.all(torch.where(mask_expanded, log_probs_expanded == -float("inf"), True))
    assert torch.all(torch.where(~mask, log_probs != -float("inf"), True))
    assert torch.all(torch.where(~mask_expanded, log_probs_expanded != -float("inf"), True))
        
    cdf_candidates = excised_normal.cdf(candidates)
    cdf_candidates_expanded = excised_normal_expanded.cdf(candidates_expanded)

    
    assert True