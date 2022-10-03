import pyro

from pyro.distributions import Delta

from pyro.infer.reparam import Reparam

class KernelABCReparam(Reparam):
    def __init__(self, kernel: pyro.contrib.gp.Kernel):
        self.kernel = kernel
        super().__init__()

    def apply(self, msg):
        name = msg["name"]
        fn = msg["fn"]
        observed_value = msg["value"]
        is_observed = msg["is_observed"]

        # TODO fail gracefully if applied to the wrong site
        # TODO disambiguate names with multi world counterfacutals...
        assert is_observed
        assert isinstance(fn, pyro.distributions.MaskedDistribution)
        fn = fn.base_dist
        assert isinstance(fn, pyro.distributions.Delta)
        assert fn.event_dim == self.kernel.input_dim

        computed_value = fn.v
        pyro.factor(name + "_factor", self.kernel(computed_value, observed_value))

        new_fn = pyro.distributions.Delta(observed_value, event_dim=fn.event_dim).mask(False)
        return {"fn": new_fn, "value": observed_value, "is_observed": True}





@reparam(config={"x": KernelABCReparam(...)})
def model(x_obs):
  ...
  x_obs = pyro.sample("x", Delta(x), obs=x_obs)
  ...