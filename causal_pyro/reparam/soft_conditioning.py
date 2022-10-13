import pyro
from pyro.contrib.gp.kernels import Kernel
from pyro.distributions import Delta, MaskedDistribution
from pyro.infer.reparam.reparam import Reparam


class KernelABCReparam(Reparam):
    """
    Reparametrizer that allows approximate conditioning on a ``pyro.deterministic`` site.
    """
    def __init__(self, kernel: Kernel):
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
        assert isinstance(fn, MaskedDistribution)
        fn = fn.base_dist
        assert isinstance(fn, Delta)
        assert fn.event_dim == self.kernel.input_dim

        computed_value = fn.v
        pyro.factor(name + "_factor", self.kernel(computed_value, observed_value))

        new_fn = Delta(observed_value, event_dim=fn.event_dim).mask(False)
        return {"fn": new_fn, "value": observed_value, "is_observed": True}
