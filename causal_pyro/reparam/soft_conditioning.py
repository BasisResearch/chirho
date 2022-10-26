<<<<<<< HEAD
from typing import Optional

=======
>>>>>>> 7e08979 (added KernelABCReparam and simple test)
import pyro
from pyro.contrib.gp.kernels import Kernel
from pyro.distributions import Delta, MaskedDistribution
from pyro.infer.reparam.reparam import Reparam


class KernelABCReparam(Reparam):
<<<<<<< HEAD
    """
    Reparametrizer that allows approximate conditioning on a ``pyro.deterministic`` site.
    """

=======
>>>>>>> 7e08979 (added KernelABCReparam and simple test)
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
<<<<<<< HEAD


class AutoSoftConditioning(pyro.infer.reparam.strategies.Strategy):
    @staticmethod
    def _is_deterministic(msg):
        return (
            msg["type"] == "sample"
            and msg["is_observed"]
            and isinstance(msg["fn"], MaskedDistribution)
            and isinstance(msg["fn"].base_dist, Delta)
            and msg["infer"].get("is_deteministic", False)
        )

    def configure(self, msg: dict) -> Optional[Reparam]:
        if not self._is_deterministic(msg):
            return None

        return KernelABCReparam(
            kernel=pyro.contrib.gp.kernels.RBF(input_dim=msg["fn"].event_dim)
        )
=======
>>>>>>> 7e08979 (added KernelABCReparam and simple test)
