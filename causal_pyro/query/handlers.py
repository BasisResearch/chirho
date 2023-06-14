import pyro


class MonteCarloIntegration(pyro.poutine.messenger.Messenger):
    def __init__(self, sample_size: int, parallel: bool = False):
        super().__init__()
        self.sample_size = sample_size
        self.parallel = parallel

    def _pyro_expectation(self, msg):

        model, name, axis, model_args, model_kwargs = msg["args"]

        pred = pyro.infer.Predictive(model, num_samples=self.sample_size, parallel=self.parallel)
        sim = pred(*model_args, **model_kwargs)
        msg["value"] = sim[name].mean(dim=axis)

        # TODO: make sure that you're taking the mean over the right axis in the case of multivariate distributions.
        # @SAM: Done, check, I modified the ops file adding axis as an argument.

        # TODO: To avoid the `NotImplementedError` use `msg["stop"] = True` or `msg["done"] = True`. Not sure which one...
        # @SAM: seems like both work, I used msg["stop"].
        msg["done"] = True  # don't run the defaults
