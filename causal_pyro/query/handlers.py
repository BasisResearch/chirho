import pyro


class MonteCarloIntegration(pyro.poutine.messenger.Messenger):
    def __init__(self, sample_size: int, parallel: bool = False):
        super().__init__()
        self.sample_size = sample_size
        self.parallel = parallel

    def _pyro_expectation(self, msg):
        model, name, model_args, model_kwargs = msg["args"]

        pred = pyro.infer.Predictive(model, self.sample_size, self.parallel)
        sim = pred(*model_args, **model_kwargs)
        # TODO: make sure that you're taking the mean over the right axis in the case of multivariate distributions.
        msg["value"] = sim[name].mean()

        # TODO: To avoid the `NotImplementedError` use `msg["stop"] = True` or `msg["done"] = True`. Not sure which one...
        msg["stop"] = True  # don't run the defaults
