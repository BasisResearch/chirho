import pyro


class MonteCarloIntegration(pyro.poutine.messenger.Messenger):
    def __init__(self, sample_size: int, parallel: bool = False):
        super().__init__()
        self.sample_size = sample_size
        self.parallel = parallel

    def _pyro_expectation(self, msg):
        model, name, axis, model_args, model_kwargs = msg["args"]

        pred = pyro.infer.Predictive(
            model, num_samples=self.sample_size, parallel=self.parallel
        )
        sim = pred(*model_args, **model_kwargs)
        msg["value"] = sim[name].mean(dim=axis)

        msg["done"] = True  # mark as computed
