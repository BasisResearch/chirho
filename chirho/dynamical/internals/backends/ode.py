import pyro


class ODERuntimeCheck(pyro.poutine.messenger.Messenger):
    def _pyro_sample(self, msg):
        raise ValueError(
            "self.__name__ only supports ODE models, and thus does not allow `pyro.sample` calls."
        )
