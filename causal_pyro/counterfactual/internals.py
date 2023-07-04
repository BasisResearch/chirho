import pyro


class EventDimMessenger(pyro.poutine.messenger.Messenger):
    reinterpreted_batch_ndim: int

    def __init__(self, reinterpreted_batch_ndim: int = 0):
        self.reinterpreted_batch_ndim = reinterpreted_batch_ndim
        super().__init__()

    def _pyro_sample(self, msg: dict) -> None:
        msg["fn"] = msg["fn"].to_event(self.reinterpreted_batch_ndim)
