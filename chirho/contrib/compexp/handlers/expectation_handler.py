import pyro


class ExpectationHandler(pyro.poutine.messenger.Messenger):

    # noinspection PyMethodMayBeStatic
    def _pyro_compute_expectation_atom(self, msg) -> None:
        if msg["done"]:
            # TODO bdt18dosjk Do something similar to the demo1 setup where we define an OOP interface as well.
            #  Then handler can be passed to expectation atoms so that users can specify different handlers
            #  for different atoms.
            raise RuntimeError("Only one default expectation handler can be in effect at a time.")
