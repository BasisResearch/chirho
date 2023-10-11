from typing import Dict

import torch

from chirho.dynamical.handlers.trace import DynamicTrace
from chirho.observational.handlers import condition


class NonInterruptingPointObservationArray(DynamicTrace):
    def __init__(
        self,
        times: torch.Tensor,
        data: Dict[str, torch.Tensor],
        eps: float = 1e-6,
    ):
        self.data = data
        # Add a small amount of time to the observation time to ensure that
        # the observation occurs after the logging period.
        self.times = times + eps

        # Require that each data element maps 1:1 with the times.
        if not all(len(v) == len(times) for v in data.values()):
            raise ValueError(
                f"Each data element must have the same length as the passed times. Got lengths "
                f"{[len(v) for v in data.values()]} for data elements {[k for k in data.keys()]}, but "
                f"expected length {len(times)}."
            )

        super().__init__(times)

    def _pyro_post_simulate(self, msg) -> None:
        dynamics, _, _, _ = msg["args"]

        if "in_SEL" not in msg.keys():
            msg["in_SEL"] = False

        # This checks whether the simulate has already redirected in a SimulatorEventLoop.
        # If so, we don't want to run the observation again.
        if msg["in_SEL"]:
            return

        # TODO: Check to make sure that the observations all fall within the outermost `simulate` start and end times.
        super()._pyro_post_simulate(msg)
        # This condition checks whether all of the simulate calls have been executed.
        if len(self.trace) == len(self.times):
            with condition(data=self.data):
                dynamics.observation(self.trace)

            # Reset the trace for the next simulate call.
            super()._reset()
