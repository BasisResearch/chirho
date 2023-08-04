from typing import Callable
from torch import Tensor
from torch.nn import Parameter
from collections import OrderedDict

KWType = OrderedDict[str, Tensor]
KWTypeNNParams = OrderedDict[str, Parameter]
ModelType = Callable[[], KWType]

StochasticFunction = Callable[[KWType], Tensor]
ExpectationFunction = Callable[[ModelType], Tensor]
