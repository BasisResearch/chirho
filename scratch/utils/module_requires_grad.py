import contextlib
from typing import Dict
import torch


@contextlib.contextmanager
def module_requires_grad_(module: torch.nn.Module, mode: bool):
    # 1. save the original requires grad status of the module.
    # 2. set the requires grad status of the module to the desired mode.
    # 3. yield
    # 4. restore the original requires grad status of the module.

    original_requires_grad_param_dict: Dict[str, bool] = dict(
        (param_name, param.requires_grad)
        for param_name, param in module.named_parameters()
    )

    for param in module.parameters():
        param.requires_grad_(mode)

    yield

    for param_name, param in module.named_parameters():
        param.requires_grad_(original_requires_grad_param_dict[param_name])
