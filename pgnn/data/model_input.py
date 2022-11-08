from torch import Tensor
from dataclasses import dataclass


@dataclass
class ModelInput:
    features: Tensor = None
    indices: Tensor = None
