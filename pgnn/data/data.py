from dataclasses import dataclass
from torch import Tensor
from .model_input import ModelInput

@dataclass
class Data:
    model_input: ModelInput = None
    labels: Tensor = None
    ood_indicators: Tensor = None