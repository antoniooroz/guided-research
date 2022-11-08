from typing import List
import torch

from pgnn.base import MCD_Base
from pgnn.configuration.configuration import Configuration
from .ppnp import PPNP

class MCD_PPNP(MCD_Base):  
    def __init__(self, nfeatures: int, nclasses: int, configuration: Configuration, adj_matrix: torch.Tensor):
        super().__init__(nfeatures=nfeatures, nclasses=nclasses, configuration=configuration)

        self.model = PPNP(
            nfeatures=nfeatures, 
            nclasses=nclasses,
            configuration=configuration,
            adj_matrix=adj_matrix
        )