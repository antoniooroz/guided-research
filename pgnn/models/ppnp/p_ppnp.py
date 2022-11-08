from typing import List
import torch

from pgnn.configuration.configuration import Configuration
from .ppnp import PPNP
from pgnn.base import P_Base

class P_PPNP(P_Base):
    def __init__(self, nfeatures: int, nclasses: int, configuration: Configuration, adj_matrix: torch.Tensor):
        super().__init__()
        self.configuration = configuration
        self.nclasses = nclasses
        self.nfeatures = nfeatures
        self.adj_matrix = adj_matrix
        
        self.set_model(PPNP(
            nfeatures=nfeatures,
            nclasses=nclasses,
            configuration=configuration,
            adj_matrix=adj_matrix
            )
        )