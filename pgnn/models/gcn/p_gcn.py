from typing import List

from pgnn.configuration.configuration import Configuration

from .gcn import GCN
import torch

from pgnn.base import P_Base

class P_GCN(P_Base):
    def __init__(self, nfeatures: int, nclasses: int, configuration: Configuration, adj_matrix: torch.Tensor):
        super().__init__()
        self.configuration = configuration
        self.nclasses = nclasses
        self.nfeatures = nfeatures
        self.adj_matrix = adj_matrix
        
        self.set_model(GCN(
            nfeatures=nfeatures,
            nclasses=nclasses,
            config=configuration,
            adj_matrix=adj_matrix
            )
        )

    