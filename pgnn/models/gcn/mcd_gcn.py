from pgnn.base import MCD_Base
from pgnn.configuration.configuration import Configuration
from .gcn import GCN
import torch
class MCD_GCN(MCD_Base):  
    def __init__(self, nfeatures: int, nclasses: int, configuration: Configuration, adj_matrix: torch.Tensor):
        super().__init__(nfeatures=nfeatures, nclasses=nclasses, configuration=configuration)

        self.model = GCN(
            nfeatures=nfeatures, 
            nclasses=nclasses,
            configuration=configuration,
            adj_matrix=adj_matrix
        )