from pgnn.base import MCD_Base
from pgnn.configuration.configuration import Configuration
from pgnn.models import GAT
import torch

class MCD_GAT(MCD_Base):  
    def __init__(self, nfeatures: int, nclasses: int, configuration: Configuration, adj_matrix: torch.Tensor):
        super().__init__(nfeatures=nfeatures, nclasses=nclasses, configuration=configuration)

        self.model = GAT(
            nfeatures=nfeatures, 
            nclasses=nclasses,
            configuration=configuration,
            adj_matrix=adj_matrix
        )