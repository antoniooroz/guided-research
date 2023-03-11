from pgnn.base import ENSEMBLE_Base
from pgnn.configuration.configuration import Configuration
from .gcn import GCN
import torch
class ENSEMBLE_GCN(ENSEMBLE_Base):  
    def __init__(self, nfeatures: int, nclasses: int, configuration: Configuration, adj_matrix: torch.Tensor, **kwargs):
        super().__init__(nfeatures=nfeatures, nclasses=nclasses, configuration=configuration)

        for _ in range(configuration.model.samples_training):
            self.models.append(GCN(
                nfeatures=nfeatures, 
                nclasses=nclasses,
                configuration=configuration,
                adj_matrix=adj_matrix
            ))