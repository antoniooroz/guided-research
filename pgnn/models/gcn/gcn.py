##############################################################
# This file is a modified version from the following source
# Author: Thomas N. Kipf and Max Welling
# Last Visited: 14.06.2022
# Title: Graph Convolutional Networks
# URL: https://github.com/tkipf/gcn
##############################################################

from pyexpat import model
from typing import List
import torch
import torch.nn as nn
from pgnn.base.network_mode import NetworkMode
from pgnn.configuration.configuration import Configuration
from pgnn.configuration.model_configuration import ModelType
from pgnn.data import model_input
from pgnn.data.model_input import ModelInput
from pgnn.result.model_output import ModelOutput

from pgnn.utils import edge_dropout, get_device, preprocess_adj
from pgnn.base import GraphConvolution, Base

class GCN(Base):
    def __init__(self, nfeatures: int, nclasses: int, configuration: Configuration, adj_matrix: torch.Tensor):
        super().__init__()
        self.configuration = configuration
        self.nclasses = nclasses
        self.nfeatures = nfeatures

        self.original_adj_matrix = adj_matrix
        
        self.preprocessed_adj_matrix = preprocess_adj(adj_matrix, laplacian=True).type(torch.float32).to(get_device())
        
        self.isolated_adj_matrix = torch.eye(self.original_adj_matrix.shape[0]).type(torch.float32).to(get_device())

        hiddenunits = [nfeatures] + self.configuration.model.hidden_layer_size + [nclasses]

        layers = []
        for i in range(len(hiddenunits)-1):            
            layers.append(GraphConvolution(
                input_dim=hiddenunits[i], 
                output_dim=hiddenunits[i+1],
                configuration=configuration,  
                bias=self.configuration.model.bias,
                activation=nn.ReLU() if i < len(hiddenunits)-2 else lambda x: x,
                dropout=self.configuration.training.drop_prob,
                name="layer"+str(i)
            ))

        self.layers = nn.Sequential(*layers)

    def forward(self, model_input: ModelInput) -> dict[NetworkMode, ModelOutput]:
        isolated_input = (model_input.features, self.isolated_adj_matrix)
        propagated_input = (model_input.features, self.get_adj_matrix())

        isolated_logits = self.layers(isolated_input)[0].index_select(
            dim=-2, index=model_input.indices)
        propagated_logits = self.layers(propagated_input)[0].index_select(
            dim=-2, index=model_input.indices)

        return {
            NetworkMode.ISOLATED: ModelOutput(
                logits=isolated_logits,
                softmax_scores=isolated_logits.softmax(-1)
            ),
            NetworkMode.PROPAGATED: ModelOutput(
                logits=propagated_logits,
                softmax_scores=propagated_logits.softmax(-1)
            )
        }

    def get_adj_matrix(self):
        if self.configuration.training.edge_drop_prob > 0 and self.configuration.model.type == ModelType.DE_GCN:
            adj_dropped = edge_dropout(self.original_adj_matrix, self.configuration.training.edge_drop_prob)
            return preprocess_adj(adj_dropped, laplacian=True).type(torch.float32).to(get_device())
        else:
            return self.preprocessed_adj_matrix