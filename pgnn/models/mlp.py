##############################################################
# This file is a modified version from the following source
# Author: Johannes Gasteiger, Aleksandar Bojchevski and Stephan Günnemann
# Last Visited: 14.06.2022
# Title: PPNP and APPNP
# URL: https://github.com/gasteigerjo/ppnp
##############################################################

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from pgnn.base import Linear, Base

import copy
import os
from pgnn.base.network_mode import NetworkMode

from pgnn.configuration.configuration import Configuration
from pgnn.base.propagation import PPRPowerIteration
from pgnn.data.model_input import ModelInput
from pgnn.result.model_output import ModelOutput

class MLP(Base):    
    def __init__(self, nfeatures: int, nclasses: int, configuration: Configuration, adj_matrix: torch.Tensor, **kwargs):
        super().__init__()
        self.configuration = configuration
        self.nclasses = nclasses
        self.nfeatures = nfeatures

        hiddenunits = [nfeatures] + self.configuration.model.hidden_layer_size + [nclasses]
        
        layers = []
        for i in range(len(hiddenunits)-1):
            layers.append(Linear(
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
        all_isolated_logits: torch.Tensor = self.layers(model_input.features)
        
        isolated_logits = all_isolated_logits.index_select(dim=-2, index=model_input.indices)
        
        return {
            NetworkMode.ISOLATED: ModelOutput(
                logits=isolated_logits,
                softmax_scores=isolated_logits.softmax(-1)
            ),
            NetworkMode.PROPAGATED: ModelOutput(
                logits=isolated_logits,
                softmax_scores=isolated_logits.softmax(-1)
            )
        }