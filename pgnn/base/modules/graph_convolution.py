import math
import numpy as np

from pgnn.configuration.configuration import Configuration
from .base_module import BaseModule
from pgnn.logger import LogWeight, LogWeightValue
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch_geometric.utils as tu
import wandb
from pyro.nn import PyroModule, PyroParam, PyroSample
import pyro.distributions as dist

from pgnn.inits import glorot, zeros

class GraphConvolution(BaseModule):
    ##############################################################
    # This class is a modified version from the following source
    # Author: Thomas N. Kipf and Max Welling
    # Last Visited: 14.06.2022
    # Title: Graph Convolutional Networks
    # URL: https://github.com/tkipf/gcn
    ##############################################################
    def __init__(self, input_dim, output_dim, configuration: Configuration, bias=False, activation=nn.ReLU(), dropout=0, name=""):
        super().__init__(input_dim=input_dim, output_dim=output_dim, configuration=configuration, name=name)

        self.weight = nn.Parameter(glorot([input_dim, output_dim]))
        
        if bias:
            self.bias = nn.Parameter(zeros([output_dim]))
        else:
            self.bias = None

        self.activation = activation
        self.dropout = nn.Dropout(p=dropout) if dropout != 0 else lambda x: x
    
    def forward(self, input):
        x, support = input
        
        x = self.dropout(x)
        x = x @ self.weight

        # Aggregation
        y = support @ x 

        # Bias
        if self.bias is not None:
            y += self.bias

        return (self.activation(y), support)

    def extra_repr(self):
        return 'input_dim={}, output_dim={}, bias_enabled={}'.format(
                self.input_dim, self.output_dim, self.bias is not None)