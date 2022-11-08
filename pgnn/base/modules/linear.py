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

class Linear(BaseModule):
    def __init__(self, input_dim, output_dim, configuration: Configuration, bias=False, activation=nn.ReLU(), dropout=0, name=""):
        super().__init__(input_dim=input_dim, output_dim=output_dim, configuration=configuration, name=name)

        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
            
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.bias = None          

        self.activation = activation
        self.dropout = nn.Dropout(p=dropout) if dropout != 0 else lambda x: x

        self.reset_parameters()

    def reset_parameters(self):
        # Our fan_in is interpreted by PyTorch as fan_out (swapped dimensions)
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', a=math.sqrt(5))
        if self.bias is not None:
            _, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_out)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x = self.dropout(x)

        y = x @ self.weight

        if self.bias is not None:
            y += self.bias

        return self.activation(y)

    def extra_repr(self):
        return 'input_dim={}, output_dim={}, bias_enabled={}'.format(
                self.input_dim, self.output_dim, self.bias_enabled)