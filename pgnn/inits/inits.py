##############################################################
# The following functions originate from the following source. 
# Modifications are applied to make it compatible with PyTorch
# Author: Thomas N. Kipf and Max Welling
# Last Visited: 14.06.2022
# Title: Graph Convolutional Networks
# URL: https://github.com/tkipf/gcn
##############################################################

import numpy as np
import torch

from pgnn.utils import get_device

def glorot(shape):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = torch.rand(shape, dtype=torch.float32).to(get_device())
    # Transform from range [0,1] to [-init_range, init_range]
    intial = (initial - 0.5) * 2 * init_range
    return initial


def zeros(shape):
    """All zeros."""
    initial = torch.zeros(shape, dtype=torch.float32).to(get_device())
    return initial


def ones(shape):
    """All ones."""
    initial = torch.ones(shape, dtype=torch.float32).to(get_device())
    return initial
