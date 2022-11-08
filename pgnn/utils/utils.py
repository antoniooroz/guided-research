##############################################################
# This file is a modified version from the following source
# Author: Johannes Gasteiger, Aleksandar Bojchevski and Stephan Günnemann
# Last Visited: 14.06.2022
# Title: PPNP and APPNP
# URL: https://github.com/gasteigerjo/ppnp
##############################################################

import numpy as np
import scipy.sparse as sp
import torch
import torch_geometric.utils as tu
import wandb
import os
from pgnn.configuration.training_configuration import Phase
from pgnn.data.data import Data
from pgnn.data.model_input import ModelInput

from pgnn.result.result import Results

def preprocess_adj(adj: torch.Tensor, laplacian=True, self_loop=True) -> torch.Tensor:
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    if self_loop:
        adj = adj + torch.eye(adj.shape[0], device=adj.device)

    if laplacian:
        adj_normalized = normalize_adj(adj)
    else:
        adj_normalized = adj / adj.sum(dim=1)

    return adj_normalized

def normalize_adj(adj: torch.Tensor) -> torch.Tensor:
    """Symmetrically normalize adjacency matrix."""
    rowsum = adj.sum(dim=1)
    d_inv_sqrt = rowsum.pow(-0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt, device=adj.device)
    return (adj @ d_mat_inv_sqrt).transpose() @ d_mat_inv_sqrt

def sparse_matrix_to_torch(X):
    coo = X.tocoo()
    indices = np.array([coo.row, coo.col])
    return torch.sparse.FloatTensor(
            torch.LongTensor(indices),
            torch.FloatTensor(coo.data),
            coo.shape)

def matrix_to_torch(X):
    if sp.issparse(X):
        return torch.FloatTensor(X.todense()).to(get_device())
    else:
        return torch.FloatTensor(X).to(get_device())

def edge_dropout(adj, p):
    adj = adj.clone().detach()
    dropout = torch.zeros(adj.shape)
    dropout = dropout.bernoulli(p=p)
    adj[dropout==1] = 0
    return adj

def get_statistics(model, attr_mat_norm, idx, labels, oods, use_cached = False):
    probs, preds, epist, alea = model.get_predictions(attr_mat_norm, idx)
                    
    probs_max = torch.max(probs, dim=1)

    id_epist = epist[oods==0]
    id_preds = preds[oods==0]
    id_labels = labels[oods==0]
    
    # Confidence for all data points
    confidence_all = id_epist.sum()
    
    # Confidences for only correctly/ wrongly predicted data points
    confidence_correct = id_epist[id_preds==id_labels].sum()
    confidence_false = id_epist[id_preds!=id_labels].sum()

    # Number correctly/wrongly predicted data points
    datapoints_correct = (id_preds==id_labels).sum()
    datapoints_false = (id_preds!=id_labels).sum()
    
    return probs, preds, confidence_all, confidence_correct, confidence_false, datapoints_correct, datapoints_false

def final_run(model, attr_matrix, idx_dict, labels, oods) -> dict[Phase, Results]:
    resultsPerPhase = {}
    for phase, idx in idx_dict.items():
        idx = idx.to(attr_matrix.device)
        data = Data(
            model_input=ModelInput(features=attr_matrix, indices=idx),
            labels=labels.to(attr_matrix.device)[idx],
            ood_indicators=oods.to(attr_matrix.device)[idx]
        )
        
        results: Results = model.step(Phase.VALTEST, data)
        resultsPerPhase[phase] = results

    return resultsPerPhase

def config_from_dict(dictionary):
    config = wandb.config
    for key in dictionary.keys():
        config[key] = dictionary[key]
        
    return config

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def get_edge_indices(adj):      
    # Adj normally expected to be csr_tensor
    if not torch.is_tensor(adj):  
        adj = torch.tensor(adj.todense())
    edge_indices, edge_weights = tu.dense_to_sparse(adj.cpu())
        
    return edge_indices.to(get_device()), edge_weights.to(get_device())

def get_self_edge_indices(adj_csr_matrix):
    adj = torch.eye(adj_csr_matrix.shape[0])
    edge_indices, edge_weights = tu.dense_to_sparse(adj)
        
    return edge_indices.to(get_device())
    
def escape_slashes(val):
    if os.name == 'nt':
        return val.replace('/','\\')
    else:
        return  val
    
