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
import torch.nn as nn

from pgnn.utils import matrix_to_torch, get_edge_indices
from pgnn.inits import zeros
from pgnn.utils.utils import get_device


def calc_A_hat(adj_matrix: torch.Tensor) -> torch.Tensor:
    nnodes = adj_matrix.shape[0]
    A = adj_matrix + torch.eye(nnodes, device=adj_matrix.device)
    D_vec = A.sum(dim=1).flatten()
    D_vec_invsqrt_corr = 1 / D_vec.sqrt()
    D_invsqrt_corr = D_vec_invsqrt_corr.diag()
    return D_invsqrt_corr @ A @ D_invsqrt_corr


def calc_ppr_exact(adj_matrix: sp.spmatrix, alpha: float) -> np.ndarray:
    nnodes = adj_matrix.shape[0]
    M = calc_A_hat(adj_matrix)
    A_inner = sp.eye(nnodes) - (1 - alpha) * M
    return alpha * np.linalg.inv(A_inner.toarray())


class PPRExact(nn.Module):
    def __init__(self, adj_matrix: sp.spmatrix, alpha: float, drop_prob: float = None):
        super().__init__()

        ppr_mat = calc_ppr_exact(adj_matrix, alpha)
        self.register_buffer('mat', torch.FloatTensor(ppr_mat))

        if drop_prob is None or drop_prob == 0:
            self.dropout = lambda x: x
        else:
            self.dropout = nn.Dropout(drop_prob)

    def forward(self, predictions: torch.FloatTensor, idx: torch.LongTensor):
        return self.dropout(self.mat[idx]) @ predictions


class PPRPowerIteration(nn.Module):
    def __init__(self, adj_matrix: torch.Tensor, alpha: float, niter: int, drop_prob: float = None):
        super().__init__()
        self.alpha = alpha
        self.niter = niter

        M = calc_A_hat(adj_matrix)
        self.register_buffer('A_hat', (1 - alpha) * M)

        if drop_prob is None or drop_prob == 0:
            self.dropout = lambda x: x
        else:
            self.dropout = nn.Dropout(drop_prob)

    def forward(self, local_preds: torch.FloatTensor, idx: torch.LongTensor) -> torch.Tensor:
        preds = local_preds
        for _ in range(self.niter):
            A_drop = self.dropout(self.A_hat)
            preds = A_drop @ preds + self.alpha * local_preds

        return preds.index_select(dim=-2, index=idx)

"""
class PPRPowerIterationAlternative(nn.Module):
    def __init__(self, adj_matrix: sp.spmatrix, alpha: float, niter: int, drop_prob: float = None):
        super().__init__()
        self.alpha = alpha
        self.niter = niter

        M = calc_A_hat(adj_matrix)
        self.register_buffer('A_hat', matrix_to_torch((1 - alpha) * M))
        
        A = torch.eye(self.A_hat.shape[0])
        for _ in range(self.niter-1):
            A = (self.A_hat @ A) + (self.alpha)*(torch.eye(self.A_hat.shape[0]))
        self.A_hat = A

        if drop_prob is None or drop_prob == 0:
            self.dropout = lambda x: x
        else:
            self.dropout = nn.Dropout(drop_prob)

    def forward(self, local_preds: torch.FloatTensor, input_preds: torch.FloatTensor, idx: torch.LongTensor):
        preds = local_preds
        A_drop = self.dropout(self.A_hat)
        preds = (A_drop @ preds)

        return preds.index_select(dim=-2, index=idx)
    
class AttentionPropagation(nn.Module):
    def __init__(self, adj_matrix: sp.spmatrix, alpha: float, niter: int, feature_size: int):
        super().__init__()
        self.alpha = alpha
        self.niter = niter

        self.softmax = nn.Softmax(dim=-1)

        M = calc_A_hat(adj_matrix)
        self.register_buffer('A_hat', matrix_to_torch((1 - alpha) * M))
        self.a = nn.Parameter(zeros([feature_size*2]))
        
        self.edge_indices = get_edge_indices(self.A_hat)[0]
        self.cat_indices = torch.cat([self.edge_indices[0].unsqueeze(0), self.edge_indices[1].unsqueeze(0)],dim=0).to(self.a.device)
        
    def forward(self, local_preds: torch.FloatTensor, input_preds: torch.FloatTensor, idx: torch.LongTensor):
        preds = local_preds
        
        compare_features = input_preds
        source_features = compare_features.index_select(dim=-2, index=self.edge_indices[1])
        target_features = compare_features.index_select(dim=-2, index=self.edge_indices[0])
        
        concat_features = torch.cat([source_features, target_features], dim=-1).to(target_features.device)
        
        edge_scores = concat_features @ self.a
        
        edge_scores = self.softmax(edge_scores)
        edge_scores = torch.sparse_coo_tensor(self.cat_indices, edge_scores, torch.Size([preds.shape[0], preds.shape[0]])).to_dense().to(self.a.device)
        
        alpha = edge_scores / edge_scores.sum(dim=-1)
        local_influence = torch.diagonal(alpha).unsqueeze(1)
        for _ in range(self.niter):
            preds = (1-local_influence) * (alpha @ preds) + local_influence * local_preds

        return preds.index_select(dim=-2, index=idx)
    
class AttentionPropagation2(nn.Module):
    def __init__(self, adj_matrix: sp.spmatrix, alpha: float, niter: int, feature_size: int):
        super().__init__()
        self.alpha = alpha
        self.niter = niter

        self.softmax = nn.Softmax(dim=-1)

        M = calc_A_hat(adj_matrix)
        self.register_buffer('A_hat', matrix_to_torch((1 - alpha) * M))
        self.a = nn.Sequential(
            nn.Linear(feature_size, 64),
            nn.Softmax(dim=-1)
        ) 
        
        
        self.edge_indices = get_edge_indices(self.A_hat)[0]
        self.cat_indices = torch.cat([self.edge_indices[0].unsqueeze(0), self.edge_indices[1].unsqueeze(0)],dim=0).to(get_device())
        
    def forward(self, local_preds: torch.FloatTensor, input_preds: torch.FloatTensor, idx: torch.LongTensor):
        preds = local_preds
        
        compare_features = self.a(input_preds)
        source_features = compare_features.index_select(dim=-2, index=self.edge_indices[1])
        target_features = compare_features.index_select(dim=-2, index=self.edge_indices[0])
        
        edge_scores = (source_features-target_features).sum(dim=-1)
        
        edge_scores = self.softmax(edge_scores)
        edge_scores = torch.sparse_coo_tensor(self.cat_indices, edge_scores, torch.Size([preds.shape[0], preds.shape[0]])).to_dense().to(get_device())
        
        alpha = edge_scores / edge_scores.sum(dim=-1)
        local_influence = torch.diagonal(alpha).unsqueeze(1)
        for _ in range(self.niter):
            preds = (1-local_influence) * (alpha @ preds) + local_influence * local_preds

        return preds.index_select(dim=-2, index=idx)
"""