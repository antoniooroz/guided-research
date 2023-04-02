##############################################################
# This file is a modified version from the following source
# Author: Johannes Gasteiger, Aleksandar Bojchevski and Stephan Günnemann
# Last Visited: 14.06.2022
# Title: PPNP and APPNP
# URL: https://github.com/gasteigerjo/ppnp
##############################################################

from typing import List, Tuple, Dict
import copy
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from math import isclose

from pgnn.configuration.training_configuration import Phase
from pgnn.utils.utils import get_device


def gen_seeds(size: int = None) -> np.ndarray:
    max_uint32 = np.iinfo(np.uint32).max
    return np.random.randint(
            max_uint32 + 1, size=size, dtype=np.uint32)


def exclude_idx(idx: np.ndarray, idx_exclude_list: List[np.ndarray]) -> np.ndarray:
    idx_exclude = np.concatenate(idx_exclude_list)
    return np.array([i for i in idx if i not in idx_exclude])


def known_unknown_split(
        idx: np.ndarray, nknown: int = 1500, seed: int = 4143496719) -> Tuple[np.ndarray, np.ndarray]:
    rnd_state = np.random.RandomState(seed)
    known_idx = rnd_state.choice(idx, nknown, replace=False)
    unknown_idx = exclude_idx(idx, [known_idx])
    return known_idx, unknown_idx


def train_stopping_split(
        idx: np.ndarray, labels: np.ndarray, ntrain_per_class: int = 20, node_types=None, training_type=None,
        nstopping: int = 500, seed: int = 2413340114, split_ratio=None, n_types_per_class=None) -> Tuple[np.ndarray, np.ndarray]:
    rnd_state = np.random.RandomState(seed)
    
    train_idx_split = []
    
    if training_type is not None and len(training_type) > 0:
        selector = (node_types == training_type[0])
        for v in training_type[1:]:
            selector += (node_types == v) 
            
        _idx = idx[selector]
        _labels = labels[selector]
        
        for i in range(max(labels) + 1):
            train_idx_split.append(rnd_state.choice(
                _idx[_labels == i], ntrain_per_class, replace=False))
    elif node_types is not None:
        for c in range(max(labels) + 1):
            class_idx = idx[labels == c]
            class_types = node_types[labels == c]
            
            for t in range(n_types_per_class):
                ntrain_per_type = int(round(split_ratio[c][t] * ntrain_per_class,0))
                assert isclose(ntrain_per_class * split_ratio[c][t], ntrain_per_type)
    
                train_idx_split.append(rnd_state.choice(
                    class_idx[class_types == t], ntrain_per_type, replace=False))
    else:
        _idx = idx
        _labels = labels
        
        for i in range(max(labels) + 1):
            train_idx_split.append(rnd_state.choice(
                _idx[_labels == i], ntrain_per_class, replace=False))
    
    train_idx = np.concatenate(train_idx_split)
    stopping_idx = rnd_state.choice(
            exclude_idx(idx, [train_idx]),
            nstopping, replace=False)
    return train_idx, stopping_idx


def gen_splits(
        labels: np.ndarray, idx_split_args: Dict[str, int],
        test: bool = False, node_types=None, training_type=None, split_ratio=None, n_types_per_class=None, valtest_type=None, sbm=False) -> Dict[Phase, torch.LongTensor]:
    all_idx = np.arange(len(labels))
    known_idx, unknown_idx = known_unknown_split(
            all_idx, idx_split_args['nknown'], node_types)
    _, cnts = np.unique(labels[known_idx], return_counts=True)
    stopping_split_args = copy.copy(idx_split_args)
    del stopping_split_args['nknown']
    train_idx, stopping_idx = train_stopping_split(
            known_idx,
            labels[known_idx], 
            node_types=node_types[known_idx] if node_types is not None else None, 
            training_type=training_type,
            split_ratio=split_ratio,
            n_types_per_class=n_types_per_class,
            **stopping_split_args)
    
    if test and not sbm:
        val_idx = unknown_idx
    else:
        val_idx = exclude_idx(known_idx, [train_idx, stopping_idx])
    
    idx_all = {
        Phase.TRAINING: torch.LongTensor(train_idx).to(get_device()),
        Phase.STOPPING: torch.LongTensor(stopping_idx).to(get_device()),
        Phase.VALTEST: torch.LongTensor(val_idx).to(get_device())
    } 
    
    if valtest_type is not None:
        selector = np.zeros(val_idx.shape, dtype=np.bool)
        for t in valtest_type:
            selector += (node_types[val_idx] == t)
        val_idx_filtered = val_idx[selector]
        idx_all[Phase.VALTEST_FOR_TYPE] = torch.LongTensor(val_idx_filtered).to(get_device())
        assert np.unique(labels[val_idx_filtered]).shape[0] == np.unique(labels).shape[0]
        
    return idx_all

def normalize_attributes(attr_matrix):
    epsilon = 1e-12
    if isinstance(attr_matrix, sp.csr_matrix):
        attr_norms = spla.norm(attr_matrix, ord=1, axis=1)
        attr_invnorms = 1 / np.maximum(attr_norms, epsilon)
        attr_mat_norm = attr_matrix.multiply(attr_invnorms[:, np.newaxis])
    else:
        attr_norms = np.linalg.norm(attr_matrix, ord=1, axis=1)
        attr_invnorms = 1 / np.maximum(attr_norms, epsilon)
        attr_mat_norm = attr_matrix * attr_invnorms[:, np.newaxis]
    return attr_mat_norm
