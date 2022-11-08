##############################################################
# This file is a modified version from the following source
# Author: Maximilian Stadler, Bertrand Charpentier, Simon Geisler, Daniel Zügner and Stephan Günnemann
# Last Visited: 14.06.2022
# Title: Graph Posterior Network
# URL: https://github.com/stadlmax/Graph-Posterior-Network
##############################################################

import math
import torch
import numpy as np
from scipy import sparse
from pgnn.configuration.configuration import Configuration
from pgnn.configuration.experiment_configuration import OOD, OODAttributeNormalization, OODPerturbMode
from pgnn.configuration.training_configuration import Phase

class OOD_Experiment(object):
    def __init__(self, 
                 configuration: Configuration, 
                 adjacency_matrix: torch.Tensor, 
                 feature_matrix: torch.Tensor,
                 idx_all: dict[Phase, torch.Tensor],
                 labels_all: torch.Tensor,
                 oods_all: torch.Tensor):
        super().__init__()
        self.configuration = configuration
        self.idx_all = idx_all
        self.labels_all = labels_all
        self.oods_all = oods_all
        self.feature_matrix = feature_matrix
        self.adjacency_matrix = adjacency_matrix
        
    def setup(configuration: Configuration, 
                 adjacency_matrix: torch.Tensor, 
                 feature_matrix: torch.Tensor,
                 idx_all: dict[Phase, torch.Tensor],
                 labels_all: torch.Tensor,
                 oods_all: torch.Tensor):
        experiment = OOD_Experiment(configuration, adjacency_matrix, feature_matrix, idx_all, labels_all, oods_all)
        
        if configuration.experiment.ood == OOD.LOC:
            experiment._loc_setup()
        elif configuration.experiment.ood == OOD.PERTURB:
            experiment._perturb_setup()
        elif configuration.experiment.ood == OOD.MIXED:
            experiment._loc_setup()
            experiment._perturb_setup()
            
        return experiment.adjacency_matrix, experiment.feature_matrix, experiment.idx_all, experiment.labels_all, experiment.oods_all
    
    # LOC Setup
    def _loc_setup(self):       
        ood_classes, id_classes = self._loc_get_loc_classes()

        if self.configuration.experiment.ood_loc_remove_edges:
            self._loc_remove_id_ood_edges(ood_classes, id_classes)

        for c in ood_classes:
            self.oods_all[self.labels_all==c] = 1
            
    def _loc_get_loc_classes(self):
        if self.configuration.experiment.ood_loc_classes is not None:
            raise NotImplementedError('Unused: Check if correct')
            return np.asarray(self.configuration.experiment.ood_loc_classes)
        
        num_classes = self.labels_all.max() + 1
        classes = np.arange(num_classes)
            
        if self.configuration.experiment.ood_loc_num_classes is not None:
            raise NotImplementedError('Unused: Check if correct')
            num_loc_classes = self.configuration.experiment.ood_loc_num_classes
        else:
            num_loc_classes = math.floor(num_classes * self.configuration.experiment.ood_loc_frac)
            
        return classes[num_classes-num_loc_classes : num_classes], classes[:num_classes-num_loc_classes]

    def _loc_remove_id_ood_edges(self, ood_classes, id_classes):
        adjacency_matrix = self.adjacency_matrix.clone().detach()
        for ood_class in ood_classes:
            for id_class in id_classes:
                # Remove OOD->ID edges
                oods_to_ids = adjacency_matrix[self.labels_all==ood_class]
                oods_to_ids[:,self.labels_all==id_class] = 0
                adjacency_matrix[self.labels_all==ood_class] = oods_to_ids
                # Remove ID->OOD edges
                ids_to_oods = adjacency_matrix[self.labels_all==id_class]
                ids_to_oods[:,self.labels_all==ood_class] = 0
                adjacency_matrix[self.labels_all==id_class] = ids_to_oods

        self.adjacency_matrix = adjacency_matrix

    
    # Perturb Setup
    def _perturb_setup(self):
        eps = 1e-10
        
        idx_perturbed = self._perturb_get_indices()
        
        noise = torch.zeros([idx_perturbed.shape[0], self.feature_matrix.shape[1]])

        self.oods_all[idx_perturbed] = 1
        feature_matrix_ood = self.feature_matrix.cpu()

        mode = self.configuration.experiment.ood_perturb_mode

        if mode == OODPerturbMode.NORMAL:
            noise = noise.normal_()
        elif mode == OODPerturbMode.BERNOULLI_AUTO_P:
            p_calc = feature_matrix_ood.clone().detach()
            p_calc[p_calc > 0] = 1
            p = p_calc.mean().item()
            noise = noise.bernoulli(p)
        elif mode == OODPerturbMode.BERNOULLI:
            noise = noise.bernoulli(self.configuration.experiment.ood_perturb_bernoulli_probability)
        elif mode == OODPerturbMode.SHUFFLE:
            raise NotImplementedError('check')
            noise = feature_matrix_ood[idx_perturbed]
            for i in range(noise.shape[0]):
                noise[i] = noise[i][torch.randperm(noise.shape[1])]
        elif mode == OODPerturbMode.ZEROS:
            raise NotImplementedError('check')
        elif mode == OODPerturbMode.ONES:
            raise NotImplementedError('check')
            noise += 1
        else:
            raise NotImplementedError()
        
        # OOD Normalization
        normalization_modes = self.configuration.experiment.ood_normalize_attributes
        
        if OODAttributeNormalization.MIRROR_NEGATIVE_VALUES in normalization_modes:
            noise[noise < 0] = -noise[noise < 0]
        
        if OODAttributeNormalization.DIV_BY_SUM in normalization_modes:
            noise = noise / (noise.sum(dim=-1).unsqueeze(-1) + eps)
        
        # Noise Scalse
        feature_matrix_ood[idx_perturbed] = noise * self.configuration.experiment.ood_perturb_noise_scale

        self.feature_matrix = feature_matrix_ood.to(self.adjacency_matrix.device)
        
    def _perturb_get_indices(self):
        def _internal_get_indices(idx):
            n_perturbed = int(idx.shape[0] * self.configuration.experiment.ood_perturb_budget)
            idx_perturbed = np.random.choice(idx, n_perturbed, replace=False)
            return torch.tensor(idx_perturbed)
        
        idx_perturbed = []
        if self.configuration.experiment.ood_perturb_train:
            raise NotImplementedError('check')
            idx_train_preturbed = _internal_get_indices(self.idx_all["train"])
            #self.idx_all["ood_train"] = idx_train_preturbed
            idx_perturbed.append(idx_train_preturbed)
            
        idx_stopping_perturbed = _internal_get_indices(self.idx_all["stopping"])
        idx_valtest_perturbed = _internal_get_indices(self.idx_all["valtest"])
        
        idx_perturbed.append(idx_stopping_perturbed)
        idx_perturbed.append(idx_valtest_perturbed)
        
        return torch.cat(idx_perturbed, dim=0)
