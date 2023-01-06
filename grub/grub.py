
from math import sqrt
from pgnn.base.network_mode import NetworkMode
from pgnn.configuration.experiment_configuration import ActiveLearningSelector
from pgnn.configuration.model_configuration import UncertaintyMode
from pgnn.configuration.training_configuration import Phase

from grub.utils.io import load_dataset
from grub.utils.sparsegraph import SparseGraph
from grub.ood.ood import OOD_Experiment
from grub.utils.preprocessing import gen_splits
import networkx as nx
import scipy as sp
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import os
from torch.utils.data import TensorDataset, DataLoader
import torch

from pgnn.configuration import Dataset, ExperimentMode, ExperimentConfiguration, Configuration, OOD
from pgnn.result.result import Results

from pgnn.utils.utils import get_device, matrix_to_torch


class GRUB:
    def __init__(self, configuration):
        self.configuration = configuration
        self.experiment_configuration = configuration.experiment
        
        if self.experiment_configuration.dataset != Dataset.GENERATED_SBM:
            self.graph = load_dataset(self.experiment_configuration.dataset.value)
            self.graph.standardize(select_lcc=True)
            self.graph.normalize_features(experiment_configuration=self.experiment_configuration)
            
        self.nfeatures = None
        self.nclasses = None
        self.feature_matrix = None
        self.adjacency_matrix = None
        self.idx_all = None
        self.labels_all = None
        self.oods_all = None
        self.dataloaders = None
        
    def init(self, seed) -> tuple[torch.Tensor, DataLoader]:
        device = get_device()
        
        self.init_graph(seed)
        self.idx_all = self.get_split(seed)
        
        self.labels_all = torch.LongTensor(self.graph.labels.astype('int64')).to(device)
        self.oods_all = torch.zeros(self.labels_all.shape).to(device)
        self.feature_matrix = self.graph.attr_matrix.clone().detach().to(device)
        self.adjacency_matrix = matrix_to_torch(self.graph.adj_matrix)
        
        # Active Learning
        if self.experiment_configuration.active_learning:
            ActiveLearning.set_starting_class(self)
        
        # OOD
        if self.experiment_configuration.ood != OOD.NONE:
            self.adjacency_matrix, self.feature_matrix, self.idx_all, self.labels_all, self.oods_all = OOD_Experiment.setup(
                configuration=self.configuration, 
                adjacency_matrix=self.adjacency_matrix, 
                feature_matrix=self.feature_matrix, 
                idx_all=self.idx_all, 
                labels_all=self.labels_all,
                oods_all=self.oods_all
            )
        
        self.init_dataloaders()
        
        self.nfeatures = self.feature_matrix.shape[1]
        # OOD-Setting: Remove left-out classes
        if self.experiment_configuration.ood_loc_remove_classes:
            self.nclasses = torch.max(self.labels_all[self.oods_all==0]).cpu().item() + 1
        else:
            self.nclasses = torch.max(self.labels_all).cpu().item() + 1
    
    def init_graph(self, seed):
        if self.experiment_configuration.dataset == Dataset.GENERATED_SBM:
            self.graph = SBM.init(self, seed)
        
        #self.plot_tsne(seed, self.graph.attr_matrix, self.graph.labels)
    
    def log_feature_distances(self, means):
        print('Feature Mean Distances:')
        for i, mean in enumerate(means[:-1]):
            out = f'{i}: '
            for i_comp, mean_comp in enumerate(means[i+1:]):
                out += f'{i_comp+i+1}  {np.linalg.norm(mean - mean_comp)} | '
            print(out)
            
    def plot_tsne(self, name, features, labels):
        """https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
        https://scipy-lectures.org/packages/scikit-learn/auto_examples/plot_tsne.html
        """
        tsne = TSNE(n_components=2, random_state=0)
        colors = 'g', 'b', 'c', 'r' #, 'm', 'y', 'k', 'w', 'orange', 'purple'
        
        features_2d = tsne.fit_transform(features.cpu())
        
        possible_labels = list(range(np.max(labels) + 1))
        
        plt.figure(figsize=(6, 5))
        for label, color in zip(possible_labels, colors):
            plt.scatter(features_2d[labels == label, 0], features_2d[labels == label, 1], c=color, label=str(label))
        
        plt.legend()
        plt.savefig(f'{os.getcwd()}/plots/features_in/{name}.png')
        plt.clf()
        
    def get_split(self, seed):
        idx_all = gen_splits(
            labels=self.graph.labels.astype('int64'), 
            idx_split_args={
                'ntrain_per_class': self.experiment_configuration.datapoints_training_per_class,
                'nstopping': self.experiment_configuration.datapoints_stopping,
                'nknown': self.experiment_configuration.datapoints_known,
                'seed': seed
            }, 
            test=self.experiment_configuration.seeds.experiment_mode==ExperimentMode.TEST
        )
        return idx_all
        
    def init_dataloaders(self, batch_size=None):
        device = get_device()
        
        if batch_size is None:
            batch_size = max((val.numel() for val in self.idx_all.values()))
        datasets = {phase: TensorDataset(ind.to(device), self.labels_all[ind].to(device), self.oods_all[ind].to(device)) for phase, ind in self.idx_all.items()}
        self.dataloaders = {phase: DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
                    for phase, dataset in datasets.items()}
    

    
