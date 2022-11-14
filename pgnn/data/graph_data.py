
from pgnn.configuration.experiment_configuration import Dataset, ExperimentMode, ExperimentConfiguration
from pgnn.data.io import load_dataset
from pgnn.data.sparsegraph import SparseGraph
from pgnn.preprocessing import gen_splits
from networkx import stochastic_block_model
import scipy as sp
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import os
import torch


class GraphData:
    def __init__(self, experiment_configuration: ExperimentConfiguration):
        self.experiment_configuration = experiment_configuration
        
        if self.experiment_configuration.dataset != Dataset.GENERATED_SBM:
            self.graph = load_dataset(experiment_configuration.dataset.value)
            self.graph.standardize(select_lcc=True)
            self.graph.normalize_features(experiment_configuration=experiment_configuration)
    
    def get_graph(self, seed) -> SparseGraph:
        if self.experiment_configuration.dataset == Dataset.GENERATED_SBM:
            adj_matrix = self.sbm_graph(seed)
            features, labels = self.sbm_features(seed)
            self.graph = SparseGraph(
                adj_matrix=adj_matrix,
                attr_matrix=features,
                labels=labels
            )
            self.graph.normalize_features(experiment_configuration=self.experiment_configuration)
            
            self.plot_tsne(seed, self.graph.attr_matrix, self.graph.labels)
            
            return self.graph
        else:
            return self.graph
    
    def sbm_graph(self, seed):
        # Create graph
        networkx_graph = stochastic_block_model(
            self.experiment_configuration.sbm_classes,
            self.experiment_configuration.sbm_connection_probabilities,
            nodelist=None,
            seed=seed,
            directed=False,
            selfloops=False,
            sparse=True
        )
        adj_matrix=sp.sparse.csr_matrix(
            ([1]*len(networkx_graph.edges), (list(map(lambda x: x[0], networkx_graph.edges)), list(map(lambda x: x[1], networkx_graph.edges)))),
        )
        
        return adj_matrix
        
    def sbm_features(self, seed):
        features = []
        labels = []
        
        random_state = seed
        
        for c, nsamples in enumerate(self.experiment_configuration.sbm_classes):
            samples = sp.stats.multivariate_normal(
                mean=sp.stats.norm.rvs(size=self.experiment_configuration.sbm_nfeatures, random_state=random_state),
                cov=np.diag(np.abs(sp.stats.norm.rvs(scale=10, size=self.experiment_configuration.sbm_nfeatures, random_state=random_state+1))),
                seed=seed
            ).rvs(size=nsamples)
            
            random_state += 10
            
            features.append(samples)
            labels.extend([c] * nsamples)
            
        features = np.concatenate(features)
        labels = np.asarray(labels)
            
        return features, labels
            
    def plot_tsne(self, seed, features, labels):
        tsne = TSNE(n_components=2, random_state=0)
        colors = 'r', 'g', 'b', 'c' #, 'm', 'y', 'k', 'w', 'orange', 'purple'
        
        features_2d = tsne.fit_transform(features.cpu())
        
        possible_labels = list(range(np.max(labels) + 1))
        
        plt.figure(figsize=(6, 5))
        for label, color in zip(possible_labels, colors):
            plt.scatter(features_2d[labels == label, 0], features_2d[labels == label, 1], c=color, label=str(label))
        
        plt.legend()
        plt.savefig(f'{os.getcwd()}/plots/features/{seed}.png')

        
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
        
        