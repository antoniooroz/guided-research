
from math import sqrt
from pgnn.configuration.experiment_configuration import Dataset, ExperimentMode, ExperimentConfiguration
from pgnn.data.io import load_dataset
from pgnn.data.sparsegraph import SparseGraph
from pgnn.preprocessing import gen_splits
import networkx as nx
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
            
            self.graph.standardize(select_lcc=True)
            self.graph.normalize_features(experiment_configuration=self.experiment_configuration)
            
            self.plot_tsne(seed, self.graph.attr_matrix, self.graph.labels)
            
            return self.graph
        else:
            return self.graph
    
    def sbm_graph(self, seed):
        # Create graph
        networkx_graph = nx.stochastic_block_model(
            self.experiment_configuration.sbm_classes,
            self.get_connection_probabilities(),
            nodelist=None,
            seed=seed,
            directed=False,
            selfloops=False,
            sparse=True
        )
        
        """https://networkx.org/documentation/stable/auto_examples/drawing/plot_labels_and_colors.html"""
        options = {"edgecolors": "tab:gray", "node_size": 25, "alpha": 1}
        
        plt.figure(figsize=(12, 10))
        spring_k =4/sqrt(sum(self.experiment_configuration.sbm_classes)) # default k is 1/sqrt
        pos = nx.spring_layout(networkx_graph, seed=seed, scale=1, k=spring_k)
        
        i = 0
        colors = ["green", "blue", "cyan", "red"]
        for sbm_class, color in zip(self.experiment_configuration.sbm_classes, colors):
            nx.draw_networkx_nodes(networkx_graph, pos, nodelist=range(i, i+sbm_class), node_color=f"tab:{color}", **options)
            i+=sbm_class
        
        
        nx.draw_networkx_edges(networkx_graph, pos, width=0.5, alpha=0.5)
        
        plt.tight_layout()
        plt.legend()
        plt.savefig(f'{os.getcwd()}/plots/graphs/{seed}.png')
        plt.clf()
        
        n_nodes = sum(self.experiment_configuration.sbm_classes)
        
        adj_matrix=sp.sparse.csr_matrix(
            ([1]*len(networkx_graph.edges), (list(map(lambda x: x[0], networkx_graph.edges)), list(map(lambda x: x[1], networkx_graph.edges)))), shape=(n_nodes, n_nodes)
        )
        
        return adj_matrix
    
    def get_connection_probabilities(self):
        # TODO: For SBM also allow frac ood, etc.
        
        N = len(self.experiment_configuration.sbm_classes)
        OOD_N = self.experiment_configuration.ood_loc_num_classes
        
        P_ID_IN = self.experiment_configuration.sbm_connection_probabilities_id_in_cluster
        P_ID_OUT = self.experiment_configuration.sbm_connection_probabilities_id_out_cluster
        P_OOD_IN = self.experiment_configuration.sbm_connection_probabilities_ood_in_cluster
        P_OOD_OUT = self.experiment_configuration.sbm_connection_probabilities_ood_out_cluster
        
        # ID
        connection_probabilities = (np.ones([N, N]) - np.eye(N)) * P_ID_OUT
        connection_probabilities += np.diag(np.ones([N]) * P_ID_IN)
        
        connection_probabilities[N-OOD_N:, :] = P_OOD_OUT
        connection_probabilities[:, N-OOD_N:] = P_OOD_OUT
        
        for i in range(N-OOD_N, N):
            connection_probabilities[i,i] = P_OOD_IN
        
        return connection_probabilities
        
    def sbm_features(self, seed):
        features = []
        labels = []
        
        random_state = seed
        
        means = []
        
        for c, nsamples in enumerate(self.experiment_configuration.sbm_classes):
            mean = sp.stats.norm.rvs(loc=self.experiment_configuration.sbm_feature_mean, scale=self.experiment_configuration.sbm_feature_variance ,size=self.experiment_configuration.sbm_nfeatures, random_state=random_state)
            means.append(mean)
            
            samples = sp.stats.multivariate_normal(
                mean=mean,
                cov=np.diag(np.abs(sp.stats.norm.rvs(loc=self.experiment_configuration.sbm_feature_sampling_variance, scale=1, size=self.experiment_configuration.sbm_nfeatures, random_state=random_state+1))),
                seed=seed
            ).rvs(size=nsamples)
            
            random_state += 10
            
            features.append(samples)
            labels.extend([c] * nsamples)
            
        features = np.concatenate(features)
        labels = np.asarray(labels)
            
        self.log_feature_distances(means)
            
        return features, labels
    
    def log_feature_distances(self, means):
        print('Feature Mean Distances:')
        for i, mean in enumerate(means[:-1]):
            out = f'{i}: '
            for i_comp, mean_comp in enumerate(means[i+1:]):
                out += f'{i_comp+i+1}  {np.linalg.norm(mean - mean_comp)} | '
            print(out)
            
            
    def plot_tsne(self, seed, features, labels):
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
        plt.savefig(f'{os.getcwd()}/plots/features/{seed}.png')
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
        
        