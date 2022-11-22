
from math import sqrt
from pgnn.configuration.experiment_configuration import ActiveLearningSelector
from pgnn.configuration.training_configuration import Phase
from pgnn.data.io import load_dataset
from pgnn.data.sparsegraph import SparseGraph
from pgnn.ood.ood import OOD_Experiment
from pgnn.preprocessing import gen_splits
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


class GraphData:
    def __init__(self, configuration: Configuration):
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
        
        self.plot_tsne(seed, self.graph.attr_matrix, self.graph.labels)
    
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
        # MPS fix -> repeats only one item in dataloader...
        #if torch.backends.mps.is_available():
        #    device = 'cpu'
        #else:
        device = get_device()
        
        if batch_size is None:
            batch_size = max((val.numel() for val in self.idx_all.values()))
        datasets = {phase: TensorDataset(ind.to(device), self.labels_all[ind].to(device), self.oods_all[ind].to(device)) for phase, ind in self.idx_all.items()}
        self.dataloaders = {phase: DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
                    for phase, dataset in datasets.items()}
    
class SBM:
    def init(graph_data: GraphData, seed):
        adj_matrix = SBM.generate_graph(graph_data, seed)
        features, labels = SBM.features(graph_data, seed)
        graph = SparseGraph(
            adj_matrix=adj_matrix,
            attr_matrix=features,
            labels=labels
        )
        
        graph.standardize(select_lcc=True)
        graph.normalize_features(experiment_configuration=graph_data.experiment_configuration)
        
        return graph
    
    def generate_graph(graph_data: GraphData, seed):
        # Create graph
        networkx_graph = nx.stochastic_block_model(
            graph_data.experiment_configuration.sbm_classes,
            SBM.sbm_connection_probabilities(graph_data),
            nodelist=None,
            seed=seed,
            directed=False,
            selfloops=False,
            sparse=True
        )
        
        """https://networkx.org/documentation/stable/auto_examples/drawing/plot_labels_and_colors.html"""
        options = {"edgecolors": "tab:gray", "node_size": 25, "alpha": 1}
        
        plt.figure(figsize=(12, 10))
        spring_k =4/sqrt(sum(graph_data.experiment_configuration.sbm_classes)) # default k is 1/sqrt
        pos = nx.spring_layout(networkx_graph, seed=seed, scale=1, k=spring_k)
        
        i = 0
        colors = ["green", "blue", "cyan", "red"]
        for sbm_class, color in zip(graph_data.experiment_configuration.sbm_classes, colors):
            nx.draw_networkx_nodes(networkx_graph, pos, nodelist=range(i, i+sbm_class), node_color=f"tab:{color}", **options)
            i+=sbm_class
        
        
        nx.draw_networkx_edges(networkx_graph, pos, width=0.5, alpha=0.5)
        
        plt.tight_layout()
        plt.legend()
        plt.savefig(f'{os.getcwd()}/plots/graphs/{seed}.png')
        plt.clf()
        
        n_nodes = sum(graph_data.experiment_configuration.sbm_classes)
        
        adj_matrix=sp.sparse.csr_matrix(
            ([1]*len(networkx_graph.edges), (list(map(lambda x: x[0], networkx_graph.edges)), list(map(lambda x: x[1], networkx_graph.edges)))), shape=(n_nodes, n_nodes)
        )
        
        return adj_matrix
    
    def sbm_connection_probabilities(graph_data: GraphData):
        # TODO: For SBM also allow frac ood, etc.
        
        N = len(graph_data.experiment_configuration.sbm_classes)
        
        P_ID_IN = graph_data.experiment_configuration.sbm_connection_probabilities_id_in_cluster
        P_ID_OUT = graph_data.experiment_configuration.sbm_connection_probabilities_id_out_cluster
        P_OOD_IN = graph_data.experiment_configuration.sbm_connection_probabilities_ood_in_cluster
        P_OOD_OUT = graph_data.experiment_configuration.sbm_connection_probabilities_ood_out_cluster
        
        # ID
        connection_probabilities = (np.ones([N, N]) - np.eye(N)) * P_ID_OUT
        connection_probabilities += np.diag(np.ones([N]) * P_ID_IN)
        
        if graph_data.experiment_configuration.ood == OOD.LOC:
            OOD_N = graph_data.experiment_configuration.ood_loc_num_classes
        
            connection_probabilities[N-OOD_N:, :] = P_OOD_OUT
            connection_probabilities[:, N-OOD_N:] = P_OOD_OUT
        
            for i in range(N-OOD_N, N):
                connection_probabilities[i,i] = P_OOD_IN
        
        return connection_probabilities
        
    def features(graph_data: GraphData, seed):
        features = []
        labels = []
        
        random_state = seed
        
        means = []
        
        for c, nsamples in enumerate(graph_data.experiment_configuration.sbm_classes):
            mean = sp.stats.norm.rvs(loc=graph_data.experiment_configuration.sbm_feature_mean, scale=graph_data.experiment_configuration.sbm_feature_variance ,size=graph_data.experiment_configuration.sbm_nfeatures, random_state=random_state)
            means.append(mean)
            
            samples = sp.stats.multivariate_normal(
                mean=mean,
                cov=np.diag(np.abs(sp.stats.norm.rvs(loc=graph_data.experiment_configuration.sbm_feature_sampling_variance, scale=1, size=graph_data.experiment_configuration.sbm_nfeatures, random_state=random_state+1))),
                seed=seed
            ).rvs(size=nsamples)
            
            random_state += 10
            
            features.append(samples)
            labels.extend([c] * nsamples)
            
        features = np.concatenate(features)
        labels = np.asarray(labels)
            
        graph_data.log_feature_distances(means)
            
        return features, labels
    
class ActiveLearning:
    def set_starting_class(graph_data: GraphData):
        if graph_data.experiment_configuration.active_learning_starting_class is not None:
            idx_training_left_out = graph_data.idx_all[Phase.TRAINING]
            idx_training_new = None
            for c in graph_data.experiment_configuration.active_learning_starting_class:
                new = idx_training_left_out[graph_data.labels_all[idx_training_left_out]==c]
                if idx_training_new is None:
                    idx_training_new = new
                else:
                    idx_training_new = torch.cat([idx_training_new, new]).to(new.device)
                    
                idx_training_left_out = idx_training_left_out[graph_data.labels_all[idx_training_left_out]!=c]

            graph_data.idx_all[Phase.TRAINING] = idx_training_new
            graph_data.idx_all[Phase.STOPPING] = torch.cat([graph_data.idx_all[Phase.STOPPING], idx_training_left_out]).to(idx_training_new.device)
    
    def __init__(self, configuration: Configuration):
        self.configuration = configuration
        self.budget = self.configuration.experiment.active_learning_budget
        self.budget_per_update = self.configuration.experiment.active_learning_budget_per_update
        self.selector = self.configuration.experiment.active_learning_selector
    
    def select(self, graph_data: GraphData, stopping_results: Results):
        idx_stopping = graph_data.idx_all[Phase.STOPPING]
        
        if self.selector == ActiveLearningSelector.RANDOM:
            random_order = torch.randperm(idx_stopping.shape[0])
            idx_stopping = idx_stopping[random_order]
            budget_for_update = min(self.budget_per_update, self.budget)
            return idx_stopping[:budget_for_update], idx_stopping[budget_for_update:]
        else:
            raise NotImplementedError()
            
            
    
    def update(self, graph_data: GraphData, stopping_results: Results):
        if self.budget == 0:
            return
        
        idx_new_training, idx_new_stopping = self.select(graph_data=graph_data, stopping_results=stopping_results)
        self.budget = max(0, self.budget - self.budget_per_update)
        
        graph_data.idx_all[Phase.TRAINING] = torch.cat([graph_data.idx_all[Phase.TRAINING], idx_new_training]).to(idx_new_training.device)
        graph_data.idx_all[Phase.STOPPING] = idx_new_stopping
        
        graph_data.init_dataloaders()