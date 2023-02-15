
from math import sqrt
from pgnn.base.network_mode import NetworkMode
from pgnn.configuration.experiment_configuration import ActiveLearningSelector
from pgnn.configuration.model_configuration import UncertaintyMode
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
        
        if self.experiment_configuration.dataset not in [Dataset.GENERATED_SBM, Dataset.GENERATED_SBM_AL]:
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
        self.types = None
        
    def init(self, seed) -> tuple[torch.Tensor, DataLoader]:
        device = get_device()
        
        self.init_graph(seed)
        self.idx_all = self.get_split(seed)
        
        self.labels_all = torch.LongTensor(self.graph.labels.astype('int64')).to(device)
        self.oods_all = torch.zeros(self.labels_all.shape).to(device)
        self.feature_matrix = self.graph.attr_matrix.clone().detach().to(device)
        self.adjacency_matrix = matrix_to_torch(self.graph.adj_matrix)
        self.types = self.graph.node_names
        
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
        elif self.experiment_configuration.dataset == Dataset.GENERATED_SBM_AL:
            self.graph = SBM_AL.init(self, seed)
        
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
            test=self.experiment_configuration.seeds.experiment_mode==ExperimentMode.TEST,
            node_types = self.graph.node_names.astype('int64'),
            training_type=self.experiment_configuration.training_type
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
        import time
        start_time = time.time()
        
        adj_matrix = SBM.generate_graph(graph_data, seed)
        end_time_adj_matrix = time.time()
        print(f'adj_matrix: {end_time_adj_matrix-start_time}')
        
        features, labels = SBM.features(graph_data, seed)
        end_time_features = time.time()
        print(f'features: {end_time_features-end_time_adj_matrix}')
        
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
        
        """
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
        """
        
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
    
class SBM_AL:
    def init(graph_data: GraphData, seed):
        import time
        start_time = time.time()
        
        adj_matrix = SBM_AL.generate_graph(graph_data, seed)
        end_time_adj_matrix = time.time()
        print(f'adj_matrix: {end_time_adj_matrix-start_time}')
        
        features, labels, types = SBM_AL.features(graph_data, seed)
        end_time_features = time.time()
        print(f'features: {end_time_features-end_time_adj_matrix}')
        
        graph = SparseGraph(
            adj_matrix=adj_matrix,
            attr_matrix=features,
            labels=labels,
            node_names=types
        )
        
        graph.standardize(select_lcc=True)
        graph.normalize_features(experiment_configuration=graph_data.experiment_configuration)
        
        return graph
    
    def generate_graph(graph_data: GraphData, seed):
        # Create graph
        networkx_graph = nx.stochastic_block_model(
            graph_data.experiment_configuration.sbm_classes,
            SBM_AL.sbm_connection_probabilities(graph_data),
            nodelist=None,
            seed=seed,
            directed=False,
            selfloops=False,
            sparse=True
        )
        """
        https://networkx.org/documentation/stable/auto_examples/drawing/plot_labels_and_colors.html"""
        
        
        """
        options = {"edgecolors": "tab:gray", "node_size": 100, "alpha": 1}
        
        plt.figure(figsize=(12, 10))
        spring_k =4/sqrt(sum(graph_data.experiment_configuration.sbm_classes)) # default k is 1/sqrt
        pos = nx.spring_layout(networkx_graph, seed=seed, scale=1, k=spring_k)
        
        i = 0
        colors = (["#00A8E8"] * 2) + (["#003459"]*2) + (["#F45B69"] * 2) + (["#5A0001"] * 2) + (["#C3FFD7"] * 2) + (["#0C8346"] * 2)
        for sbm_class, color in zip(graph_data.experiment_configuration.sbm_classes, colors):
            nx.draw_networkx_nodes(networkx_graph, pos, nodelist=range(i, i+sbm_class), node_color=color, **options)
            i+=sbm_class
        
        nx.draw_networkx_edges(networkx_graph, pos, width=0.5, alpha=0.5)
        
        plt.tight_layout()
        plt.legend()
        plt.savefig(f'{os.getcwd()}/plots/graphs/sbmal_{seed}.png')
        plt.clf()
        """
        
        n_nodes = sum(graph_data.experiment_configuration.sbm_classes)
        
        adj_matrix=sp.sparse.csr_matrix(
            ([1]*len(networkx_graph.edges), (list(map(lambda x: x[0], networkx_graph.edges)), list(map(lambda x: x[1], networkx_graph.edges)))), shape=(n_nodes, n_nodes)
        )
        
        return adj_matrix
    
    def sbm_connection_probabilities(graph_data: GraphData):
        # TODO: For SBM also allow frac ood, etc.
        
        N = len(graph_data.experiment_configuration.sbm_classes)
        
        assert N%4==0, "Need 4 types per class, Informed Central, Uninformed Central, Informed Edge, Uninformed Edge"
        
        P_IN = graph_data.experiment_configuration.sbm_connection_probabilities_id_in_cluster
        P_OUT = graph_data.experiment_configuration.sbm_connection_probabilities_id_out_cluster
        
        edge = np.ones([1,N]) * P_OUT
        for i in range(N//4):
            start = i*4
            edge[0][start:start+2] = 0
            
        central = np.zeros([1,N])
        
        connection_probabilities = []
        for i in range(N//4):
            start = i*4
            edge_with_in_class = edge.copy()
            central_with_in_class = central.copy()
            edge_with_in_class[0][start:start+4] = P_IN
            central_with_in_class[0][start:start+4] = P_IN

            connection_probabilities.append(central_with_in_class)
            connection_probabilities.append(central_with_in_class)
            
            connection_probabilities.append(edge_with_in_class)
            connection_probabilities.append(edge_with_in_class)
        
        return np.concatenate(connection_probabilities, axis=0)
        
    def features(graph_data: GraphData, seed):
        features = []
        labels = []
        types = []
        
        random_state = seed
        
        means = []
        
        for _ in range(len(graph_data.experiment_configuration.sbm_classes)//4):
            means.append(sp.stats.norm.rvs(
                loc=graph_data.experiment_configuration.sbm_feature_mean, 
                scale=graph_data.experiment_configuration.sbm_feature_variance,
                size=graph_data.experiment_configuration.sbm_nfeatures, 
                random_state=random_state
            ))
            
            random_state += 10
        
        for c, nsamples in enumerate(graph_data.experiment_configuration.sbm_classes):
            is_informed = c%2==0 
            mean = means[c//4]
            
            variance = graph_data.experiment_configuration.sbm_feature_sampling_variance_informed if is_informed else graph_data.experiment_configuration.sbm_feature_sampling_variance
            
            samples = sp.stats.multivariate_normal(
                mean=mean,
                cov=np.diag([variance] * mean.shape[0]),
                seed=seed
            ).rvs(size=nsamples)
            
            features.append(samples)
            labels.extend([c//4] * nsamples)
            types.extend([c%4] * nsamples)
            
        features = np.concatenate(features)
        labels = np.asarray(labels)
        types = np.asarray(types)
            
        graph_data.log_feature_distances(means)
            
        return features, labels, types
    
class ActiveLearning:    
    def set_starting_class(graph_data: GraphData):
        if graph_data.experiment_configuration.active_learning_starting_class is not None:
            idx_training_left_out = graph_data.idx_all[Phase.TRAINING]
            idx_training_new = None
            for c in graph_data.experiment_configuration.active_learning_starting_class:
                new = idx_training_left_out[graph_data.labels_all[idx_training_left_out]==c]
                idx_training_left_out = idx_training_left_out[graph_data.labels_all[idx_training_left_out]!=c]
                
                if graph_data.experiment_configuration.active_learning_start_cap_per_class < graph_data.experiment_configuration.datapoints_training_per_class:
                    order = torch.randperm(new.shape[0])
                    new = new[order]
                    
                    left_out = new[graph_data.experiment_configuration.active_learning_start_cap_per_class:]
                    idx_training_left_out = torch.cat([idx_training_left_out, left_out], dim=0).to(left_out.device)
                    
                    new = new[:graph_data.experiment_configuration.active_learning_start_cap_per_class]
                
                if idx_training_new is None:
                    idx_training_new = new
                else:
                    idx_training_new = torch.cat([idx_training_new, new]).to(new.device)

            graph_data.idx_all[Phase.TRAINING] = idx_training_new
            graph_data.idx_all[Phase.ACTIVE_LEARNING] = idx_training_left_out.to(idx_training_new.device)
    
    def __init__(self, configuration: Configuration):
        self.configuration = configuration
        
        self.dynamic_update = self.configuration.experiment.active_learning_dynamic_update
        self.dynamic_update_patience = self.configuration.experiment.active_learning_dynamic_update_patience
        self.update_interval = self.configuration.experiment.active_learning_update_interval
        
        self.budget = self.configuration.experiment.active_learning_budget
        self.budget_per_update = self.configuration.experiment.active_learning_budget_per_update
        
        self.selector = self.configuration.experiment.active_learning_selector
        self.network_mode = self.configuration.experiment.active_learning_selector_network_mode
        self.uncertainty_mode = self.configuration.experiment.active_learning_selector_uncertainty_mode
        self.l2_distance_logging = self.configuration.experiment.active_learning_l2_distance_logging
        self.l2_distance_use_centroids = self.configuration.experiment.active_learning_l2_distance_use_centroids
        
        self.saved_loss = None
        self.saved_epoch = 0
    
    def select(self, graph_data: GraphData, active_learning_results: Results):
        idx_active_learning = graph_data.idx_all[Phase.ACTIVE_LEARNING]
        budget_for_update = min(self.budget_per_update, self.budget)
        
        if self.l2_distance_logging or self.selector == ActiveLearningSelector.L2_DISTANCE:
            l2_distances = self._l2_distances(graph_data)
        else:
            l2_distances = None
        
        if self.selector == ActiveLearningSelector.RANDOM:
            order = torch.randperm(idx_active_learning.shape[0])
        elif self.selector == ActiveLearningSelector.UNCERTAINTY:
            resultsForNetworkMode = active_learning_results.networkModeResults[self.network_mode]
            
            if self.uncertainty_mode == UncertaintyMode.ALEATORIC: 
                uncertainties = resultsForNetworkMode._aleatoric_uncertainties
            else:
                uncertainties = resultsForNetworkMode._epistemic_uncertainties
            
            order = uncertainties.argsort(axis=0)
        elif self.selector == ActiveLearningSelector.FIXED:
            labels = graph_data.labels_all[idx_active_learning]
            types = graph_data.types[idx_active_learning]
            budget_per_class = budget_for_update // labels.max().item()+1
            
            importance = np.zeros(idx_active_learning.shape[0])
            
            for c in range(labels.max().item()+1):
                select = np.zeros(idx_active_learning.shape[0])
                for t in graph_data.experiment_configuration.active_learning_training_type:
                    select += (labels == c) * (types == t)
                make_important_indeces = select.argsort().flip()
                importance[make_important_indeces[:budget_per_class]] = 1.0
                    
            order = importance.argsort()
        elif self.selector == ActiveLearningSelector.L2_DISTANCE:
            order = l2_distances.cpu().numpy().argsort(axis=0)
        else:
            raise NotImplementedError()
        
        idx_active_learning = idx_active_learning[order]
        
        split_index = idx_active_learning.shape[0]-budget_for_update
        
        # L2 Distance Logging
        mean_l2_distance_in = None
        mean_l2_distance_out = None
        if l2_distances is not None and self.l2_distance_logging:
            l2_distances = l2_distances[order]
            mean_l2_distance_in = l2_distances[split_index:].mean().item()
            mean_l2_distance_out = l2_distances[:split_index].mean().item()
        
        #      Additional Training Samples      , Leftover Active Learning Samples , mean_l2_distance in, mean_l2_distance_left_out
        return idx_active_learning[split_index:], idx_active_learning[:split_index], mean_l2_distance_in, mean_l2_distance_out
    
    def _l2_distances(self, graph_data: GraphData):
        training_labels = graph_data.labels_all[graph_data.idx_all[Phase.TRAINING]]
        training_features = graph_data.feature_matrix[graph_data.idx_all[Phase.TRAINING]]
        
        if self.l2_distance_use_centroids:
            training_labels_unique = training_labels.unique()
            
            centroids = []
            for c in training_labels_unique:
                features_in_class = training_features[training_labels == c]
                centroid = features_in_class.mean(0)
                centroids.append(centroid.unsqueeze(0))
            
            compare_to = torch.cat(centroids, dim=0).to(training_features.device)
        else:
            compare_to = training_features
        
        active_learning_features = graph_data.feature_matrix[graph_data.idx_all[Phase.ACTIVE_LEARNING]]
        
        l2_distances = torch.cdist(active_learning_features, compare_to, p=2)
        
        l2_distances = l2_distances.min(dim=-1).values
        
        return l2_distances
            
    def _should_update(self, epoch: int = 0, loss: float = 0):
        if self.dynamic_update:
            if self.saved_loss is None or loss < self.saved_loss:
                self.saved_loss = loss
                self.saved_epoch = epoch 
                return False
            elif epoch-self.saved_epoch >= self.dynamic_update_patience:
                self.saved_loss = None
                self.saved_epoch = 0
                return True
            else:
                return False
        elif epoch >= self.update_interval and epoch%self.update_interval==0:
            return True
        else:
            return False
            
    def update(self, graph_data: GraphData, active_learning_results: Results, epoch: int = 0, loss: float = 0, early_stopping = None, training_phase = None):
        if self.budget == 0 or not self._should_update(epoch, loss):
            return {
                'mean_l2_distance_in': None,
                'mean_l2_distance_out': None,
                'added_nodes': 0
            }
        idx_new_training, idx_new_active_learning, mean_l2_distance_in, mean_l2_distance_out = self.select(graph_data=graph_data, active_learning_results=active_learning_results)
        
        graph_data.idx_all[Phase.TRAINING] = torch.cat([graph_data.idx_all[Phase.TRAINING], idx_new_training]).to(idx_new_training.device)
        graph_data.idx_all[Phase.ACTIVE_LEARNING] = idx_new_active_learning
        
        graph_data.init_dataloaders()
        
        self.budget = max(0, self.budget - self.budget_per_update)
        
        early_stopping.init_for_training_phase(
            enabled=graph_data.configuration.training.early_stopping[training_phase],
            patience=graph_data.configuration.training.patience[training_phase],
            max_epochs=graph_data.configuration.training.max_epochs[training_phase],
            reset_best=True,
        )
        
        if graph_data.configuration.experiment.active_learning_retrain:
            early_stopping.load_first()
        
        return {
            'mean_l2_distance_in': mean_l2_distance_in,
            'mean_l2_distance_out': mean_l2_distance_out,
            'added_nodes': 1
        }