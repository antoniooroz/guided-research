from math import sqrt
from pgnn.base.network_mode import NetworkMode
from pgnn.configuration.experiment_configuration import ActiveLearningSelector
from pgnn.configuration.model_configuration import UncertaintyMode
from pgnn.configuration.training_configuration import Phase
from pgnn.data.active_learning import ActiveLearning
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

from pgnn.utils.utils import get_device, matrix_to_torch, final_run, balanced_weights

class GraphData:
    def __init__(self, configuration: Configuration):
        self.configuration = configuration
        self.experiment_configuration = configuration.experiment
        
        if not self.experiment_configuration.dataset.name.startswith('GENERATED'):
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
            self.graph = SBM().init_graph(self, seed)
        elif self.experiment_configuration.dataset == Dataset.GENERATED_SBM_AL:
            self.graph = SBM_AL().init_graph(self, seed)
        elif self.experiment_configuration.dataset == Dataset.GENERATED_SBM_AL2:
            self.graph = SBM_AL2().init_graph(self, seed)
        elif self.experiment_configuration.dataset == Dataset.GENERATED_SBM_AL3:
            self.graph = SBM_AL3().init_graph(self, seed)
            
        assert self.experiment_configuration.num_classes == max(self.graph.labels)+1
        
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
        if self.experiment_configuration.dataset in [Dataset.GENERATED_SBM_AL, Dataset.GENERATED_SBM_AL2, Dataset.GENERATED_SBM_AL3]:
            if self.experiment_configuration.dataset == Dataset.GENERATED_SBM_AL:
                N = 4
            elif self.experiment_configuration.dataset in [Dataset.GENERATED_SBM_AL2, Dataset.GENERATED_SBM_AL3]:
                N = 1+self.experiment_configuration.sbm_al2_uninformative_layers
            else:
                raise NotImplementedError()
            
            node_types = self.graph.node_names.astype('int64')
            
            split_ratio = []
            for c in range(self.experiment_configuration.num_classes):
                class_split_ratio = []
                for t in range(N):
                    class_split_ratio.append(self.experiment_configuration.sbm_classes[N*c + t])
                nodes_in_class = sum(class_split_ratio)
                for t in range(N):
                    class_split_ratio[t] = class_split_ratio[t] / nodes_in_class
                split_ratio.append(class_split_ratio)
        else:
            node_types = None
            split_ratio = None
            N = None
        
        is_sbm = self.experiment_configuration.dataset.name.startswith('GENERATED_SBM')
        
        if is_sbm:
            datapoints_known = self.graph.labels.shape[0]
        else:
            datapoints_known = self.experiment_configuration.datapoints_known
        
        idx_all = gen_splits(
            labels=self.graph.labels.astype('int64'), 
            idx_split_args={
                'ntrain_per_class': self.experiment_configuration.datapoints_training_per_class,
                'nstopping': self.experiment_configuration.datapoints_stopping,
                'nknown': datapoints_known,
                'seed': seed
            }, 
            test=self.experiment_configuration.seeds.experiment_mode==ExperimentMode.TEST,
            node_types = node_types,
            training_type=self.experiment_configuration.training_type,
            valtest_type=self.experiment_configuration.valtest_type,
            split_ratio=split_ratio,
            n_types_per_class=N,
            sbm=is_sbm
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
    def init_graph(self, graph_data: GraphData, seed):
        adj_matrix = self.generate_graph(graph_data, seed)
        features, labels, types = self.features(graph_data, seed)
        
        graph = SparseGraph(
            adj_matrix=adj_matrix,
            attr_matrix=features,
            labels=labels,
            node_names=types
        )
        
        graph.standardize(select_lcc=True)
        graph.normalize_features(experiment_configuration=graph_data.experiment_configuration)
        
        return graph
    
    def generate_graph(self, graph_data: GraphData, seed):
        # Create graph
        networkx_graph = nx.stochastic_block_model(
            graph_data.experiment_configuration.sbm_classes,
            self.sbm_connection_probabilities(graph_data),
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
    
    def sbm_connection_probabilities(self, graph_data: GraphData):
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
        
    def features(self, graph_data: GraphData, seed):
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
                seed=random_state
            ).rvs(size=nsamples)
            
            random_state += 10
            
            features.append(samples)
            labels.extend([c] * nsamples)
            
        features = np.concatenate(features)
        labels = np.asarray(labels)
            
        graph_data.log_feature_distances(means)
            
        return features, labels, None
    
class SBM_AL(SBM):        
    def sbm_connection_probabilities(self, graph_data: GraphData):
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
        
    def features(self, graph_data: GraphData, seed):
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
                seed=random_state
            ).rvs(size=nsamples)
            random_state += 10
            
            features.append(samples)
            labels.extend([c//4] * nsamples)
            types.extend([c%4] * nsamples)
            
        features = np.concatenate(features)
        labels = np.asarray(labels)
        types = np.asarray(types)
            
        graph_data.log_feature_distances(means)
            
        return features, labels, types
    
class SBM_AL2(SBM_AL):
    def sbm_connection_probabilities(self, graph_data: GraphData):
        N = len(graph_data.experiment_configuration.sbm_classes)
        UNINFORMATIVE_LAYERS = graph_data.experiment_configuration.sbm_al2_uninformative_layers
        LAYERS_PER_CLASS = 1 + UNINFORMATIVE_LAYERS
        
        assert N%LAYERS_PER_CLASS==0, "central informative nodes + number of uninformed layers per class"
        
        P_IN = graph_data.experiment_configuration.sbm_connection_probabilities_id_in_cluster
        P_OUT = graph_data.experiment_configuration.sbm_connection_probabilities_id_out_cluster
        
        edge = np.ones([1,N]) * P_OUT
        for i in range(N//LAYERS_PER_CLASS):
            start = i*LAYERS_PER_CLASS
            edge[0][start:start+LAYERS_PER_CLASS-1] = 0
            
        central = np.zeros([1,N])
        
        connection_probabilities = []
        for i in range(N//LAYERS_PER_CLASS):
            start = i*LAYERS_PER_CLASS
            
            # Central layers
            for j in range(start, start+LAYERS_PER_CLASS-1):
                central_with_in_class = central.copy()
                start_for_in_class = j if j==start else j-1
                central_with_in_class[0][start_for_in_class:j+2] = P_IN
                connection_probabilities.append(central_with_in_class)

            # edge layer
            edge_with_in_class = edge.copy()
            start_for_in_class = start+LAYERS_PER_CLASS-2
            edge_with_in_class[0][start_for_in_class:start_for_in_class+2] = P_IN
            connection_probabilities.append(edge_with_in_class)
        
        return np.concatenate(connection_probabilities, axis=0)
        
    def features(self, graph_data: GraphData, seed):
        UNINFORMATIVE_LAYERS = graph_data.experiment_configuration.sbm_al2_uninformative_layers
        LAYERS_PER_CLASS = 1 + UNINFORMATIVE_LAYERS
        
        features = []
        labels = []
        types = []
        
        random_state = seed
        
        means = []
        
        for _ in range(len(graph_data.experiment_configuration.sbm_classes)//LAYERS_PER_CLASS):
            means.append(sp.stats.norm.rvs(
                loc=graph_data.experiment_configuration.sbm_feature_mean, 
                scale=graph_data.experiment_configuration.sbm_feature_variance,
                size=graph_data.experiment_configuration.sbm_nfeatures, 
                random_state=random_state
            ))
            
            random_state += 10
        
        for c, nsamples in enumerate(graph_data.experiment_configuration.sbm_classes):
            is_informed = c%LAYERS_PER_CLASS==0 
            mean = means[c//LAYERS_PER_CLASS]
            
            if is_informed:
                variance = graph_data.experiment_configuration.sbm_feature_sampling_variance_informed
            else:
                variance = graph_data.experiment_configuration.sbm_feature_sampling_variance
            
            samples = sp.stats.multivariate_normal(
                mean=mean,
                cov=np.diag([variance] * mean.shape[0]),
                seed=random_state
            ).rvs(size=nsamples)
            random_state+=10
            
            features.append(samples)
            labels.extend([c//LAYERS_PER_CLASS] * nsamples)
            types.extend([c%LAYERS_PER_CLASS] * nsamples)
            
        features = np.concatenate(features)
        labels = np.asarray(labels)
        types = np.asarray(types)
            
        graph_data.log_feature_distances(means)
            
        return features, labels, types
    
class SBM_AL3(SBM_AL2):
    def features(self, graph_data: GraphData, seed):
        UNINFORMATIVE_LAYERS = graph_data.experiment_configuration.sbm_al2_uninformative_layers
        LAYERS_PER_CLASS = 1 + UNINFORMATIVE_LAYERS
        CLASSES = len(graph_data.experiment_configuration.sbm_classes)//LAYERS_PER_CLASS
        
        features = []
        labels = []
        types = []
        
        random_state = seed
        
        means = []
        
        for _ in range(CLASSES):
            means.append(sp.stats.norm.rvs(
                loc=graph_data.experiment_configuration.sbm_feature_mean, 
                scale=graph_data.experiment_configuration.sbm_feature_variance,
                size=graph_data.experiment_configuration.sbm_nfeatures, 
                random_state=random_state
            ))
            random_state += 10
            
        mean_matrix = np.concatenate([np.expand_dims(m, axis=0) for m in means], axis=0)
        
        for c, nsamples in enumerate(graph_data.experiment_configuration.sbm_classes):
            is_informed = c%LAYERS_PER_CLASS==0 
            mean = means[c//LAYERS_PER_CLASS]
            
            variance = graph_data.experiment_configuration.sbm_feature_sampling_variance_informed
            self_weight = graph_data.experiment_configuration.sbm_al3_uninformed_self_weight/CLASSES
            
            if not is_informed:
                mean_weights = np.abs(sp.stats.norm.rvs(
                    loc=0, 
                    scale=1,
                    size=CLASSES, 
                    random_state=random_state
                ))
                random_state+=10
                
                mean_weights[c//LAYERS_PER_CLASS] = 0
                
                mean_weights = mean_weights / mean_weights.sum()
                means_without_self = mean_weights@mean_matrix
                
                mean = self_weight*mean + (1-self_weight)*means_without_self
                
            samples = sp.stats.multivariate_normal(
                mean=mean,
                cov=np.diag([variance] * mean.shape[0]),
                seed=random_state
            ).rvs(size=nsamples)
            random_state+=10
            
            features.append(samples)
            labels.extend([c//LAYERS_PER_CLASS] * nsamples)
            types.extend([c%LAYERS_PER_CLASS] * nsamples)
            
        features = np.concatenate(features)
        labels = np.asarray(labels)
        types = np.asarray(types)
            
        graph_data.log_feature_distances(means)
            
        return features, labels, types