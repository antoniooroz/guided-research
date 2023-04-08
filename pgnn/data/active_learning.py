import torch
import numpy as np

from pgnn.utils.utils import final_run, balanced_weights
from pgnn.configuration.training_configuration import Phase
from pgnn.configuration.model_configuration import UncertaintyMode
from pgnn.configuration.experiment_configuration import ActiveLearningSelector
from pgnn.result import Results
from pgnn.configuration import Configuration


class ActiveLearning:    
    def __init__(self, configuration: Configuration):
        self.configuration = configuration
        
        self.budget = self.configuration.experiment.active_learning_budget
        self.budget_per_update = self.configuration.experiment.active_learning_budget_per_update
        
        self.selector = self.configuration.experiment.active_learning_selector
        self.network_mode = self.configuration.experiment.active_learning_selector_network_mode
        self.uncertainty_mode = self.configuration.experiment.active_learning_selector_uncertainty_mode
        self.l2_distance_logging = self.configuration.experiment.active_learning_l2_distance_logging
        self.l2_distance_use_centroids = self.configuration.experiment.active_learning_l2_distance_use_centroids
        
        self.saved_loss = None
        self.saved_epoch = 0
        
    def set_starting_class(graph_data):
        if graph_data.experiment_configuration.active_learning_starting_class is None:
            return
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
    
    def select(self, graph_data, active_learning_results: Results):
        idx_active_learning = graph_data.idx_all[Phase.ACTIVE_LEARNING]
        budget_for_update = min(self.budget_per_update, self.budget)
        
        if self.l2_distance_logging or self.selector == ActiveLearningSelector.L2_DISTANCE:
            l2_distances = self._selector_l2_distances(graph_data)
        else:
            l2_distances = None
        
        if self.selector == ActiveLearningSelector.RANDOM:
            order = torch.randperm(idx_active_learning.shape[0])
        elif self.selector == ActiveLearningSelector.UNCERTAINTY:
            order = self._selector_uncertainty(active_learning_results=active_learning_results, invert=False)
        elif self.selector == ActiveLearningSelector.UNCERTAINTY_INVERTED:
            order = self._selector_uncertainty(active_learning_results=active_learning_results, invert=True)
        elif self.selector == ActiveLearningSelector.RANKED:
            order = self._selector_ranked(active_learning_results=active_learning_results)
        elif self.selector == ActiveLearningSelector.FIXED_UNBALANCED:
            order = self._selector_fixed_unbalanced(
                graph_data=graph_data,
                idx_active_learning=idx_active_learning
            )
        elif self.selector == ActiveLearningSelector.FIXED:
            order = self._selector_fixed(
                graph_data=graph_data,
                idx_active_learning=idx_active_learning,
                budget_for_update=budget_for_update
            )
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
    
    def _selector_uncertainty(self, active_learning_results: Results, invert=False):
        resultsForNetworkMode = active_learning_results.networkModeResults[self.network_mode]
        
        inverter = -1 if invert else 1
            
        if self.uncertainty_mode == UncertaintyMode.ALEATORIC: 
            uncertainties = resultsForNetworkMode._aleatoric_uncertainties * inverter
        else:
            uncertainties = resultsForNetworkMode._epistemic_uncertainties * inverter
        
        return uncertainties.argsort(axis=0)
    
    def _selector_ranked(self, active_learning_results: Results):
        resultsForNetworkMode = active_learning_results.networkModeResults[self.network_mode]
        rank_aleatoric = resultsForNetworkMode._aleatoric_uncertainties.argsort(axis=0).argsort(axis=0)
        rank_epistemic = resultsForNetworkMode._epistemic_uncertainties.argsort(axis=0).argsort(axis=0)
        
        # high epistemic, low aleatoric
        rank = rank_epistemic - rank_aleatoric
        
        return rank.argsort(axis=0)
        
    
    def _selector_fixed_unbalanced(self, graph_data, idx_active_learning):
        types = graph_data.types[idx_active_learning.cpu()]
        idx = np.arange(idx_active_learning.shape[0])
        select = np.zeros(idx_active_learning.shape[0])
        for t in graph_data.experiment_configuration.active_learning_training_type:
            select += (types == t)
            
        idx_for_type = np.random.permutation(idx[select==1])
        
        if idx_for_type.shape[0] == idx.shape[0]:
            return idx_for_type
        else:
            other_idx = np.random.permutation(idx[select==0])
            return np.concatenate([other_idx, idx_for_type], axis=0)
        
    def _selector_fixed(self, graph_data, idx_active_learning, budget_for_update):
        labels_all_numpy = graph_data.labels_all.cpu().numpy()
        labels = labels_all_numpy[idx_active_learning.cpu()]
        
        assert budget_for_update % (labels_all_numpy.max()+1) == 0 or budget_for_update == 1
        
        if budget_for_update == 1:
            # Take class with least samples
            budget_per_class = budget_for_update
            
            labels_training = labels_all_numpy[graph_data.idx_all[Phase.TRAINING].cpu()]
            
            if np.isscalar(labels_training):
                labels_training = np.array([labels_training])
            
            # also regard labels which are not in set
            labels_training = np.concatenate([
                labels_training, 
                np.arange(graph_data.configuration.experiment.num_classes)
            ], axis=0)
            
            labels_training_unique, labels_training_counts = np.unique(labels_training, return_counts=True)
            
            label_classes_range = [labels_training_unique[labels_training_counts.argmin()]]
            print(label_classes_range)
        else:
            # Take equal amount of nodes for each class
            budget_per_class = budget_for_update // (labels_all_numpy.max().item()+1)
            
            label_classes_range = range(labels_all_numpy.max()+1)
        
        types = graph_data.types[idx_active_learning.cpu()]
            
        importance = np.zeros(idx_active_learning.shape[0])
        
        for c in label_classes_range:
            select = np.zeros(idx_active_learning.shape[0])
            for t in graph_data.experiment_configuration.active_learning_training_type:
                select += (labels == c) * (types == t)
            
            assert select.sum() >= budget_per_class
            
            make_important_indeces = np.arange(select.shape[0])[select==1]
            make_important_indeces = np.random.permutation(make_important_indeces)
            importance[make_important_indeces[:budget_per_class]] = 1.0
                
        return importance.argsort()
    
    def _selector_l2_distances(self, graph_data):
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
    
    def step(self, graph_data, early_stopping, logger, model, cycle):
        if not self.configuration.experiment.active_learning or self.budget == 0:
            return
        
        early_stopping.load_best()
        
        activeLearningResults = final_run(model, graph_data.feature_matrix, graph_data.idx_all, graph_data.labels_all, graph_data.oods_all)
        
        active_learning_update_logs = self.update(
            graph_data=graph_data,
            active_learning_results=activeLearningResults[Phase.ACTIVE_LEARNING],
            early_stopping=early_stopping
        )
        
        activeLearningResults[Phase.ACTIVE_LEARNING].info.mean_l2_distance_in = active_learning_update_logs['mean_l2_distance_in']
        activeLearningResults[Phase.ACTIVE_LEARNING].info.mean_l2_distance_out = active_learning_update_logs['mean_l2_distance_out']
        activeLearningResults[Phase.ACTIVE_LEARNING].info.active_learning_added_nodes = active_learning_update_logs['added_nodes']
        logger.logActiveLearning(resultsPerPhase=activeLearningResults, step=cycle)
        
        # Early Stopping
        early_stopping.reset_best()
        if graph_data.configuration.experiment.active_learning_retrain:
            early_stopping.load_first()
        
        # Balanced loss
        if self.configuration.training.balanced_loss:
            loss_balance_weights = balanced_weights(
                n_classes=graph_data.nclasses, 
                labels=graph_data.labels_all[graph_data.idx_all[Phase.TRAINING]]
            )
        else:
            loss_balance_weights = None
            
        return loss_balance_weights
            
    def update(self, graph_data, active_learning_results: Results, early_stopping = None):            
        idx_new_training, idx_new_active_learning, mean_l2_distance_in, mean_l2_distance_out = self.select(graph_data=graph_data, active_learning_results=active_learning_results)
        
        graph_data.idx_all[Phase.TRAINING] = torch.cat([graph_data.idx_all[Phase.TRAINING], idx_new_training]).to(idx_new_training.device)
        graph_data.idx_all[Phase.ACTIVE_LEARNING] = idx_new_active_learning
        
        graph_data.init_dataloaders()
        
        self.budget = max(0, self.budget - self.budget_per_update)
        
        return {
            'mean_l2_distance_in': mean_l2_distance_in,
            'mean_l2_distance_out': mean_l2_distance_out,
            'added_nodes': 1
        }