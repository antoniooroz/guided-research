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
            
    def update(self, graph_data: GraphData, active_learning_results: Results, epoch: int = 0, loss: float = 0):
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
        
        return {
            'mean_l2_distance_in': mean_l2_distance_in,
            'mean_l2_distance_out': mean_l2_distance_out,
            'added_nodes': 1
        }