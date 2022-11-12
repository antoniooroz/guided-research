
from pgnn.configuration.experiment_configuration import Dataset, ExperimentMode, ExperimentConfiguration
from pgnn.data.io import load_dataset
from pgnn.data.sparsegraph import SparseGraph
from pgnn.preprocessing import gen_splits


class GraphData:
    def __init__(self, experiment_configuration: ExperimentConfiguration):
        self.experiment_configuration = experiment_configuration
        
        if self.experiment_configuration.dataset != Dataset.GENERATED_SBM:
            self.graph = load_dataset(experiment_configuration.dataset.value)
            self.graph.standardize(select_lcc=True)
            self.graph.normalize_features(experiment_configuration=experiment_configuration)
    
    def get_graph(self, seed) -> SparseGraph:
        if self.experiment_configuration.dataset == Dataset.GENERATED_SBM:
            pass
        else:
            return self.graph
        
        
    def get_split(self, seed):
        if self.experiment_configuration.dataset == Dataset.GENERATED_SBM:
            pass
        else:
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
        
        