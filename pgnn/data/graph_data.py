from pgnn.configuration.configuration import Configuration
from pgnn.configuration.experiment_configuration import Dataset, ExperimentMode
from pgnn.data.io import load_dataset
from pgnn.data.sparsegraph import SparseGraph
from pgnn.preprocessing import gen_splits


class GraphData:
    def __init__(self, configuration: Configuration):
        self.configuration = configuration
        
        if self.configuration.experiment.dataset != Dataset.GENERATED_SBM:
            self.graph = load_dataset(configuration.experiment.dataset.value)
            self.graph.standardize(select_lcc=True)
            self.graph.normalize_features(configuration=configuration)
    
    def get_graph(self, seed) -> SparseGraph:
        if self.configuration.experiment.dataset == Dataset.GENERATED_SBM:
            pass
        else:
            return self.graph
        
        
    def get_split(self, seed):
        if self.configuration.experiment.dataset == Dataset.GENERATED_SBM:
            pass
        else:
            idx_all = gen_splits(
                labels=self.graph.labels.astype('int64'), 
                idx_split_args={
                    'ntrain_per_class': self.configuration.experiment.datapoints_training_per_class,
                    'nstopping': self.configuration.experiment.datapoints_stopping,
                    'nknown': self.configuration.experiment.datapoints_known,
                    'seed': seed
                }, 
                test=self.configuration.experiment.seeds.experiment_mode==ExperimentMode.TEST
            )
            return idx_all
        
        