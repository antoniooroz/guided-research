from enum import Enum
from typing import Any, Optional

from pgnn.configuration.base_configuration import BaseConfiguration

class ExperimentConfiguration(BaseConfiguration):
    
    def __init__(self, dictionary: dict[str, Any] = None):
        from pgnn.result.result import NetworkMode
        from pgnn.configuration.model_configuration import UncertaintyMode
        
        self.dataset: Dataset = Dataset.CORA_ML
        self.num_classes: int = 10
    
        self.seeds: Seeds = Seeds()
        self.datapoints_training_per_class: int = 20
        self.datapoints_stopping: int = 500
        self.datapoints_known: int = 1500
        self.iterations_per_seed = 5 # Iterations per seed
        
        # Stochastic Block Model
        self.sbm_nfeatures = 30
        self.sbm_feature_mean = 0
        self.sbm_feature_variance = 10
        self.sbm_feature_sampling_variance = 10
        self.sbm_feature_sampling_variance_informed = 0.1
        self.sbm_ood_euclidian_distance = 10
        self.sbm_classes = [125, 125, 200, 125]
        self.sbm_connection_probabilities_id_in_cluster = 0.01
        self.sbm_connection_probabilities_id_out_cluster = 0.001
        self.sbm_connection_probabilities_ood_in_cluster = 0.01
        self.sbm_connection_probabilities_ood_out_cluster = 0.001
        
        self.sbm_al2_uninformative_layers = 3
        self.sbm_al3_uninformed_self_weight = 0.5
        
        self.training_type = None
        self.active_learning_training_type = None
        self.valtest_type: list[int] = None
        
        self.active_learning: bool = False
        self.active_learning_retrain: bool = False
        self.active_learning_start_cap_per_class: int = 20
        self.active_learning_budget: int = 50
        self.active_learning_budget_per_update: int = 10
        self.active_learning_starting_class: Optional[list[int]] = None
        self.active_learning_selector: ActiveLearningSelector = ActiveLearningSelector.RANDOM
        self.active_learning_selector_network_mode: NetworkMode = NetworkMode.ISOLATED
        self.active_learning_selector_uncertainty_mode: UncertaintyMode = UncertaintyMode.EPISTEMIC
        self.active_learning_l2_distance_logging: bool = False
        self.active_learning_l2_distance_use_centroids: bool = False
        
        # OOD
        self.ood: OOD = OOD.NONE 
        self.ood_loc_classes: list[int] = None
        self.ood_loc_num_classes: int = None
        self.ood_loc_frac: float = 0.45
        self.ood_loc_remove_classes: bool = False
        self.ood_loc_remove_edges: bool = True
        self.ood_perturb_train: bool = False
        self.ood_perturb_noise_scale: float = 1.0
        self.ood_perturb_bernoulli_probability: float = 0.5 # Used only for OODPerturbMode.BERNOULLI
        self.ood_perturb_budget: float = 0.1
        self.ood_perturb_mode: OODPerturbMode = OODPerturbMode.BERNOULLI_AUTO_P
        self.ood_normalize_attributes: list[AttributeNormalization] = [OODAttributeNormalization.DIV_BY_SUM, OODAttributeNormalization.MIRROR_NEGATIVE_VALUES]
        
        # GPN
        self.binary_attributes: bool = False
        self.normalize_attributes: AttributeNormalization = AttributeNormalization.DEFAULT
        
        self.from_dict(dictionary)
class ExperimentMode(Enum):
    DEVELOPMENT = 'development'
    TEST = 'test'
    
class Dataset(Enum):
    GENERATED_SBM = 'generated_sbm'
    GENERATED_SBM_AL = 'generated_sbm_al'
    GENERATED_SBM_AL2 = 'generated_sbm_al2'
    GENERATED_SBM_AL3 = 'generated_sbm_al3'
    CORA_ML = 'cora_ml'
    CITESEER = 'citeseer'

class OOD(Enum):
    NONE = 'none'
    LOC = 'loc'
    PERTURB = 'perturb'
    MIXED = 'mixed'
    
class OODPerturbMode(Enum):
    NORMAL = 'normal'
    BERNOULLI = 'bernoulli'
    BERNOULLI_AUTO_P = 'bernoulli_auto_p'
    SHUFFLE = 'shuffle'
    ZEROS = 'zeros'
    ONES = 'ones'

class Seeds(BaseConfiguration):
    def __init__(self, dictionary: dict[str, Any] = None):
        self.specific_seeds: list[int] = None
        self.start: int = 0
        self.end: int = 20
        self.experiment_mode: ExperimentMode = ExperimentMode.DEVELOPMENT
        
        self.from_dict(dictionary)
        
        self.seed_list = self._get_seeds()
        
    def _get_seeds(self) -> list[int]:
        test_seeds = [
            2144199730,  794209841, 2985733717, 2282690970, 1901557222,
            2009332812, 2266730407,  635625077, 3538425002,  960893189,
            497096336, 3940842554, 3594628340,  948012117, 3305901371,
            3644534211, 2297033685, 4092258879, 2590091101, 1694925034
        ]
        development_seeds = [
            2413340114, 3258769933, 1789234713, 2222151463, 2813247115,
            1920426428, 4272044734, 2092442742, 841404887, 2188879532,
            646784207, 1633698412, 2256863076,  374355442,  289680769,
            4281139389, 4263036964,  900418539,  119332950, 1628837138
        ]
        
        if self.specific_seeds:
            seed_list = self.specific_seeds
        elif self.experiment_mode==ExperimentMode.TEST:
            seed_list = test_seeds
        else:
            seed_list = development_seeds
        seed_list = seed_list[self.start:self.end]
        
        return seed_list
    
class ActiveLearningSelector(Enum):
    RANDOM = 0
    UNCERTAINTY = 1
    L2_DISTANCE = 2
    FIXED = 3
    FIXED_UNBALANCED = 4
    UNCERTAINTY_INVERTED = 5
    RANKED = 6
    
class OODAttributeNormalization(Enum):
    DIV_BY_SUM = 1
    MIRROR_NEGATIVE_VALUES = 2 

class AttributeNormalization(Enum):
    NONE = 0
    DEFAULT = 1
    DIV_BY_SUM = 2