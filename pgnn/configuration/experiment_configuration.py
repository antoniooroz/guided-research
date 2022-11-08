from enum import Enum
from typing import Any

from pgnn.configuration.base_configuration import BaseConfiguration

class ExperimentConfiguration(BaseConfiguration):
    def __init__(self, dictionary: dict[str, Any] = None):
        self.dataset : Dataset = Dataset.CORA_ML
        self.seeds: Seeds = Seeds()
        self.datapoints_training_per_class: int = 20
        self.datapoints_stopping: int = 500
        self.datapoints_known: int = 1500
        self.iterations_per_seed = 5 # Iterations per seed
        
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
    

    
class OODAttributeNormalization(Enum):
    DIV_BY_SUM = 1
    MIRROR_NEGATIVE_VALUES = 2 

class AttributeNormalization(Enum):
    NONE = 0
    DEFAULT = 1
    DIV_BY_SUM = 2