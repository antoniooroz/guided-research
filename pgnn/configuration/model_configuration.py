from enum import Enum, auto
from typing import Any
from pgnn.configuration.base_configuration import BaseConfiguration

class ModelConfiguration(BaseConfiguration):
    def __init__(self, dictionary: dict[str, Any] = None):
        # Default
        self.type: ModelType = ModelType.PPNP
        self.bias: bool = False
        self.hidden_layer_size: list[int] = [64]
        
        # GAT
        self.gat_hidden_layers_heads: list[int] = [8]
        self.gat_skip_connections: bool = True 
        
        # PPNP 
        self.disable_dropout_on_input: bool = False
        self.ppnp_power_iterations = 10
        self.ppnp_teleportation_alpha = 0.1 # 0.2 for microsoft academic        

        # Bayesian
        self.samples_prediction: int = 10
        self.samples_training: int = 3
        self.initial_mean: float = 0.0
        self.initial_var: float = 0.0
        self.train_mean: bool = True
        self.train_var: bool = True
        #self.pred_score: str = 'softmax'
        self.weight_prior: str = 'normal'
        self.guide_init_scale: float = 0.1
        self.uncertainty_estimation: str = 'entropy_per_sample_mean'
        self.vectorize: bool = True
        self.network_mode_uncertainty_aggregation_network_lambda: float = 0.5
        self.network_mode_uncertainty_aggregation_normalize: bool = False
        self.network_combination: NucType = NucType.MEAN
        
        # GPN
        self.gpn_model: dict[str, Any] = {}
        
        
        self.from_dict(dictionary)
        
class NucType(Enum):    
    MAX = 'max'
    MIN = 'min'
    MEAN = 'mean'
    
class ModelType(Enum):    
    PPNP = 'PPNP'
    P_PPNP = 'P_PPNP'
    MIXED_PPNP = 'P_PPNP'
    MCD_PPNP = 'MCD_PPNP'
    
    GCN = 'GCN'
    P_GCN = 'P_GCN'
    MIXED_GCN = 'P_GCN'
    MCD_GCN = 'MCD_GCN'
    DE_GCN = 'MCD_GCN'
    
    GAT = 'GAT'
    MCD_GAT = 'MCD_GAT'
    P_GAT = 'P_GAT'
    P_ATT_GAT = 'P_GAT'
    P_PROJ_GAT = 'P_GAT'
    MIXED_GAT = 'P_GAT'
    MIXED_ATT_GAT = 'P_GAT'
    MIXED_PROJ_GAT = 'P_GAT'
    
    GPN = 'GPN'
    
    # TODO ...    
    def gat_bayesian_projection() -> list['ModelType']:
        return [ModelType.P_GAT, ModelType.P_PROJ_GAT, ModelType.MIXED_GAT, ModelType.MIXED_PROJ_GAT]

    def gat_bayesian_attention() -> list['ModelType']:
        return [ModelType.P_GAT, ModelType.P_ATT_GAT, ModelType.MIXED_GAT, ModelType.MIXED_ATT_GAT]
    
    def mcds() -> list['ModelType']:
        return [ModelType.MCD_PPNP]
    
    def get_base_type(model_type: 'ModelType') -> 'ModelType':
        if model_type == ModelType.MCD_PPNP:
            return ModelType.PPNP
        else:
            raise NotImplementedError()