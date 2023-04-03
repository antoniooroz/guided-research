from typing import Any
from pgnn.configuration.base_configuration import BaseConfiguration
from pgnn.configuration.experiment_configuration import ExperimentConfiguration

from pgnn.configuration.training_configuration import TrainingConfiguration
from pgnn.configuration.model_configuration import ModelConfiguration

class Configuration(BaseConfiguration):
    def __init__(self, dictionary: dict[str, Any] = None):
        self.config: list[str] = None
        self.custom_name: str = '<default>'
        self.tags: list[str] = None
        self.debug: bool = False
        self.wandb_entity: str = 'tum_daml_ba_antoniooroz'
        self.wandb_project: str = 'GR2'
        self.load: str = None
        
        self.training: TrainingConfiguration = TrainingConfiguration()
        self.experiment: ExperimentConfiguration = ExperimentConfiguration()
        self.model: ModelConfiguration = ModelConfiguration()
        
        self.from_dict(dictionary)
        
        # Static args for logging
        from pgnn.utils import get_device
        self.device = get_device()
        
        self.custom_name += f'|{self.experiment.active_learning_selector}|{self.experiment.active_learning_selector_network_mode}|{self.experiment.active_learning_selector_uncertainty_mode}|{self.experiment.active_learning_training_type}|{self.experiment.active_learning_l2_distance_use_centroids}|{self.experiment.active_learning_starting_class}'