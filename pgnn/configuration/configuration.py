from typing import Any
from pgnn.configuration.base_configuration import BaseConfiguration
from pgnn.configuration.experiment_configuration import ExperimentConfiguration

from pgnn.configuration.training_configuration import TrainingConfiguration
from pgnn.configuration.model_configuration import ModelConfiguration


class Configuration(BaseConfiguration):
    def __init__(self, dictionary: dict[str, Any] = None):
        self.config: list[str] = None
        self.custom_name: str = '<default>'
        self.load: str = None
        
        self.training: TrainingConfiguration = TrainingConfiguration()
        self.experiment: ExperimentConfiguration = ExperimentConfiguration()
        self.model: ModelConfiguration = ModelConfiguration()
        
        self.from_dict(dictionary)
        
        