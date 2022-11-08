from enum import Enum
from typing import Any
from pgnn.configuration.base_configuration import BaseConfiguration

class TrainingConfiguration(BaseConfiguration):
    def __init__(self, dictionary: dict[str, Any] = None):
        self.learning_rate: dict[Phase, float] = {Phase.TRAINING: 0.1}
        self.reg_lambda: dict[Phase, float] = {Phase.TRAINING: 0.0}
        self.early_stopping_variable: StopVariable = StopVariable.LOSS
        self.early_stopping: dict[Phase, bool] = {Phase.TRAINING: True}
        self.skip_training: bool = False
        self.max_epochs: dict[Phase, int] = {Phase.TRAINING: 1000}
        self.patience: dict[Phase, int] = {Phase.TRAINING: 100}
        self.drop_prob: float = 0.0
        self.train_without_network_effects: bool = False
        self.optimizer: OptimizerType = OptimizerType.CLIPPED_ADAM  # Not for GPN
        self.optimizer_clip_norm: dict[Phase, float] = {Phase.TRAINING: 10.0} # CLIPPED_ADAM
        self.optimizer_learning_rate_decay: dict[Phase, float] = {Phase.TRAINING: 1.0} # CLIPPED_ADAM
        self.phases: list[Phase] = [Phase.TRAINING]
        self.wandb_logging_during_training: bool = True
        
        # OOD
        self.ood_eval_during_training: bool = False
        
        # Edge Dropout
        self.edge_drop_prob: float = 0.0
        
        self.from_dict(dictionary)
        
class OptimizerType(Enum):
    CLIPPED_ADAM = 'clipped_adam'
    ADAM = 'adam'    
class Phase(Enum):
    # TRAINING
    WARMUP = 'warmup'
    TRAINING = 'training'
    FINETUNING = 'finetuning'
    
    def training_phases() -> list['Phase']:
        return [Phase.WARMUP, Phase.TRAINING, Phase.FINETUNING]
    
    INIT = 'init'
    STOPPING = 'stopping'
    VALTEST = 'valtest'
    
    def get_phases(training_phase: 'Phase') -> list['Phase']:
        return [training_phase, Phase.STOPPING]
    
class StopVariable(Enum):
    LOSS = 'loss'
    ACCURACY = 'accuracy'
    
    def multiplier(stop_variable) -> int:
        if stop_variable == StopVariable.LOSS:
            return -1
        elif stop_variable == StopVariable.ACCURACY:
            return 1
        else:
            raise NotImplementedError()