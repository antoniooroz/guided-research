##############################################################
# This file is a modified version from the following source
# Author: Johannes Gasteiger, Aleksandar Bojchevski and Stephan Günnemann
# Last Visited: 14.06.2022
# Title: PPNP and APPNP
# URL: https://github.com/gasteigerjo/ppnp
##############################################################

from dataclasses import dataclass
from typing import Any, List
import copy
from enum import Enum
import numpy as np

from torch.nn import Module
from pgnn.base.network_mode import NetworkMode
from pgnn.configuration.training_configuration import StopVariable

from pgnn.result.result import NetworkModeResult, Results

@dataclass
class SavedState:
    state_dict: dict[str, Any] = None
    value: float = None
    epoch: int = 0

class EarlyStopping:
    def __init__(self, model, stop_variable: StopVariable):
        self.model = model
        self.stop_variable = stop_variable
        self.best = SavedState()
        
    def init_for_training_phase(self, enabled: bool = True, patience: int = 100, max_epochs: int = 1000):
        self.enabled = enabled
        self.patience = patience
        self.patience_reset = patience
        self.max_epochs = max_epochs
        
    def check_stop(self, result: NetworkModeResult, epoch: int):
        if not self.enabled:
            return False
        
        currentValue = StopVariable.multiplier(self.stop_variable) * getattr(result, self.stop_variable.value)
        
        if self.best.value is None or self.best.value < currentValue:
            self.best = SavedState(state_dict=self.model.custom_state_dict(), value=currentValue, epoch=epoch)
            self.patience = self.patience_reset
        else:
            self.patience = self.patience - 1
            
        return self.patience <= 0
    
    def load_best(self):
        if self.enabled:
            self.model.load_custom_state_dict(self.best.state_dict)
        