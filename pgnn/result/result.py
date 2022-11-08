from dataclasses import dataclass, asdict
from typing import Any, Optional
from pgnn.base.network_mode import NetworkMode

from pgnn.data.data import Data
from .model_output import ModelOutput

import torch
from sklearn.metrics import roc_auc_score

class NetworkModeResult:
    
    def __init__(self, 
                 model_output: ModelOutput = None, 
                 loss: float = 0.0, 
                 loss_unnormalized = None,
                 data: Data = None):
        self.model_output: ModelOutput = model_output if model_output else ModelOutput()
        
        if model_output and model_output.predicted_classes is not None and data:
            correct_datapoint_indicator = (model_output.predicted_classes==data.labels)
            self.datapoints_correct: int = correct_datapoint_indicator[data.ood_indicators==0].sum().item()
            self.datapoints_false: int = (~correct_datapoint_indicator)[data.ood_indicators==0].sum().item()
            
            if data.ood_indicators.sum().item() > 0:
                # sum of empty tensor is buggy on MPS, therefore check if tensors will be empty
                self.ood_datapoints_correct: int = correct_datapoint_indicator[data.ood_indicators==1].sum().item()
                self.ood_datapoints_false: int = (~correct_datapoint_indicator)[data.ood_indicators==1].sum().item()
            else:
                self.ood_datapoints_correct: int = 0
                self.ood_datapoints_false: int = 0
        else:
            self.datapoints_correct: int = 0
            self.datapoints_false: int = 0
            
            self.ood_datapoints_correct: int = 0
            self.ood_datapoints_false: int = 0
        
        if loss_unnormalized:
            self._loss_unnormalized = loss_unnormalized
        else:
            self._loss_unnormalized: float = loss * self.datapoints
        
        self.ood_indicators: torch.Tensor = data.ood_indicators if data else None
        
        self._ood_auc_roc_aleatoric: float = None
        self._ood_auc_roc_epistemic: float = None

    @property
    def datapoints(self) -> int:
        return self.datapoints_correct + self.datapoints_false
        
    @property
    def loss(self) -> Optional[float]:
        if not self.datapoints: 
            return None
        return self._loss_unnormalized / self.datapoints
    
    @property
    def accuracy(self) -> Optional[float]:
        if not self.datapoints: 
            return None
        return self.datapoints_correct / self.datapoints
    
    @property
    def ood_datapoints(self) -> int:
        return self.ood_datapoints_correct + self.ood_datapoints_false
    
    @property
    def ood_accuracy(self) -> Optional[float]:
        if not self.ood_datapoints: 
            return None
        return self.ood_datapoints_correct / self.ood_datapoints
    
    @property
    def ood_auc_roc_aleatoric(self) -> Optional[float]:
        if not self.ood_datapoints:
            return None
        
        if self._ood_auc_roc_aleatoric is None and self.model_output.aleatoric_uncertainties is not None:
            self._ood_auc_roc_aleatoric = self._roc_auc_score(self.ood_indicators, self.model_output.aleatoric_uncertainties)
        return self._ood_auc_roc_aleatoric
    
    @property
    def ood_auc_roc_epistemic(self) -> Optional[float]:
        if not self.ood_datapoints:
            return None
        
        if self._ood_auc_roc_epistemic is None and self.model_output.epistemic_uncertainties is not None:
            self._ood_auc_roc_epistemic = self._roc_auc_score(self.ood_indicators, self.model_output.epistemic_uncertainties)
        return self._ood_auc_roc_epistemic
    
    def __add__(self, o):
        result: NetworkModeResult = NetworkModeResult(
            model_output=self.model_output + o.model_output,
            loss_unnormalized=self._loss_unnormalized + o._loss_unnormalized,
        )
        result.datapoints_correct = self.datapoints_correct + o.datapoints_correct
        result.datapoints_false = self.datapoints_false + o.datapoints_false
        
        result.ood_datapoints_correct = self.ood_datapoints_correct + o.ood_datapoints_correct
        result.ood_datapoints_false = self.ood_datapoints_false + o.ood_datapoints_false
        
        result.ood_indicators = torch.cat([self.ood_indicators, o.ood_indicators]).to(self.ood_indicators.device)
        
        return result
    
    def _roc_auc_score(self, ood_indicators: torch.Tensor, epistemic_uncertainties: torch.Tensor):
        ood_indicators = ood_indicators.to('cpu')
        epistemic_uncertainties = epistemic_uncertainties.to('cpu')
        
        return roc_auc_score(ood_indicators, epistemic_uncertainties)
    
    def to_dict(self, prefix: str ='') -> dict[str, Any]:
        result = {}
        
        for prop in dir(self):
            if not prop.startswith('_') and prop != 'model_output':
                if not isinstance(getattr(self, prop), torch.Tensor):
                    result[f"{prefix}{prop}"] = getattr(self, prop)
    
        return result
    
@dataclass 
class Info():
    duration: float = 0.0
    seed: int = 0
    iteration: int = 0
    
    def __add__(self, o):
        return Info(
            duration=self.duration + o.duration,
            seed=self.seed,
            iteration=self.iteration
        )
        
    def to_dict(self, prefix: str ='') -> dict[str, Any]:
        result = {}
        for key, val in asdict(self).items():
            result[f"{prefix}{key}"] = val
        
        return result

@dataclass
class Results():
    
    def __init__(self, networkModeResults: dict[NetworkMode, NetworkModeResult] = None, info: Info = None):
        if networkModeResults:
            self.networkModeResults = networkModeResults
        else:
            self.networkModeResults: dict[NetworkMode, NetworkModeResult] = {}
            
        if info:
            self.info = info
        else: 
            self.info = Info()

    
    def __add__(self, o):
        newNetworkModeResults: dict[NetworkMode, NetworkModeResult] = {}
        for network_mode in set(self.networkModeResults.keys()).union(set(o.networkModeResults.keys())):
            selfVal = self.networkModeResults.get(network_mode, None)
            oVal = o.networkModeResults.get(network_mode, None)
            
            if selfVal and oVal:
                newNetworkModeResults[network_mode] = selfVal + oVal
            elif selfVal:
                newNetworkModeResults[network_mode] = selfVal
            else:
                newNetworkModeResults[network_mode] = oVal
        
        return Results(
            networkModeResults=newNetworkModeResults,
            info=self.info + o.info
        )
        
    def to_dict(self, prefix: str ='') -> dict[str, Any]:
        result = {}
        for key, val in self.networkModeResults.items():
            result.update(val.to_dict(f"{prefix}{key.name}/"))
        
        result.update(self.info.to_dict(f"{prefix}info/"))
        
        return result