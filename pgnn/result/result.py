from collections import Counter, OrderedDict
from dataclasses import dataclass, asdict
from typing import Any, Optional
from pgnn.base.network_mode import NetworkMode

from pgnn.data.data import Data
from .model_output import ModelOutput

import torch
from sklearn.metrics import roc_auc_score

import numpy as np

class NetworkModeResult:
    def __init__(self, 
                 model_output: ModelOutput = None, 
                 loss: float = 0.0, 
                 loss_unnormalized = None,
                 data: Data = None,
                 configuration = None):
        
        # Necessary Model Outputs, don't want to leave them on GPU        
        predicted_classes = model_output.predicted_classes.cpu().detach().numpy() if model_output and model_output.predicted_classes is not None else None
        self._epistemic_uncertainties =  model_output.epistemic_uncertainties.cpu().detach().numpy() if model_output and model_output.epistemic_uncertainties is not None else None
        self._aleatoric_uncertainties =  model_output.aleatoric_uncertainties.cpu().detach().numpy() if model_output else None
        
        labels = data.labels.cpu().detach().numpy() if data else None
        ood_indicators = data.ood_indicators.cpu().detach().numpy() if data else None
        
        self.datapoints_correct_per_class = []
        self.datapoints_false_per_class = []
        
        self.ood_datapoints_correct_per_class = []
        self.ood_datapoints_false_per_class = []
        
        if predicted_classes is not None and data:
            correct_datapoint_indicator = (predicted_classes==labels)
            
            self.datapoints_correct: int = correct_datapoint_indicator[ood_indicators==0].sum().item()
            self.datapoints_false: int = (~correct_datapoint_indicator)[ood_indicators==0].sum().item()
            
            self.ood_datapoints_correct: int = correct_datapoint_indicator[ood_indicators==1].sum().item()
            self.ood_datapoints_false: int = (~correct_datapoint_indicator)[ood_indicators==1].sum().item()
            
            for c in range(configuration.experiment.num_classes):
                in_class = labels==c
                correct_indicator_per_class = in_class * correct_datapoint_indicator
                false_indicator_per_class = in_class * (~correct_datapoint_indicator)
                
                self.datapoints_correct_per_class.append(correct_indicator_per_class[ood_indicators==0].sum().item())
                self.datapoints_false_per_class.append(false_indicator_per_class[ood_indicators==0].sum().item())
                
                self.ood_datapoints_correct_per_class.append(correct_indicator_per_class[ood_indicators==1].sum().item())
                self.ood_datapoints_false_per_class.append(false_indicator_per_class[ood_indicators==1].sum().item())
        else:
            self.datapoints_correct: int = 0
            self.datapoints_false: int = 0
            
            self.ood_datapoints_correct: int = 0
            self.ood_datapoints_false: int = 0
            
            for c in range(configuration.experiment.num_classes):
                self.datapoints_correct_per_class.append(0)
                self.datapoints_false_per_class.append(0)
                
                self.ood_datapoints_correct_per_class.append(0)
                self.ood_datapoints_false_per_class.append(0)
        
        if loss_unnormalized:
            self._loss_unnormalized = loss_unnormalized
        else:
            self._loss_unnormalized: float = loss * self.datapoints
        
        self._ood_indicators = ood_indicators
        
        self._ood_auc_roc_aleatoric: float = None
        self._ood_auc_roc_epistemic: float = None
        
        self._datapoints_per_class = None
        self._ood_datapoints_per_class = None

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
        
        if self._ood_auc_roc_aleatoric is None and self._aleatoric_uncertainties is not None:
            self._ood_auc_roc_aleatoric = self._roc_auc_score(self._ood_indicators, self._aleatoric_uncertainties)
        return self._ood_auc_roc_aleatoric
    
    @property
    def ood_auc_roc_epistemic(self) -> Optional[float]:
        if not self.ood_datapoints:
            return None
        
        if self._ood_auc_roc_epistemic is None and self._epistemic_uncertainties is not None:
            self._ood_auc_roc_epistemic = self._roc_auc_score(self._ood_indicators, self._epistemic_uncertainties)
        return self._ood_auc_roc_epistemic
    
    @property
    def datapoints_per_class(self) -> list[float]:
        if self._datapoints_per_class is not None:
            return self._datapoints_per_class
        
        datapoints_per_class = []
        for corrects, wrongs in zip(self.datapoints_correct_per_class, self.datapoints_false_per_class):
            datapoints_per_class.append(corrects + wrongs)
        
        self._datapoints_per_class = datapoints_per_class    
        return self._datapoints_per_class
    
    @property
    def ood_datapoints_per_class(self) -> list[float]:
        if self._ood_datapoints_per_class is not None:
            return self._ood_datapoints_per_class
        
        ood_datapoints_per_class = []
        for corrects, wrongs in zip(self.ood_datapoints_correct_per_class, self.ood_datapoints_false_per_class):
            ood_datapoints_per_class.append(corrects + wrongs)
            
        self._ood_datapoints_per_class = ood_datapoints_per_class
        return self._ood_datapoints_per_class
    
    @property
    def ood_accuracy_per_class(self) -> list[float]:
        accuracy_per_class = []
        for corrects, datapoints in zip(self.ood_datapoints_correct_per_class, self.ood_datapoints_per_class):
            accuracy_per_class.append(corrects / (datapoints + 1e-9))
        
        return accuracy_per_class
    
    @property
    def accuracy_per_class(self) -> list[float]:
        accuracy_per_class = []
        for corrects, datapoints in zip(self.datapoints_correct_per_class, self.datapoints_per_class):
            accuracy_per_class.append(corrects / (datapoints + 1e-9))
            
        return accuracy_per_class
    
    def __add__(self, o: 'NetworkModeResult'):
        result: NetworkModeResult = NetworkModeResult(
            model_output=None,
            loss_unnormalized=self._loss_unnormalized + o._loss_unnormalized
        )
        result._aleatoric_uncertainties = np.concatenate([self._aleatoric_uncertainties, o._aleatoric_uncertainties])
        result._epistemic_uncertainties = np.concatenate([self._epistemic_uncertainties, o._epistemic_uncertainties])
        
        result.datapoints_correct = self.datapoints_correct + o.datapoints_correct
        result.datapoints_false = self.datapoints_false + o.datapoints_false
        
        result.ood_datapoints_correct = self.ood_datapoints_correct + o.ood_datapoints_correct
        result.ood_datapoints_false = self.ood_datapoints_false + o.ood_datapoints_false
        
        result._ood_indicators = np.concatenate([self._ood_indicators, o._ood_indicators])
        
        result.datapoints_correct_per_class = list(map(lambda x: x[0]+x[1], self.datapoints_correct_per_class, o.datapoints_correct_per_class))
        result.datapoints_false_per_class = list(map(lambda x: x[0]+x[1], self.datapoints_false_per_class, o.datapoints_false_per_class))
        
        result.ood_datapoints_correct_per_class = list(map(lambda x: x[0]+x[1], self.ood_datapoints_correct_per_class, o.ood_datapoints_correct_per_class))
        result.ood_datapoints_false_per_class = list(map(lambda x: x[0]+x[1], self.ood_datapoints_false_per_class, o.ood_datapoints_false_per_class))
        
        return result
    
    def _roc_auc_score(self, ood_indicators: torch.Tensor, uncertainties: torch.Tensor):        
        return roc_auc_score(ood_indicators, uncertainties)
    
    def to_dict(self, prefix: str ='') -> dict[str, Any]:
        result = {}
        
        for prop in dir(self):
            if not prop.startswith('_') and prop != 'model_output':
                prop_val = getattr(self, prop)
                if isinstance(prop_val, dict):
                    for key, val in prop_val.items():
                        result[f"{prefix}{prop}_{key}"] = val
                elif isinstance(prop_val, list):
                    for index, val in enumerate(prop_val):
                        result[f"{prefix}{prop}_{index}"] = val
                else:
                    result[f"{prefix}{prop}"] = prop_val
    
        return result
    
@dataclass 
class Info():
    duration: float = 0.0
    seed: int = 0
    iteration: int = 0
    number_of_nodes: int = 0
    mean_l2_distance_in: float = None
    mean_l2_distance_out: float = None
    active_learning_added_nodes: int = 0
    
    def __add__(self, o):
        return Info(
            duration=self.duration + o.duration,
            seed=self.seed,
            iteration=self.iteration,
            number_of_nodes=self.number_of_nodes+o.number_of_nodes,
            mean_l2_distance_in = self.mean_l2_distance_in,
            mean_l2_distance_out = self.mean_l2_distance_out
        )
        
    def to_dict(self, prefix: str ='') -> dict[str, Any]:
        result = {}
        for key, val in asdict(self).items():
            result[f"{prefix}{key}"] = val
        
        return result

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