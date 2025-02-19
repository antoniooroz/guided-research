from dataclasses import dataclass
from typing import Optional
import torch
import pyro

from pgnn.base.network_mode import NetworkMode

class ModelOutput:
    def __init__(self,
                logits: Optional[torch.Tensor] = None,
                softmax_scores: Optional[torch.Tensor] = None,
                predicted_classes: Optional[torch.Tensor] = None,
                epistemic_uncertainties: Optional[torch.Tensor] = None,
                aleatoric_uncertainties: Optional[torch.Tensor] = None):
        self.logits = logits # Filled in forward
        self.softmax_scores = softmax_scores # Filled in forward
        self.predicted_classes = predicted_classes # Filled in predict
        self.epistemic_uncertainties = epistemic_uncertainties # Filled in predict
        self.aleatoric_uncertainties = aleatoric_uncertainties # Filled in predict
    
    def __add__(self, o):
        new_output = ModelOutput()
        for key, val in self.__dict__.items():
            new_output.__dict__[key] = self._cat(val, o.__dict__[key])

        return new_output
        
    def _cat(self, own_tensor: torch.Tensor, other_tensor: torch.Tensor) -> torch.Tensor:
        if own_tensor is not None and other_tensor is not None:
            return torch.cat([own_tensor, other_tensor]).to(own_tensor.device)
        elif own_tensor is not None:
            return own_tensor
        else:
            return other_tensor
        
    def cat_list(l: list['ModelOutput']) -> 'ModelOutput':
        model_output = ModelOutput()
        for attr_name in model_output.__dict__.keys():
            vals = map(lambda x: x.__dict__[attr_name], l)
            vals = filter(lambda x: x is not None, vals)
            vals = list(map(lambda x: x.unsqueeze(0), vals))
            if vals:
                model_output.__dict__[attr_name] = torch.cat(vals).to(vals[0].device)
                
        return model_output
            
    def pyro_return_sites(self):
        l = []
        for network_mode in NetworkMode:
            for key in self.__dict__:
                l.append(f"model_output-{network_mode}-{key}")
        
        return tuple(l)
        
                
    def pyro_deterministic(self, network_mode: NetworkMode):
        """ 
        ONLY NEEDED FOR BAYESIAN MODELS
        
        call pyro.deterministic for each tensor. 

        Args:
            network_mode (NetworkMode): network_mode for which the model_output was generated
        """
        
        for key, val in self.__dict__.items():
            if val is not None:
                pyro.deterministic(
                    name=f"model_output-{network_mode}-{key}",
                    value=val
                )
                
    def from_pyro_result(self, pyro_result, network_mode: NetworkMode):
        """
        ONLY NEEDED FOR BAYESIAN MODELS
        Fills the tensors from a pyro result

        Args:
            pyro_result: Pyro Result
            network_mode (NetworkMode): network_mode for which the model_output is needed
        """
        
        for key in self.__dict__:
            pyro_key = f"model_output-{network_mode}-{key}"
            if pyro_key in pyro_result:
                self.__dict__[key] = pyro_result[pyro_key]
                
    def cpu(self):
        for key in self.__dict__:
            if isinstance(self.__dict__[key], torch.Tensor):
                self.__dict__[key] = self.__dict__[key].cpu().numpy()
                
        return self
                

class GPNModelOutput(ModelOutput):
    def __init__(self,
                logits: Optional[torch.Tensor] = None,
                softmax_scores: Optional[torch.Tensor] = None,
                predicted_classes: Optional[torch.Tensor] = None,
                epistemic_uncertainties: Optional[torch.Tensor] = None,
                aleatoric_uncertainties: Optional[torch.Tensor] = None,
                alpha: Optional[torch.Tensor] = None):
        super().__init__(logits, softmax_scores, predicted_classes, epistemic_uncertainties, aleatoric_uncertainties)
        self.alpha = alpha
    