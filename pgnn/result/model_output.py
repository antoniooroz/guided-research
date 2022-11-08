from dataclasses import dataclass
from typing import Optional
import torch
import pyro

from pgnn.base.network_mode import NetworkMode

@dataclass
class ModelOutput:
    logits: Optional[torch.Tensor] = None # Filled in forward
    softmax_scores: Optional[torch.Tensor] = None # Filled in forward
    predicted_classes: Optional[torch.Tensor] = None # Filled in predict
    epistemic_uncertainties: Optional[torch.Tensor] = None # Filled in predict
    aleatoric_uncertainties: Optional[torch.Tensor] = None # Filled in predict
    
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
            vals = list(filter(lambda x: x is not None, map(lambda x: x.__dict__[attr_name], l)))
            if vals:
                model_output[attr_name] = torch.cat(vals).to(vals[0].device)
            
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
                
@dataclass   
class GPNModelOutput(ModelOutput):
    alpha: Optional[torch.Tensor] = None