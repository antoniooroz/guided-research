from enum import Enum
import torch
from pgnn.base.network_mode import NetworkMode

from pgnn.configuration.configuration import Configuration
from pgnn.configuration.model_configuration import NucType
from pgnn.result.model_output import ModelOutput

nuc_functions = {}

def nuc(func):
    """
        Adapted from https://stackoverflow.com/a/54075852/17134758
        Last visited: 06.07.2022
    """
    nuc_functions[func.__name__] = func
    return func

def combine(combinationMethod: NucType, model_outputs: dict[NetworkMode, ModelOutput]) -> None:
        combined_model_output = ModelOutput()
        
        for uncertainty_name in ['aleatoric_uncertainties', 'epistemic_uncertainties']:
            if model_outputs[NetworkMode.PROPAGATED].__dict__[uncertainty_name] is not None:
                combined_model_output.__dict__[uncertainty_name] = combine_tensors(
                    combinationMethod=combinationMethod,
                    tensor_isolated=model_outputs[NetworkMode.ISOLATED].__dict__[uncertainty_name],
                    tensor_propagated=model_outputs[NetworkMode.PROPAGATED].__dict__[uncertainty_name]
                )   
        
        model_outputs[NetworkMode.COMBINED] = combined_model_output

def combine_tensors(combinationMethod: NucType,  
            tensor_isolated: torch.Tensor,
            tensor_propagated: torch.Tensor) -> torch.Tensor:
    return nuc_functions[combinationMethod.value](tensor_isolated, tensor_propagated)

def _cat(tensor_isolated: torch.Tensor, tensor_propagated: torch.Tensor) -> torch.Tensor:
    return torch.cat([tensor_isolated.unsqueeze(0), tensor_propagated.unsqueeze(0)]).to(tensor_isolated.device)

@nuc
def max(tensor_isolated: torch.Tensor, tensor_propagated: torch.Tensor) -> torch.Tensor:
    c = _cat(tensor_isolated, tensor_propagated)
    return c.max(dim=0).values

@nuc
def min(tensor_isolated: torch.Tensor, tensor_propagated: torch.Tensor) -> torch.Tensor:
    c = _cat(tensor_isolated, tensor_propagated)
    return c.min(dim=0).values

@nuc
def mean(tensor_isolated: torch.Tensor, tensor_propagated: torch.Tensor) -> torch.Tensor:
    c = _cat(tensor_isolated, tensor_propagated)
    return c.mean(dim=0)


    
