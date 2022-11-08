##############################################################
# Parts of this file contain content from the following source
# Author: Johannes Gasteiger, Aleksandar Bojchevski and Stephan Günnemann
# Last Visited: 14.06.2022
# Title: PPNP and APPNP
# URL: https://github.com/gasteigerjo/ppnp
##############################################################

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from pgnn.base.network_mode import NetworkMode
from pgnn.configuration.training_configuration import Phase
from pgnn.logger.log_weight import LogWeight
from pgnn.result import ModelOutput, NetworkModeResult, Results

from pgnn.utils import get_device, escape_slashes
import pgnn.base.uncertainty_estimation as UE
import pgnn.base.network_uncertainty_combination as NUC
from pyro.nn import PyroModule
from pgnn.data import Data, ModelInput
from pgnn.configuration import *

import copy
import os

class Base(PyroModule):
    SAVE_FILE_NAME: str = '{dir}/saved_models/{mode} [{name}] [{seed}] [{iter}].save'
    
    def __init__(self):
        super().__init__()
        self.nfeatures: int = None
        self.nclasses: int = None
        self.configuration: Configuration = None
        self.fcs = None
        self.reg_params = None
        self.dropout = None
        self.act_fn = None
    
    def forward(self, model_input: ModelInput) -> dict[NetworkMode, ModelOutput]:
        # Should return propagated and isolated model outputs where only softmax_scores is filled
        raise NotImplementedError()
    
    def predict(self, model_input: ModelInput) -> dict[NetworkMode, ModelOutput]:
        model_outputs: dict[NetworkMode, ModelOutput] = self.forward(model_input)
        
        for model_output in model_outputs.values():
            max_probabilities = model_output.softmax_scores.max(-1)
            model_output.predicted_classes = max_probabilities.indices
            model_output.aleatoric_uncertainties = UE.entropy({
                "probs_mean": model_output.softmax_scores
            })
        
        NUC.combine(
            combinationMethod=self.configuration.model.network_combination,
            model_outputs=model_outputs
        )
        
        return model_outputs
    
    def init(self, pytorch_seed, model_name, iteration, data_seed):
        torch.manual_seed(seed=pytorch_seed)
        self.pytorch_seed = pytorch_seed
        self.model_name = model_name
        self.iteration = iteration
        self.data_seed = data_seed
        
        self.optimizer = {
            Phase.TRAINING: torch.optim.Adam(
                    self.parameters(), 
                    lr=self.configuration.training.learning_rate[Phase.TRAINING], 
                    weight_decay=self.configuration.training.reg_lambda[Phase.TRAINING]
                )
        }
    
    def step(self, phase: Phase, data: Data) -> Results:
        is_training = phase in Phase.training_phases()
        self.set_eval(not is_training)
        
        if is_training:
            self.optimizer[phase].zero_grad()
        
        with torch.set_grad_enabled(is_training):
            model_outputs = self.predict(data.model_input)
            results = Results()
                
            for network_mode in model_outputs.keys():
                model_output = model_outputs[network_mode]
                loss = 0
                
                if model_output.softmax_scores is not None:
                    # Calculate loss
                    loss = F.nll_loss(
                        input=torch.log(model_output.softmax_scores[data.ood_indicators==0]), 
                        target=data.labels[data.ood_indicators==0]
                    )

                    if is_training and network_mode==NetworkMode.PROPAGATED:
                        loss.backward()
                        self.optimizer[phase].step()
                        
                    loss = loss.item()
                    
                results.networkModeResults[network_mode] = NetworkModeResult(
                    model_output=model_output,
                    loss=loss,
                    data=data
                )
                
        return results
    
    def custom_state_dict(self):
        return {
            'model': copy.deepcopy(self.state_dict()),
            'config': copy.deepcopy(self.configuration.to_dict()),
            'torch_seed': copy.deepcopy(self.pytorch_seed),
            'data_seed': copy.deepcopy(self.data_seed),
            'iteration': copy.deepcopy(self.iteration),
            'model_name': copy.deepcopy(self.model_name)
        }
           
    def save_model(self, custom_state_dict = None):
        if custom_state_dict is None:
            custom_state_dict = self.custom_state_dict()
        torch.save(custom_state_dict, escape_slashes(
            Base.SAVE_FILE_NAME.format(
                dir=os.getcwd(),
                mode=self.configuration.model.type.name,
                name=self.configuration.custom_name,
                seed=self.data_seed,
                iter=self.iteration
            )))
        
    def load_custom_state_dict(self, state_dict):
        self.init(
            pytorch_seed=state_dict["torch_seed"], 
            model_name=state_dict["model_name"],
            iteration=state_dict["iteration"], 
            data_seed=state_dict["data_seed"]
        )
        self.load_state_dict(state_dict['model'])
        
    def load_model(self, mode: ModelType, name: str, seed: int, iter: int):
        state_dict = torch.load(escape_slashes(
            Base.SAVE_FILE_NAME.format(
                dir=os.getcwd(),
                mode=mode.value,
                name=name,
                seed=seed,
                iter=iter
            )), map_location=get_device())
        self.load_custom_state_dict(state_dict)
        
    def set_eval(self, eval = True):
        if eval:
            self.eval()
        else:
            self.train()
        torch.set_grad_enabled(not eval)
            
    def log_weights(self):
        log = {}
        for layer in self.layers:
            logWeights: list[LogWeight] = layer.log_weights()
            for logWeight in logWeights:
                log[logWeight.name] = logWeight
        return log