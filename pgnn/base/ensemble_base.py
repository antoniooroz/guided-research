import os
from pgnn.base import Base
import torch
import torch.nn as nn
from pgnn.base.network_mode import NetworkMode

from pgnn.configuration.configuration import Configuration
from pgnn.configuration.model_configuration import ModelType
from pgnn.data.model_input import ModelInput
from pgnn.result.model_output import ModelOutput
import pgnn.base.uncertainty_estimation as UE
import pgnn.base.network_uncertainty_combination as NUC
from pgnn.utils.utils import escape_slashes

from .base import Base

import pgnn.base.uncertainty_estimation as UE


class ENSEMBLE_Base(Base):
    def __init__(self, nfeatures: int, nclasses: int, configuration: Configuration):
        super().__init__()
        self.nfeatures = nfeatures
        self.nclasses = nclasses
        self.configuration = configuration
        self.models: list[Base] = nn.ModuleList()
        
        assert configuration.model.samples_training == configuration.model.samples_prediction, "For ensembles, number of training samples and prediction sampels needs to be the same"
    
    def forward(self, model_input: ModelInput) -> dict[NetworkMode, ModelOutput]:
        raise NotImplementedError()

    def predict(self, model_input: ModelInput) -> dict[NetworkMode, ModelOutput]:        
        model_output_samples: dict[NetworkMode, list[ModelOutput]] = {
            NetworkMode.ISOLATED: [],
            NetworkMode.PROPAGATED: []
        }
        model_outputs: dict[NetworkMode, ModelOutput] = {}
        
        nsamples = self.configuration.model.samples_training        
        
        # Generate Samples
        for i in range(nsamples):
            output_per_mode = self.models[i].forward(model_input)
            for key in model_output_samples.keys():
                model_output_samples[key].append(output_per_mode[key])
        
        # Accumulate samples and calculate outputs
        for key in model_output_samples.keys():
            model_output = ModelOutput.cat_list(model_output_samples[key])
            all_softmax_scores = model_output.softmax_scores
            mean_softmax_scores = (torch.sum(all_softmax_scores, dim=0) / nsamples).squeeze(0)    
            max_probabilities = mean_softmax_scores.max(dim=-1)
            
            model_output.softmax_scores = mean_softmax_scores
            model_output.predicted_classes = max_probabilities.indices
            model_output.epistemic_uncertainties = UE.get_uncertainty(
                configuration=self.configuration, 
                uncertainty_metric=self.configuration.model.uncertainty, probs_all=all_softmax_scores, 
                probs_mean=mean_softmax_scores, 
                preds=max_probabilities.indices
            )

            model_output.aleatoric_uncertainties = UE.get_uncertainty(
                configuration=self.configuration, 
                uncertainty_metric='probability', probs_all=all_softmax_scores, 
                probs_mean=mean_softmax_scores, 
                preds=max_probabilities.indices
            )
            
            model_outputs[key] = model_output
            
                
        NUC.combine(
            combinationMethod=self.configuration.model.network_combination,
            model_outputs=model_outputs
        )
        
        return model_outputs
        
    def init(self, pytorch_seed, model_name, iteration, data_seed):
        super().init(pytorch_seed=pytorch_seed, model_name=model_name, iteration=iteration, data_seed=data_seed)
        for model in self.models:
            model.init(pytorch_seed=pytorch_seed, model_name=model_name, iteration=iteration, data_seed=data_seed)
        
    def load_custom_state_dict(self, state_dict):
        self.init(
            pytorch_seed=state_dict["torch_seed[0]"], 
            model_name=state_dict["model_name[0]"],
            iteration=state_dict["iteration[0]"], 
            data_seed=state_dict["data_seed[0]"]
        )
        for i in range(len(self.models)):
            model = self.models[i]
            model.load_custom_state_dict(self._get_state_dict_for_single_model(state_dict=state_dict, i=i))
        
    def custom_state_dict(self):
        state_dict = {}
        for i in range(len(self.models)):
            state_dict_model = self.models[i].custom_state_dict()
            for key, val in state_dict_model.items():
                state_dict[f"{key}[{i}]"] = val
        return state_dict
    
    def _get_state_dict_for_single_model(self, state_dict, i):
        state_dict_model = {}
        for key, val in state_dict.items():
            number = f"[{i}]"
            if key.endswith(number):
                state_dict_model[key.replace(number, '')] = val
        return state_dict_model
        
    def log_weights(self):
        return {}