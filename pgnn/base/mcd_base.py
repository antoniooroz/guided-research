from pgnn.base import Base
import torch
from pgnn.base.network_mode import NetworkMode

from pgnn.configuration.configuration import Configuration
from pgnn.configuration.model_configuration import ModelType
from pgnn.configuration.training_configuration import Phase
from pgnn.data.data import Data
from pgnn.data.model_input import ModelInput
from pgnn.result.model_output import ModelOutput
import pgnn.base.uncertainty_estimation as UE
import pgnn.base.network_uncertainty_combination as NUC

from .base import Base

import pgnn.base.uncertainty_estimation as UE


class MCD_Base(Base):
    def __init__(self, nfeatures: int, nclasses: int, configuration: Configuration):
        super().__init__()
        self.nfeatures = nfeatures
        self.nclasses = nclasses
        self.configuration = configuration
        self.model: Base = None
    
    def forward(self, model_input: ModelInput) -> dict[NetworkMode, ModelOutput]:
        return self.model.forward(model_input)

    def predict(self, model_input: ModelInput) -> dict[NetworkMode, ModelOutput]:        
        model_output_samples: dict[NetworkMode, list[ModelOutput]] = {
            NetworkMode.ISOLATED: [],
            NetworkMode.PROPAGATED: []
        }
        model_outputs: dict[NetworkMode, ModelOutput] = {}
        
        if self.training:
            nsamples = self.configuration.model.samples_training 
        else:
            nsamples = self.configuration.model.samples_prediction
        
        # Generate Samples
        for _ in range(nsamples):
            output_per_mode = self.forward(model_input)
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
    
    def load_model(self, mode: ModelType, name: str, seed: int, iter: int):
        self.model.load_model(
            mode=ModelType.get_base_type(mode, self.configuration.model.load_mcd_from_base_model),
            name=name,
            seed=seed,
            iter=iter
        )
        
    def save_model(self, custom_state_dict = None):
        self.model.save_model(custom_state_dict)
        
    def init(self, pytorch_seed, model_name, iteration, data_seed):
        super().init(pytorch_seed=pytorch_seed, model_name=model_name, iteration=iteration, data_seed=data_seed)
        self.model.init(pytorch_seed=pytorch_seed, model_name=model_name, iteration=iteration, data_seed=data_seed)
        
    def load_custom_state_dict(self, state_dict):
        self.init(
            pytorch_seed=state_dict["torch_seed"], 
            model_name=state_dict["model_name"],
            iteration=state_dict["iteration"], 
            data_seed=state_dict["data_seed"]
        )
        self.model.load_custom_state_dict(state_dict)
        
    def custom_state_dict(self):
        return self.model.custom_state_dict()
        
    def log_weights(self):
        return self.model.log_weights()
    
    def eval(self):
        """
        Adapted from https://discuss.pytorch.org/t/using-dropout-in-evaluation-mode/27721
        Last visited: 14.06.2022
        """
        super().eval()
        # is_mcd because DropEdge also uses MCD base class but doesn't utilize normal dropout in eval
        if self.configuration.model.type in ModelType.mcds():
            for m in self.model.modules():
                if m.__class__.__name__.startswith('Dropout'):
                    m.train()