##############################################################
# Parts of this file contain content from the following source
# Author: Johannes Gasteiger, Aleksandar Bojchevski and Stephan Günnemann
# Last Visited: 14.06.2022
# Title: PPNP and APPNP
# URL: https://github.com/gasteigerjo/ppnp
##############################################################

from pyro.infer.predictive import Predictive
import torch
import torch.nn.functional as F
from pgnn.base.p_wrapper import P_Wrapper
import pgnn.base.uncertainty_estimation as UE
import pgnn.base.network_uncertainty_combination as NUC
from pgnn.base.base import Base
from pgnn.configuration.training_configuration import Phase
from pgnn.data.data import Data
from pgnn.result.result import NetworkModeResult, Results

from pgnn.utils import get_statistics, get_device, escape_slashes
import pgnn.base.uncertainty_estimation as UE

import pyro
from pyro.nn import PyroModule
from pyro.infer import SVI, TraceMeanField_ELBO, Predictive
from pyro.infer.autoguide import AutoNormal
from pgnn.data.model_input import ModelInput
from pgnn.result import NetworkMode, ModelOutput

from pyro import poutine

import pyro.distributions as dist
import os

import copy

class P_Base(Base):
    def __init__(self):
        super().__init__()
        
        self.return_sites = ("obs", "_RETURN") + ModelOutput().pyro_return_sites()
        
        self.nclasses = None
        self.nfeatures = None
        self.fcs = None
        self.dropout = None
        self.act_fn = None
        self.model_wrapper = None

    def set_model(self, model):
        self.model_wrapper = P_Wrapper(model)
        self.pyronize(self.model_wrapper.model)

    def predict(self, model_input: ModelInput) -> dict[NetworkMode, ModelOutput]:
        nsamples = self.configuration.model.samples_prediction
        
        predictive = Predictive(
            model=self.model_wrapper, 
            guide=self.guide, 
            num_samples=nsamples,
            return_sites=self.return_sites, 
            parallel=True
        )
        pyro_result = predictive(model_input)
        
        model_outputs = {
            NetworkMode.PROPAGATED: ModelOutput(),
            NetworkMode.ISOLATED: ModelOutput()
        }
        
        for network_mode, model_output in model_outputs.items():
            # Get Model Output
            model_output.from_pyro_result(pyro_result, network_mode)
            
            all_softmax_scores = model_output.softmax_scores
            mean_softmax_scores = (torch.sum(all_softmax_scores, dim=0) / nsamples).squeeze(0)    
            max_probabilities = mean_softmax_scores.max(dim=-1)
            
            model_output.softmax_scores = mean_softmax_scores
            model_output.predicted_classes = max_probabilities.indices
            model_output.epistemic_uncertainties = UE.get_uncertainty(
                configuration=self.configuration, 
                uncertainty_metric=self.configuration.model.uncertainty_estimation, probs_all=all_softmax_scores, 
                probs_mean=mean_softmax_scores, 
                preds=max_probabilities.indices
            )

            model_output.aleatoric_uncertainties = UE.get_uncertainty(
                configuration=self.configuration, 
                uncertainty_metric='probability', probs_all=all_softmax_scores, 
                probs_mean=mean_softmax_scores, 
                preds=max_probabilities.indices
            )
                
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
        
        self.guide = AutoNormal(self.model_wrapper, init_scale=self.configuration.model.guide_init_scale)
        self.elbo = TraceMeanField_ELBO(
            num_particles=self.configuration.model.samples_training, 
            vectorize_particles=self.configuration.model.vectorize
        )
        
        self.optimizer = {
            Phase.TRAINING: pyro.optim.ClippedAdam({
                    "lr": self.configuration.training.learning_rate[Phase.TRAINING], 
                    "weight_decay": self.configuration.training.reg_lambda[Phase.TRAINING],
                    "clip_norm": self.configuration.training.optimizer_clip_norm[Phase.TRAINING],
                    "lrd": self.configuration.training.optimizer_learning_rate_decay[Phase.TRAINING]
                })
        }
        
        self.svi = {
            Phase.TRAINING: SVI(self.model_wrapper, self.guide, self.optimizer[Phase.TRAINING], loss=self.elbo)
        }

    def step(self, phase: Phase, data: Data) -> Results:
        is_training = phase in Phase.training_phases()
        self.set_eval(not is_training)

        # Loss only on ID values
        id_model_input = ModelInput(
            features=data.model_input.features,
            indices=data.model_input.indices[data.ood_indicators==0]
        )
        id_labels = data.labels[data.ood_indicators==0]

        # Loss Calculation/Step
        with poutine.scale(scale=1.0/id_labels.shape[0]):
            if is_training:
                loss = self.svi[Phase.TRAINING].step(id_model_input, id_labels)
            else:
                loss = self.svi[Phase.TRAINING].evaluate_loss(id_model_input, id_labels)  

        # ModelOutput results need to be calculated seperately
        # because Pyro doesn't support outputting the other values with SVI
        model_outputs = self.predict(data.model_input)
        results = Results()
            
        for network_mode in model_outputs.keys():
            model_output = model_outputs[network_mode]
                
            results.networkModeResults[network_mode] = NetworkModeResult(
                model_output=model_output,
                loss=loss if network_mode==NetworkMode.PROPAGATED else 0,
                data=data
            )
                
        return results

    def custom_state_dict(self):
        return {
            'model': copy.deepcopy(self.model_wrapper.state_dict()),
            'guide': copy.deepcopy(self.guide.state_dict()),
            'pyro_params': copy.deepcopy(pyro.get_param_store().get_state()),
            'config': copy.deepcopy(self.configuration.to_dict()),
            'torch_seed': copy.deepcopy(self.pytorch_seed),
            'data_seed': copy.deepcopy(self.data_seed),
            'iteration': copy.deepcopy(self.iteration),
            'model_name': copy.deepcopy(self.model_name)
        }
        
    def load_custom_state_dict(self, state_dict):
        self.init(
            pytorch_seed=state_dict["torch_seed"], 
            model_name=state_dict["model_name"],
            iteration=state_dict["iteration"], 
            data_seed=state_dict["data_seed"]
        )
        
        # Fixes loading error
        features = torch.zeros([self.adj_matrix.shape[0], self.nfeatures], device=get_device())
        indices = torch.tensor([0]).to(get_device())
        labels = torch.tensor([0]).to(get_device())     
        self.set_eval(False)
        SVI(self.model_wrapper, self.guide, pyro.optim.ClippedAdam({
                    "lr": 0.1, 
                    "weight_decay": 0.0,
                    "clip_norm": 10.0,
                    "lrd": 1.0
                }), loss=TraceMeanField_ELBO(
            num_particles=self.configuration.model.samples_training, 
            vectorize_particles=self.configuration.model.vectorize
        )).step(ModelInput(features=features, indices=indices), labels)

        pyro.clear_param_store()
        self.model_wrapper.load_state_dict(state_dict['model'])
        self.guide.load_state_dict(state_dict['guide'])
        pyro.get_param_store().set_state(state_dict['pyro_params'])
        
                
        
    def set_eval(self, eval = True):
        if eval:
            self.eval()
        else:
            self.train()
        torch.set_grad_enabled(not eval)

        self.guide.requires_grad_(not eval)
            
    def log_weights(self):
        return self.model_wrapper.model.log_weights()

    def pyronize(self, model):
        num_layers = len(model.layers)
        for i, layer in enumerate(model.layers): 
            name = self.configuration.model.type.name
            if (name[:name.find("_")] == "Mixed") and i < num_layers - 1:
                continue

            layer.pyronize_weights()