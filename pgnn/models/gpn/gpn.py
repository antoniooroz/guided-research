##############################################################
# This file is a modified version from the following source
# Author: Maximilian Stadler, Bertrand Charpentier, Simon Geisler, Daniel Zügner and Stephan Günnemann
# Last Visited: 14.06.2022
# Title: Graph Posterior Network
# URL: https://github.com/stadlmax/Graph-Posterior-Network
##############################################################

import datetime
from typing import Dict, Tuple, List
from pgnn.base.network_mode import NetworkMode
import pgnn.base.network_uncertainty_combination as NUC
from pgnn.base.base import Base
from pgnn.configuration.model_configuration import ModelType

from pgnn.configuration.training_configuration import Phase
from pgnn.data.model_input import ModelInput
from pgnn.result.model_output import GPNModelOutput, ModelOutput
from pgnn.result.result import NetworkModeResult, Results
from .gpn_fns.utils.config import ModelConfiguration
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric.utils as tu
from pgnn.data.data import Data
from .gpn_fns.nn import uce_loss, entropy_reg
from .gpn_fns.layers import APPNPPropagation, LinearSequentialLayer
from .gpn_fns.utils import apply_mask
from pgnn.configuration import Configuration
from .gpn_fns.utils import Prediction
from .gpn_fns.layers import Density, Evidence
from .gpn_fns.model import Model
import numpy as np

from ...utils.utils import get_statistics, get_device, get_edge_indices, escape_slashes
import os
import copy


class GPN(Model):
    """Graph Posterior Network model"""

    def __init__(self, nfeatures, nclasses, configuration: Configuration, adj_matrix, training_labels):
        configuration.model.gpn_model['dim_features'] = nfeatures
        configuration.model.gpn_model['num_classes'] = nclasses
        super().__init__(ModelConfiguration(**configuration.model.gpn_model))

        if self.params.num_layers is None:
            num_layers = 0

        else:
            num_layers = self.params.num_layers

        if num_layers > 2:
            self.input_encoder = LinearSequentialLayer(
                self.params.dim_features,
                [self.params.dim_hidden] * (num_layers - 2),
                self.params.dim_hidden,
                batch_norm=self.params.batch_norm,
                dropout_prob=self.params.dropout_prob,
                activation_in_all_layers=True)
        else:
            self.input_encoder = nn.Sequential(
                nn.Linear(self.params.dim_features, self.params.dim_hidden),
                nn.ReLU(),
                nn.Dropout(p=self.params.dropout_prob))

        self.latent_encoder = nn.Linear(self.params.dim_hidden, self.params.dim_latent)

        use_batched = True if self.params.use_batched_flow else False 
        self.flow = Density(
            dim_latent=self.params.dim_latent,
            num_mixture_elements=self.params.num_classes,
            radial_layers=self.params.radial_layers,
            maf_layers=self.params.maf_layers,
            gaussian_layers=self.params.gaussian_layers,
            use_batched_flow=use_batched)

        self.evidence = Evidence(scale=self.params.alpha_evidence_scale)

        self.propagation = APPNPPropagation(
            K=self.params.K,
            alpha=self.params.alpha_teleport,
            add_self_loops=self.params.add_self_loops,
            cached=False,
            normalization='sym')
        
        self.edge_indices, self.edge_weights = get_edge_indices(adj_matrix)
        self.training_labels = training_labels
        self.configuration: Configuration = configuration

        assert self.params.pre_train_mode in ('encoder', 'flow', None)
        assert self.params.likelihood_type in ('UCE', 'nll_train', 'nll_train_and_val', 'nll_consistency', None)

    def forward(self, model_input: ModelInput) -> dict[NetworkMode, GPNModelOutput]:
        return self.forward_impl(model_input)

    def forward_impl(self, model_input: ModelInput) -> dict[NetworkMode, GPNModelOutput]:
        edge_index = self.edge_indices
        h = self.input_encoder(model_input.features)
        z = self.latent_encoder(h)

        # compute feature evidence (with Normalizing Flows)
        # log p(z, c) = log p(z | c) p(c)
        p_c = self.get_class_probabilities(model_input.features, model_input.indices)
        log_q_ft_per_class = self.flow(z) + p_c.view(1, -1).log()

        if '-plus-classes' in self.params.alpha_evidence_scale:
            further_scale = self.params.num_classes
        else:
            further_scale = 1.0

        beta_ft = self.evidence(
            log_q_ft_per_class, dim=self.params.dim_latent,
            further_scale=further_scale).exp()

        alpha_features = 1.0 + beta_ft

        beta = self.propagation(beta_ft, edge_index)
        alpha = 1.0 + beta

        soft_iso = alpha_features / alpha_features.sum(-1, keepdim=True)
        soft = alpha / alpha.sum(-1, keepdim=True)
        logits = None
        
        log_soft = soft.log()
        
        max_soft_iso, hard_iso = soft_iso.max(dim=-1)
        max_soft, hard = soft.max(dim=-1)

        # ---------------------------------------------------------------------------------
        pred = Prediction(
            # predictions and intermediary scores
            alpha=alpha,
            soft=soft,
            log_soft=log_soft,
            hard=hard,

            logits=logits,
            latent=z,
            latent_features=z,

            hidden=h,
            hidden_features=h,

            evidence=beta.sum(-1),
            evidence_ft=beta_ft.sum(-1),
            log_ft_per_class=log_q_ft_per_class,

            # prediction confidence scores
            prediction_confidence_aleatoric=max_soft,
            prediction_confidence_epistemic=alpha[torch.arange(hard.size(0)), hard],
            prediction_confidence_structure=None,

            # sample confidence scores
            sample_confidence_aleatoric=max_soft,
            sample_confidence_aleatoric_isolated=max_soft_iso,
            sample_confidence_epistemic=alpha.sum(-1),
            sample_confidence_features=alpha_features.sum(-1),
            sample_confidence_structure=None
        )
        # ---------------------------------------------------------------------------------

        return {
            NetworkMode.PROPAGATED: GPNModelOutput(
                alpha=alpha[model_input.indices],
                softmax_scores=soft[model_input.indices],
                predicted_classes=hard[model_input.indices],
                epistemic_uncertainties=pred.sample_confidence_epistemic[model_input.indices],
                aleatoric_uncertainties=pred.sample_confidence_aleatoric[model_input.indices]
            ),
            NetworkMode.ISOLATED: GPNModelOutput(
                alpha=alpha_features[model_input.indices],
                softmax_scores=soft_iso[model_input.indices],
                predicted_classes=hard_iso[model_input.indices],
                epistemic_uncertainties=pred.sample_confidence_features[model_input.indices],
                aleatoric_uncertainties=pred.sample_confidence_aleatoric_isolated[model_input.indices]
            )
        }

    def get_optimizer(self, lr: float, weight_decay: float) -> List[optim.Adam]:
        flow_lr = lr if self.params.factor_flow_lr is None else self.params.factor_flow_lr * lr
        flow_weight_decay = weight_decay if self.params.flow_weight_decay is None else self.params.flow_weight_decay

        flow_params = list(self.flow.named_parameters())
        flow_param_names = [f'flow.{p[0]}' for p in flow_params]
        flow_param_weights = [p[1] for p in flow_params]

        all_params = list(self.named_parameters())
        params = [p[1] for p in all_params if p[0] not in flow_param_names]

        # all params except for flow
        flow_optimizer = optim.Adam(flow_param_weights, lr=flow_lr, weight_decay=flow_weight_decay)
        model_optimizer = optim.Adam(
            [{'params': flow_param_weights, 'lr': flow_lr, 'weight_decay': flow_weight_decay},
             {'params': params}],
            lr=lr, weight_decay=weight_decay)

        return [model_optimizer, flow_optimizer]

    def get_warmup_optimizer(self, lr: float, weight_decay: float) -> List[optim.Adam]:
        model_optimizer, flow_optimizer = self.get_optimizer(lr, weight_decay)

        if self.params.pre_train_mode == 'encoder':
            warmup_optimizer = model_optimizer
        else:
            warmup_optimizer = flow_optimizer

        return [warmup_optimizer]

    def get_finetune_optimizer(self, lr: float, weight_decay: float) -> List[optim.Adam]:
        # similar to warmup
        return [self.get_warmup_optimizer(lr, weight_decay)]

    def uce_loss(self, model_output: GPNModelOutput, data: Data, approximate=True) -> Tuple[torch.Tensor, torch.Tensor]:
        #alpha_train, y = apply_mask(idx, labels, prediction.alpha, split='train')
        alpha_train = model_output.alpha
        reg = self.params.entropy_reg
        return uce_loss(alpha_train, data.labels, reduction='sum'), \
            entropy_reg(alpha_train, reg, approximate=approximate, reduction='sum')

    def loss(self, model_output: GPNModelOutput, data: Data) -> Dict[str, torch.Tensor]:
        uce, reg = self.uce_loss(model_output, data)
        n_train = data.model_input.indices.shape[0] if self.params.loss_reduction == 'mean' else 1
        return {'UCE': uce / n_train, 'REG': reg / n_train}

    def warmup_loss(self, model_output: GPNModelOutput, data: Data) -> Dict[str, torch.Tensor]:
        if self.params.pre_train_mode == 'encoder':
            return self.CE_loss(model_output, data)

        return self.loss(model_output, data)
    
    def CE_loss(self, model_output: GPNModelOutput, data: Data, reduction='mean') -> Dict[str, torch.Tensor]:
        y_hat = model_output.softmax_scores.log()

        return {
            'CE': F.nll_loss(y_hat, data.labels, reduction=reduction)
        }

    def finetune_loss(self, model_output: GPNModelOutput, data: Data) -> Dict[str, torch.Tensor]:
        return self.warmup_loss(model_output, data)

    def likelihood(self, model_output: GPNModelOutput, data: Data) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def get_class_probabilities(self, attr_matrix: torch.sparse.FloatTensor, idx: torch.LongTensor) -> torch.Tensor:
        l_c = torch.zeros(self.params.num_classes, device=attr_matrix.device)
        y_train = self.training_labels

        # calculate class_counts L(c)
        for c in range(self.params.num_classes):
            class_count = (y_train == c).sum()
            l_c[c] = class_count

        L = l_c.sum()
        p_c = l_c / L

        return p_c
    
    # Compability Functions
    def predict(self, model_input: ModelInput) -> dict[NetworkMode, GPNModelOutput]:
        model_outputs: dict[NetworkMode, GPNModelOutput] = self.forward(model_input)
        
        NUC.combine(
            combinationMethod=self.configuration.model.network_combination,
            model_outputs=model_outputs
        )
        
        return model_outputs
    
    def init(self, pytorch_seed, date_time_str, iteration, data_seed):
        torch.manual_seed(seed=pytorch_seed)
        self.pytorch_seed = pytorch_seed
        self.date_time_str = date_time_str
        self.iteration = iteration
        self.data_seed = data_seed
        
        self.optimizer = {
            Phase.WARMUP: self.get_warmup_optimizer(self.configuration.training.learning_rate[Phase.WARMUP], self.configuration.training.reg_lambda[Phase.WARMUP]),
            Phase.TRAINING: self.get_optimizer(self.configuration.training.learning_rate[Phase.TRAINING], self.configuration.training.reg_lambda[Phase.TRAINING])
        }
    
    def step(self, phase: Phase, data: Data) -> Results:
        is_training = phase in Phase.training_phases()
        self.set_eval(not is_training)
        if is_training:
            for optimizer in self.optimizer[phase]:
                optimizer.zero_grad()
        
        id_data = Data(
            model_input=ModelInput(
                features=data.model_input.features,
                indices=data.model_input.indices[data.ood_indicators==0]
            ),
            labels=data.labels[data.ood_indicators==0]
        )
        train_model_outputs = self.forward(id_data.model_input) 
        loss = torch.zeros([1], device=get_device())
        
        if is_training:
            loss_dict = self.loss(train_model_outputs[NetworkMode.PROPAGATED], id_data)
            for loss_item in loss_dict.values():
                loss += loss_item
            loss.backward()
            for optimizer in self.optimizer[phase]:
                optimizer.step()
        else:
            loss = F.nll_loss(train_model_outputs[NetworkMode.PROPAGATED].softmax_scores.log(), id_data.labels, reduction=self.params.loss_reduction).cpu().detach()    
        
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
            'model': copy.deepcopy(self.state_dict()),
            'config': copy.deepcopy(self.configuration.to_dict()),
            'torch_seed': copy.deepcopy(self.pytorch_seed),
            'data_seed': copy.deepcopy(self.data_seed),
            'iteration': copy.deepcopy(self.iteration),
            'date_time': copy.deepcopy(self.date_time_str)
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
        return {}
    
        