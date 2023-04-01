##############################################################
# This file is a modified version from the following source
# Author: Johannes Gasteiger, Aleksandar Bojchevski and Stephan Günnemann
# Last Visited: 14.06.2022
# Title: PPNP and APPNP
# URL: https://github.com/gasteigerjo/ppnp
##############################################################

import time
import logging
import torch
from torch.utils.data import TensorDataset, DataLoader
from pgnn.base.base import Base
from pgnn.base.network_mode import NetworkMode
from pgnn.configuration.configuration import Configuration
from pgnn.configuration.experiment_configuration import OOD, ExperimentMode
from pgnn.configuration.training_configuration import Phase
from pgnn.data.graph_data import ActiveLearning, GraphData
from pgnn.logger import Logger
import tqdm
from pgnn.data import Data, ModelInput

from pgnn.result.result import Info, Results

import pgnn.models as models
from pgnn.utils.utils import get_active_learning_cycles, matrix_to_torch, balanced_weights

from .data.sparsegraph import SparseGraph
from .preprocessing import gen_seeds
from pgnn.utils import EarlyStopping, get_device, final_run

import wandb
import pyro

def train_model(graph_data: GraphData, seed: int, iteration: int,
                logger: Logger, configuration: Configuration = None):
    device = get_device()
    graph_data.init(seed)

    # Torch Seed and Logging
    logging.log(21, f"Training Model: {configuration.model.type.name}")
    logging.log(22, f"Seed: {seed}")
    logging.log(22, f"Seed-Iteration: {iteration+1}/{configuration.experiment.iterations_per_seed}")
    torch_seed = gen_seeds()
    torch.manual_seed(seed=torch_seed) # TODO: Maybe make reproducible aswell
    logging.log(22, f"PyTorch seed: {torch_seed}")
    
    # Model
    model = getattr(models, configuration.model.type.value)(**{
        'configuration': configuration,
        'nfeatures': graph_data.nfeatures,
        'nclasses': graph_data.nclasses,
        'adj_matrix': graph_data.adjacency_matrix,
        'training_labels': graph_data.labels_all[graph_data.idx_all[Phase.TRAINING]] # GPN parameter
    }).to(device)
    model.init(torch_seed, configuration.custom_name, iteration, seed)
    
    if configuration.load is not None:
        model.load_model(
            mode=configuration.model.type,
            name=configuration.custom_name,
            seed=seed,
            iter=iteration
        )
        
    #logger.watch(model)
    
    pyro.clear_param_store()
    
    early_stopping = EarlyStopping(
        model=model,
        stop_variable=configuration.training.early_stopping_variable
    )

    active_learning = ActiveLearning(configuration)

    if configuration.training.balanced_loss:
        loss_balance_weights = balanced_weights(
            n_classes=graph_data.nclasses, 
            labels=graph_data.labels_all[graph_data.idx_all[Phase.TRAINING]]
        )
    else:
        loss_balance_weights = None

    # Training
    start_time = time.time()
    if not configuration.training.skip_training:
        cycles = get_active_learning_cycles(configuration) if configuration.experiment.active_learning else 1
        
        for cycle in range(cycles):
            logging.log(22, f'Cycle: {cycle+1}/{cycles}')
            for training_phase in Phase.training_phases():
                if training_phase not in configuration.training.phases:
                    continue
                
                pbar = tqdm.tqdm(range(configuration.training.max_epochs[training_phase]))

                early_stopping.init_for_training_phase(
                    enabled=configuration.training.early_stopping[training_phase],
                    patience=configuration.training.patience[training_phase],
                    max_epochs=configuration.training.max_epochs[training_phase]
                )
                
                for epoch in pbar:
                    resultsPerPhase: dict[Phase, Results] = {}
                    
                    for phase in Phase.get_phases(training_phase, active_learning=configuration.experiment.active_learning):
                        start_time_phase = time.time()
                        results = Results()
                        
                        number_of_nodes = 0
                        
                        dataloader_phase = Phase.TRAINING if phase in Phase.training_phases() else phase
                        
                        ########################################################################
                        # Training Step                                                        #
                        ########################################################################
                        for idx, labels, oods in graph_data.dataloaders[dataloader_phase]:
                            data = Data(
                                model_input=ModelInput(features=graph_data.feature_matrix, indices=idx.to(device)),
                                labels=labels.to(device),
                                ood_indicators=oods.to(device)
                            )
                            
                            results += model.step(
                                phase=phase if epoch > 0 else Phase.INIT, 
                                data=data,
                                loss_balance_weights=loss_balance_weights
                            )
                            number_of_nodes += idx.shape[0]
                            
                        
                        results.info = Info(
                            duration=time.time() - start_time_phase,
                            seed=seed,
                            iteration=iteration,
                            number_of_nodes=number_of_nodes
                        )
                        
                        resultsPerPhase[phase] = results

                    ########################################################################
                    # Logging                                                              #
                    ########################################################################
                    logger.logStep(
                        results=resultsPerPhase,
                        weights=model.log_weights()
                    )
                    
                    pbar.set_postfix({'stopping acc': '{:.3f}'.format(resultsPerPhase[Phase.STOPPING].networkModeResults[NetworkMode.PROPAGATED].accuracy)})
                    ########################################################################
                    # Early Stopping                                                       #
                    ########################################################################
                    if early_stopping.check_stop(resultsPerPhase[Phase.STOPPING].networkModeResults[NetworkMode.PROPAGATED], epoch):
                        break
                
                early_stopping.load_best()
                    
            loss_balance_weights = active_learning.step(
                graph_data=graph_data,
                early_stopping=early_stopping,
                logger=logger,
                model=model,
                cycle=cycle
            )
            
        runtime = time.time() - start_time
        runtime_perepoch = runtime / (epoch + 1)
        
        early_stopping.load_best()
        # Save best model - Deactivated because of space
        # model.save_model()
        
        pbar.close()
    else:
        epoch, runtime, runtime_perepoch = 0, 0, 0

    # Evaluation
    model.set_eval()
    
    finalResultsPerPhase = final_run(model, graph_data.feature_matrix, graph_data.idx_all, graph_data.labels_all, graph_data.oods_all)
    
    logger.logEval(resultsPerPhase=finalResultsPerPhase, weights=model.log_weights())
    logger.logAdditionalStats({
        'last_epoch': epoch,
        'best_epoch': early_stopping.best.epoch,
        'runtime': runtime,
        'runtime_per_epoch': runtime_perepoch
    })