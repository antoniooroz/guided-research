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
from pgnn.logger import Logger
import tqdm
from pgnn.data import Data, ModelInput

from pgnn.ood import OOD_Experiment
from pgnn.result.result import Info, Results

import pgnn.models as models
from pgnn.utils.utils import matrix_to_torch

from .data.sparsegraph import SparseGraph
from .preprocessing import gen_seeds, gen_splits
from pgnn.utils import EarlyStopping, get_device, final_run

import pyro

def get_dataloaders(idx, labels, oods_all, batch_size=None):
    # MPS fix -> repeats only one item in dataloader...
    if torch.backends.mps.is_available():
        device = 'cpu'
    else:
        device = get_device()
    
    if batch_size is None:
        batch_size = max((val.numel() for val in idx.values()))
    datasets = {phase: TensorDataset(ind.to(device), labels[ind].to(device), oods_all[ind].to(device)) for phase, ind in idx.items()}
    dataloaders = {phase: DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
                   for phase, dataset in datasets.items()}
    return dataloaders


def train_model(graph: SparseGraph, seed: int, iteration: int,
                logger: Logger, configuration: Configuration = None):
    device = get_device()
    
    # Data
    idx_all = gen_splits(
        labels=graph.labels.astype('int64'), 
        idx_split_args={
            'ntrain_per_class': configuration.experiment.datapoints_training_per_class,
            'nstopping': configuration.experiment.datapoints_stopping,
            'nknown': configuration.experiment.datapoints_known,
            'seed': seed
        }, 
        test=configuration.experiment.seeds.experiment_mode==ExperimentMode.TEST
    )
    labels_all = torch.LongTensor(graph.labels.astype('int64')).to(device)
    oods_all = torch.zeros(labels_all.shape).to(device)
    feature_matrix = graph.attr_matrix.clone().detach().to(device)
    adjacency_matrix = matrix_to_torch(graph.adj_matrix)
    
    # OOD
    if configuration.experiment.ood != OOD.NONE:
        adjacency_matrix, feature_matrix, idx_all, labels_all, oods_all = OOD_Experiment.setup(
            configuration=configuration, 
            adjacency_matrix=adjacency_matrix, 
            feature_matrix=feature_matrix, 
            idx_all=idx_all, 
            labels_all=labels_all,
            oods_all=oods_all
        )
    
    # OOD-Setting: Remove left-out classes
    if configuration.experiment.ood_loc_remove_classes:
        nclasses = torch.max(labels_all[oods_all==0]).cpu().item() + 1
    else:
        nclasses = torch.max(labels_all).cpu().item() + 1
    
    nfeatures = feature_matrix.shape[1]        

    # Torch Seed and Logging
    logging.log(21, f"Training Model: {configuration.model.type.name}")
    torch_seed = gen_seeds()
    torch.manual_seed(seed=torch_seed) # TODO: Maybe make reproducible aswell
    logging.log(22, f"PyTorch seed: {torch_seed}")
    
    # Model
    model = getattr(models, configuration.model.type.value)(
        configuration=configuration,
        nfeatures=nfeatures,
        nclasses=nclasses,
        adj_matrix=adjacency_matrix,
        training_labels=labels_all[idx_all[Phase.TRAINING]] # GPN parameter
    ).to(device)
    model.init(torch_seed, configuration.custom_name, iteration, seed)
    
    if configuration.load is not None:
        model.load_model(
            mode=configuration.model.type,
            name=configuration.custom_name,
            seed=seed,
            iter=iteration
        )
        
    logger.watch(model)

    # Dataloaders, etc.
    dataloaders = get_dataloaders(idx_all, labels_all, oods_all)
    
    pyro.clear_param_store()
    
    early_stopping = EarlyStopping(
        model=model,
        stop_variable=configuration.training.early_stopping_variable
    )

    # Training
    start_time = time.time()
    if not configuration.training.skip_training:
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
                for phase in Phase.get_phases(training_phase):
                    start_time_phase = time.time()
                    results = Results()
                    
                    dataloader_phase = Phase.TRAINING if phase in Phase.training_phases() else phase
                    ########################################################################
                    # Training Step                                                        #
                    ########################################################################
                    for idx, labels, oods in dataloaders[dataloader_phase]:
                        data = Data(
                            model_input=ModelInput(features=feature_matrix, indices=idx.to(device)),
                            labels=labels.to(device),
                            ood_indicators=oods.to(device)
                        )
                        
                        results += model.step(phase if epoch > 0 else Phase.INIT, data)
                        
                    ########################################################################
                    # Logging                                                              #
                    ########################################################################
                    results.info = Info(
                        duration=time.time() - start_time_phase,
                        seed=seed,
                        iteration=iteration
                    )
                    
                    resultsPerPhase[phase] = results
                    logger.logStep(
                        phase=phase, 
                        results=results,
                        weights=model.log_weights()
                    )
                    
                
                pbar.set_postfix({'stopping acc': '{:.3f}'.format(resultsPerPhase[Phase.STOPPING].networkModeResults[NetworkMode.PROPAGATED].accuracy)})
                ########################################################################
                # Early Stopping                                                       #
                ########################################################################
                if early_stopping.check_stop(resultsPerPhase[Phase.STOPPING].networkModeResults[NetworkMode.PROPAGATED], epoch):
                    break           
                    
            # Load best model parameters from era
            early_stopping.load_best()

        runtime = time.time() - start_time
        runtime_perepoch = runtime / (epoch + 1)
        
        # Save best model
        model.save_model()
        
        pbar.close()
    else:
        epoch, runtime, runtime_perepoch = 0, 0, 0

    # Evaluation
    model.set_eval()
    
    resultsPerPhase = final_run(model, feature_matrix, idx_all, labels_all, oods_all)
    
    logger.logEval(resultsPerPhase=resultsPerPhase, weights=model.log_weights())
    logger.logAdditionalStats({
        'last_epoch': epoch,
        'best_epoch': early_stopping.best.epoch,
        'runtime': runtime,
        'runtime_per_epoch': runtime_perepoch
    })