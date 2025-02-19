##############################################################
# This file is based on the reproduce_results_pytorch.ipynb file from the following source
# Author: Johannes Gasteiger, Aleksandar Bojchevski and Stephan Günnemann
# Last Visited: 14.06.2022
# Title: PPNP and APPNP
# URL: https://github.com/gasteigerjo/ppnp
##############################################################

import logging

from sacred import Experiment
from pgnn.configuration import experiment_configuration
from pgnn.configuration.configuration import Configuration
from pgnn.configuration.experiment_configuration import ExperimentMode

from pgnn.training import train_model

from pgnn.logger import Logger
from pgnn.data.graph_data import GraphData

import  pgnn.utils.arguments_parsing as arguments_parsing
import seml

ex = Experiment()
seml.setup_logger(ex)

@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))
    
@ex.automain
def run(config = None, overrides = None):
    logging.basicConfig(
        format='%(asctime)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO + 2)
    
    # Configuration
    config = arguments_parsing.seml_config_parsing(config)
    config_dict = arguments_parsing.overwrite_with_config_args(config)
    arguments_parsing.parse_dict(config_dict, overrides)
    configuration = Configuration(config_dict)
    
    # TEST mode warning
    if configuration.experiment.seeds.experiment_mode == ExperimentMode.TEST:
        logging.log(32, f"TEST MODE enabled")

    graph_data = GraphData(configuration)

    # Logging
    logger = Logger(configuration)
    
    # Iterations
    total_iterations = configuration.experiment.iterations_per_seed * len(configuration.experiment.seeds.seed_list)
    current_iteration = 0
    
    # Training & Evaluation
    for seed in configuration.experiment.seeds.seed_list:
        for iteration in range(configuration.experiment.iterations_per_seed):            
            current_iteration += 1
            logging.log(22, f"Iteration {current_iteration}/{total_iterations}")
            
            logger.newIteration(seed, iteration)
            train_model(
                graph_data=graph_data, 
                seed=seed, 
                iteration=iteration, 
                logger=logger, 
                configuration=configuration
            )
            logger.finishIteration()

    return logger.finish()

    