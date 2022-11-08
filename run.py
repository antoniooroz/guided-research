##############################################################
# This file is based on the reproduce_results_pytorch.ipynb file from the following source
# Author: Johannes Gasteiger, Aleksandar Bojchevski and Stephan Günnemann
# Last Visited: 14.06.2022
# Title: PPNP and APPNP
# URL: https://github.com/gasteigerjo/ppnp
##############################################################

import logging

import wandb
from pgnn.configuration.configuration import Configuration
from pgnn.configuration.experiment_configuration import ExperimentMode

from pgnn.training import train_model
from pgnn.data.io import load_dataset

from pgnn.logger import Logger

import  pgnn.utils.arguments_parsing as arguments_parsing


def run():
    logging.basicConfig(
        format='%(asctime)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO + 2)
    
    # Configuration
    args = arguments_parsing.parse_args()
    config_dict = arguments_parsing.overwrite_with_config_args(args)
    configuration = Configuration(config_dict)
    
    # TEST mode warning
    if configuration.experiment.seeds.experiment_mode == ExperimentMode.TEST:
        logging.log(32, f"TEST MODE enabled")

    # Graph
    graph = load_dataset(configuration.experiment.dataset.value)
    graph.standardize(select_lcc=True)
    graph.normalize_features(configuration=configuration)

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
                graph=graph, 
                seed=seed, 
                iteration=iteration, 
                logger=logger, 
                configuration=configuration
            )
            logger.finishIteration()

    logger.finish()
    
        
if __name__ == "__main__":
    run()
    

    