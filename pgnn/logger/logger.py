from pgnn.configuration.configuration import Configuration
from pgnn.configuration.training_configuration import Phase
from pgnn.result.result import Results
from .log_weight import LogWeight
import wandb
import pgnn.utils.stat_helpers as stat_helpers

class Logger():
    def __init__(self, configuration: Configuration):
        self.wandb_run = wandb.init(
            project=configuration.wandb_project, entity=configuration.wandb_entity,
            name=f'{configuration.model.type.name} [{configuration.custom_name}]',
            config=configuration.to_dict(),
            reinit=True
        )
        self.wandb_run.tags = configuration.tags
        self.configuration = configuration
        self.iterations:list[LogIteration] = []
        
    def newIteration(self, seed, iteration):
        self.iterations.append(LogIteration(
            configuration=self.configuration, 
            seed=seed, 
            iteration=iteration,
            start_step=0 if len(self.iterations)==0 else self.iterations[-1].step
        ))
        
    def logStep(self, results: dict[Phase, Results], weights: LogWeight):
        self.iterations[-1].logStep(results, weights)
        
    def logActiveLearning(self, resultsPerPhase: dict[Phase, Results], step):
        for phase, results in resultsPerPhase.items():
            self.iterations[-1].logActiveLearning(phase, results, step)
        
    def logEval(self, resultsPerPhase: dict[Phase, Results], weights: LogWeight):
        for phase, results in resultsPerPhase.items():
            self.iterations[-1].logEval(phase, results, weights)
        
    def logAdditionalStats(self, stats):
        self.iterations[-1].logAdditionalStats(stats)
        
    def finish(self):
        results_table = self.createResultsTable()
        
        logs = {}
        for column in results_table.columns:
            logs.update(stat_helpers.get_stats_for_column(results_table, column, column))
        
        wandb.log({**logs, 'results_table': results_table})
        
        self.wandb_run.finish()
        
        return logs
        
    def finishIteration(self):
        pass
    
    def createResultsTable(self) -> wandb.Table:
        all_results = list(map(lambda x: x.getEvaluationResults(), self.iterations))
        
        columns = set()
        for results in all_results:
            columns.update(results.keys())
            
        columns = list(columns)
        data = []
        for results in all_results:
            data_for_iter = []
            for column in columns:
                data_for_iter.append(results[column])
            data.append(data_for_iter)
            
        results_table = wandb.Table(data=data, columns=columns)
        
        return results_table
        
        
    def watch(self, model):
        wandb.watch(model)
        
    

class LogIteration():
    def __init__(self, configuration: Configuration, seed, iteration, start_step=0):
        self.seed = seed
        self.iteration = iteration
        self.configuration = configuration
        self.step = start_step
        
        self.log_training = {
            Phase.TRAINING: [],
            Phase.STOPPING: [],
            Phase.ACTIVE_LEARNING: []
        }
        
        self.log_active_learning = {
            Phase.TRAINING: [],
            Phase.STOPPING: [],
            Phase.VALTEST: [],
            Phase.ACTIVE_LEARNING: []
        }
        
        self.log_evaluation = {
            Phase.TRAINING: None,
            Phase.STOPPING: None,
            Phase.VALTEST: None,
            Phase.ACTIVE_LEARNING: None
        }
        
        if configuration.experiment.active_learning:
            self.log_training[Phase.ACTIVE_LEARNING] = []
        
        self.additionalStats=None
        
    def logStep(self, results, weights):
        for phase, val in results.items():
            if phase in Phase.training_phases():
                phase = Phase.TRAINING
            logStep = LogStep(self.configuration, self.step, 'train', phase, val, weights)
            self.log_training[phase].append(logStep)
        self.step += 1
        
    def logActiveLearning(self, phase, results, step):
        self.log_active_learning[phase].append(LogStep(self.configuration, 0, f'al/{step}', phase, results))
        
    def logEval(self, phase, results, weights):
        self.log_evaluation[phase] = LogStep(self.configuration, 0, 'eval', phase, results, weights)
        
    def logAdditionalStats(self, stats):
        self.additionalStats = stats
        
    def getEvaluationResults(self):
        data = {'seed': self.seed, 'iteration': self.iteration, 'steps': self.step}
        for logStep in self.log_evaluation.values():
            data.update(logStep.results)
            
        for l in self.log_active_learning.values():
            for logStep in l:
                data.update(logStep.results)
            
        return data
            
        
class LogStep():
    def __init__(self, configuration: Configuration, step, mode, phase: Phase, results: Results, weights: dict[str, LogWeight] = None):
        self.mode = mode
        self.phase = phase
        self.configuration = configuration
        self.log_prefix = f"{self.mode}/{self.phase.name}/"
        
        self.step = step
        self.results = results.to_dict(prefix=self.log_prefix)
        
        if self.configuration.training.wandb_logging_during_training:
            # TODO: Maybe add weights
            wandb.log(
                data=self.results, 
                step=step
            )  
        

    