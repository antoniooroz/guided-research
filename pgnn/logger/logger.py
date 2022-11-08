from pgnn.configuration.configuration import Configuration
from pgnn.configuration.training_configuration import Phase
from pgnn.result.result import Results
from .log_weight import LogWeight
import wandb
import pgnn.utils.stat_helpers as stat_helpers

class Logger():
    def __init__(self, configuration: Configuration):
        self.wandb_run = wandb.init(
            project="PGNN", 
            name=f'{configuration.model.type.name} [{configuration.custom_name}]',
            config=configuration.to_dict(),
            reinit=True
        )
        self.configuration=configuration
        self.iterations:list[LogStep] = []
        
    def newIteration(self, seed, iteration):
        self.iterations.append(LogIteration(configuration=self.configuration, seed=seed, iteration=iteration))
        
    def logStep(self, phase: Phase, results: Results, weights: LogWeight):
        self.iterations[-1].logStep(phase, results, weights)
        
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
        
        logs['results_table'] = results_table
        
        wandb.log(logs)
        
        self.wandb_run.finish()
        
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
    def __init__(self, configuration: Configuration, seed, iteration):
        self.seed = seed
        self.iteration = iteration
        self.configuration = configuration
        self.step = 0
        
        self.log_training = {
            Phase.TRAINING: [],
            Phase.STOPPING: []
        }
        self.log_evaluation = {
            Phase.TRAINING: None,
            Phase.STOPPING: None,
            Phase.VALTEST: None
        }
        self.additionalStats=None
        
    def logStep(self, phase, results, weights):
        if phase in Phase.training_phases():
            phase = Phase.TRAINING
        logStep = LogStep(self.step, 'train', phase, results, weights)
        self.step += 1
        
        self.log_training[phase].append(logStep)
        if self.configuration.training.wandb_logging_during_training:
            logStep.wandbLog()
        
    def logEval(self, phase, results, weights):
        self.log_evaluation[phase] = LogStep(0, 'eval', phase, results, weights)
        
    def logAdditionalStats(self, stats):
        self.additionalStats = stats
        
    def getEvaluationResults(self):
        data = {'seed': self.seed, 'iteration': self.iteration, 'steps': self.step}
        for logStep in self.log_evaluation.values():
            data.update(logStep.getResults())
            
        return data
            
        
class LogStep():
    def __init__(self, step, mode, phase: Phase, results: Results, weights: dict[str, LogWeight] = None):
        self.mode = mode
        self.phase = phase
        self.log_prefix = f"{self.mode}/{self.phase.name}/"
        
        self.step = step
        self.results = results
        self.weights = weights
        
    def getResults(self):
        results = self.results.to_dict(prefix=self.log_prefix)
            
        return results
        
    def wandbLog(self):        
        wandb.log(
            data=self.getResults(), 
        )

    