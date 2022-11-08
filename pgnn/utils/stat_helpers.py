import numpy as np
import wandb
import statistics
import torch

def get_stats_for_column(table, column_name, logging_prefix):
    column_data = table.get_column(column_name, convert_to="numpy")
    iters = table.get_column("iteration", convert_to="numpy")
    seeds = table.get_column("seed", convert_to="numpy")

    mean, standard_deviation = _calculate_mean_and_standard_deviation(column_data)
    
    if column_data.dtype != 'object':
        max_idx = np.argmax(column_data)
        min_idx = np.argmin(column_data)
    else:
        max_idx = 0
        min_idx = 0
    
    return {
        "mean/" + logging_prefix: mean,
        "std/" + logging_prefix: standard_deviation,
        "name_max/" + logging_prefix: str(iters[max_idx]) + '_' + str(seeds[max_idx]),
        "max/" + logging_prefix: column_data[max_idx],
        "name_min/" + logging_prefix: str(iters[min_idx]) + '_' + str(seeds[min_idx]),
        "min/" + logging_prefix: column_data[min_idx],
    }

def accumulated_stats(stats, prefix, mode="mean"):
    if mode=="mean":
        fn = statistics.mean
    elif mode=="max":
        fn = max
    elif mode=="min":
        fn = min
    else:
        raise NotImplementedError()

    return {
        mode+"/"+prefix+"_loss":  wandb.plot.line(wandb.Table(data=get_stat_from_2d_list(stats["losses"], stats["max_steps"], fn), columns = ["step", "loss"]), "step", "loss", stroke=None, title=mode+" "+prefix+" "+"loss"),

        mode+"/"+prefix+"_conf_all":  wandb.plot.line(wandb.Table(data=get_stat_from_2d_list(stats["confs_all"], stats["max_steps"], fn), columns = ["step", "confidence"]), "step", "confidence", stroke=None, title=mode+" "+prefix+" "+"confidence all"),

        mode+"/"+prefix+"_acc":  wandb.plot.line(wandb.Table(data=get_stat_from_2d_list(stats["accs"], stats["max_steps"], fn), columns = ["step", "acc"]), "step", "acc", stroke=None, title=mode+" "+prefix+" "+"acc"),

        mode+"/"+prefix+"_conf_correct":  wandb.plot.line(wandb.Table(data=get_stat_from_2d_list(stats["confs_correct"], stats["max_steps"], fn), columns = ["step", "confidence"]), "step", "confidence", stroke=None, title=mode+" "+prefix+" "+"confidence correct"),

        mode+"/"+prefix+"_conf_false":  wandb.plot.line(wandb.Table(data=get_stat_from_2d_list(stats["confs_false"], stats["max_steps"], fn), columns = ["step", "confidence"]), "step", "confidence", stroke=None, title=mode+" "+prefix+" "+"confidence false")
    }

def get_stat_from_2d_list(losses, max_steps, fn=statistics.mean):
    mean_losses = []
     
    for i in range(max_steps):
        losses_for_step = []
        for j in range(len(losses)):
            if len(losses[j]) > i:
                losses_for_step.append(losses[j][i])
        mean_losses.append([i, fn(losses_for_step)])
    return mean_losses

# OOD
def ood_average_stats(prefix, stats, max_steps):
    def get_values(key):
        return list(map(lambda y: list(map(lambda x: x[key], y)), stats))
    logs = {}

    if len(stats) == 0 or len(stats[0]) == 0:
        return logs

    for key in stats[0][0].keys():
        logs["ood/avg_"+prefix+"_"+key] = wandb.plot.line(
                table = wandb.Table(
                    data=get_stat_from_2d_list(get_values(key), max_steps), 
                    columns = ["step", key]
                ), 
                x = "step", 
                y = key, 
                stroke=None, 
                title="ood/avg_"+prefix+"_"+key
            )

    return logs

def ood_final_stats(prefix, stats):
    logs = {}
    for key in stats[0].keys():
        values = list(map(lambda x: x[key], stats))
        mean, standard_deviation = _calculate_mean_and_standard_deviation(values)
        logs["ood/final_"+prefix+"_"+key] = mean
        logs["ood/final_"+prefix+"_"+key+"_std"] = standard_deviation
    return logs

def _calculate_mean_and_standard_deviation(values):
    if values.dtype != 'object':
        values = torch.FloatTensor(values)
        mean = values.mean()
        standard_deviation = torch.sqrt(torch.sum(torch.square(values - mean)) / values.shape[0])
        
        return mean.item(), standard_deviation.item()
    else:
        return None, None
    