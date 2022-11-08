from torch import Tensor

class LogWeightValue():
    def __init__(self, tensor: Tensor):
        self.mean = tensor.mean()
        self.min = tensor.min()
        self.max = tensor.max()
            
class LogWeight():
    def __init__(self, name: str, mu: LogWeightValue, sigma: LogWeightValue = None, type: str = 'proj'):
        self.name = name
        self.mu = mu
        self.sigma = sigma