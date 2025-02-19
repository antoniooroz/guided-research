##############################################################
# This file is a modified version from the following source
# Author: Maximilian Stadler, Bertrand Charpentier, Simon Geisler, Daniel Zügner and Stephan Günnemann
# Last Visited: 14.06.2022
# Title: Graph Posterior Network
# URL: https://github.com/stadlmax/Graph-Posterior-Network
##############################################################
import torch.distributions as D
import pgnn.models.gpn.gpn_fns.distributions as E
from .base import Likelihood

# pylint: disable=abstract-method
class Categorical(D.Categorical, Likelihood):
    """
    Extension of PyTorch's native Categorical distribution to be used as a likelihood function.
    """

    @classmethod
    def __prior__(cls):
        return E.Dirichlet

    @classmethod
    def from_model_params(cls, x):
        return cls(x.softmax(-1))

    @property
    def mean(self):
        return self.logits.argmax(-1)

    @property
    def sufficient_statistic_mean(self):
        return self.probs

    def to(self, *args, **kwargs):
        if 'probs' in self.__dict__:
            self.probs = self.probs.to(*args, **kwargs)
        else:
            self.logits = self.logits.to(*args, **kwargs)
        return self
