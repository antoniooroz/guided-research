##############################################################
# This file is a modified version from the following source
# Author: Maximilian Stadler, Bertrand Charpentier, Simon Geisler, Daniel Zügner and Stephan Günnemann
# Last Visited: 14.06.2022
# Title: Graph Posterior Network
# URL: https://github.com/stadlmax/Graph-Posterior-Network
##############################################################
from typing import Any, List
import copy
from attr.exceptions import FrozenInstanceError


class HalfFrozenObject:
    """object which does not allow attributes to bet set without properly calling a setter"""
    def to_dict(self, ignore: List[str] = None) -> dict:
        d = {}
        ignore = set() if ignore is None else set(ignore)
        for name, value in vars(self).items():
            if (value is not None) and (name not in ignore):
                d[name] = value
        return d

    def set_value(self, name: str, value: Any) -> None:
        if hasattr(self, name):
            object.__setattr__(self, name, value)

        else:
            raise FrozenInstanceError(f'instance of class {self.__class__.__name__} has no attribute {name}')


    def set_values(self, **kwargs):
        for k, v in kwargs.items():
            self.set_value(k, v)

    def clone(self):
        return copy.deepcopy(self)
