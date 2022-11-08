from enum import Enum


class NetworkMode(Enum):
    PROPAGATED = 'propagated'
    ISOLATED = 'isolated'
    COMBINED = 'combined'
    