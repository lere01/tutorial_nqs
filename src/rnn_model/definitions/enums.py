# src/definitions/enums.py

from enum import Enum
from .configs import RNNConfig, TransformerConfig

class ModelType(Enum):
    RNN = 1
    TRANSFORMER = 2

class LossType(Enum):
    HAMILTONIAN = 1
    DATA = 2

class ModelConfigType(Enum):
    RNN = RNNConfig
    TRANSFORMER = TransformerConfig
