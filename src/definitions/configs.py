# src/definitions/configs.py

from typing import NamedTuple

class RNNConfig(NamedTuple):
    output_dim: int = 2
    num_hidden_units: int = 64

class TransformerConfig(NamedTuple):
    num_layers: int = 5
    input_dim: int = 64
    num_heads: int = 8
    dim_feedforward: int = 32
    dropout_prob: float = 0.1

class VMCConfig(NamedTuple):
    n_samples: int = 1000
    nx: int = 4
    learning_rate: float = 0.01
    num_epochs: int = 1000
    output_dim: int = 2
    sequence_length: int = 16
    num_hidden_units: int = 64

RNNConfigType = RNNConfig
TransformerConfigType = TransformerConfig
