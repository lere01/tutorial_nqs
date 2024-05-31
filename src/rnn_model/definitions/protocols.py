# src/definitions/protocols.py

import jax.numpy as jnp
from typing import NamedTuple
from typing import Protocol, List, Tuple, Union, runtime_checkable

@runtime_checkable
class VMCProtocol(Protocol):
    def sample(self, num_samples: int) -> List[Union[float, Tuple[float, ...]]]:
        """
        Method to sample quantum states.
        
        Parameters:
        - num_samples (int): Number of samples to generate.
        
        Returns:
        - List[Union[float, Tuple[float, ...]]]: List of samples.
        """
        pass
    
    def get_logpsi(self, samples: List[Union[float, Tuple[float, ...]]]) -> List[float]:
        """
        Method to compute the logarithm of the wavefunction for given samples.
        
        Parameters:
        - samples (List[Union[float, Tuple[float, ...]]]): List of samples.
        
        Returns:
        - List[float]: List of logarithms of wavefunction values.
        """
        pass
    
    def local_energy(self, samples: List[Union[float, Tuple[float, ...]]]) -> List[float]:
        """
        Method to compute the local energy for given samples.
        
        Parameters:
        - samples (List[Union[float, Tuple[float, ...]]]): List of samples.
        
        Returns:
        - List[float]: List of local energy values.
        """
        pass
    
    def get_interactions(self, samples: List[Union[float, Tuple[float, ...]]]) -> List[float]:
        """
        Method to compute interactions (all-to-all or nearest-nearest neighbours) for given samples.
        
        Parameters:
        - samples (List[Union[float, Tuple[float, ...]]]): List of samples.
        
        Returns:
        - List[float]: List of interaction values.
        """
        pass


@runtime_checkable
class RNNModelProtocol(Protocol):
    def rnn_type(self):
        pass


@runtime_checkable
class TransformerModelProtocol(Protocol):
    def transformer_type(self):
        pass

@runtime_checkable
class QuantumModelProtocol(Protocol):
    def setup(self):
        ...
