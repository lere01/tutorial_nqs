import os
from jax import nn as jnn
from flax import linen as nn
from jax import numpy as jnp
from flax import serialization
from jax import random, jit, lax
from dataclasses import dataclass

from typing import Union, Tuple, List, Any

file_dir = os.path.abspath(os.path.join(os.path.dirname('__file__'), '.'))

class RNNModel(nn.Module):
    """
    Neural Network for parameterizing our wavefunction.
    """
    output_dim: int
    num_hidden_units: int

    def setup(self):
        # Initialize the GRU cell with the specified number of hidden units
        gru_cell = nn.GRUCell(
            name='gru_cell',
            features=self.num_hidden_units,
            # kernel_init = jnn.initializers.glorot_uniform()
        )
        self.rnn = nn.RNN(gru_cell, return_carry=True)
        self.dense = nn.Dense(
            self.output_dim,
            name = 'dense_layer',
            # kernel_init = jnn.initializers.glorot_uniform()
        )

    
    def __call__(self, x, initial_carry=None):
        # Apply GRU layers
        carry, x = self.rnn(x, initial_carry = initial_carry)

        # Output layer
        x = self.dense(x)

        return carry, x

@dataclass
class Autoload:
    output_dim: int = 2
    num_hidden_units: int = 128
    sequence_length: int = 16
    saved_params_dir: str = os.path.join(file_dir, "saved_params", "model_params2.pkl")
    
    def __post_init__(self):
        self.rng = random.PRNGKey(1245)
        self.model = RNNModel(output_dim=self.output_dim, num_hidden_units=self.num_hidden_units)
        dummy_input = jnp.zeros((1, self.sequence_length, self.output_dim))
        self.params_arch = self.model.init(random.PRNGKey(1234), dummy_input)
        
        with open(self.saved_params_dir, "rb") as f:
            self.param_bytes = f.read()

        self.load()


    def load(self):
        """Return loaded model and params
        """
        self.params = serialization.from_bytes(self.params_arch, self.param_bytes)
    

    def sample(self, num_samples: int = 1000) -> List[Union[float, Tuple[float, ...]]]:
        # Initialize the hidden state and inputs
        initial_hidden_state = jnp.zeros((num_samples, self.num_hidden_units))
        inputs = 0.0 * jnn.one_hot(jnp.zeros((num_samples, 1), dtype=jnp.float32), self.output_dim)

        # Pre-allocate array for samples
        samples = jnp.zeros((num_samples, self.sequence_length), dtype=jnp.int32)

        # Pre-generate random keys for sampling
        keys = random.split(self.rng, 16)

        @jit
        def step_fn(i, state):
            hidden_state, inputs, sample_array = state
            # Run a single RNN cell
            hidden_state, logits = self.model.apply(self.params, inputs, initial_carry=hidden_state)

            # Compute log probabilities
            log_probs = jnn.log_softmax(logits)

            # Sample from the categorical distribution
            sample = random.categorical(keys[i], log_probs)
            # Reshape sample for compatibility with sample_array
            sample = sample.reshape((num_samples, 1))

            # Update inputs with the sampled data
            inputs = jnn.one_hot(sample, self.output_dim)

            # Update the samples array using .at method
            sample_array = sample_array.at[:, i].set(sample[:, 0])

            return hidden_state, inputs, sample_array

        # Run the sampling loop
        _, _, samples = lax.fori_loop(0, 16, step_fn, (initial_hidden_state, inputs, samples))

        return samples
    
    def logpsi(self, samples: List[Union[float, Tuple[float, ...]]]) -> List[float]:
        ss = (0, self.sequence_length - 1)
        nsamples = samples.shape[0]
        data   = jnn.one_hot(samples[:, ss[0]:ss[1]], self.output_dim)
        x0 = 0.0 * jnn.one_hot(jnp.zeros((nsamples, 1)), self.output_dim)
        inputs = jnp.concatenate([x0, data], axis = 1)

        hidden_state = jnp.zeros((nsamples, self.num_hidden_units))

        _, logits = self.model.apply(self.params, inputs, initial_carry = hidden_state)
        log_probs = nn.activation.log_softmax(logits)

        logP   = jnp.sum(jnp.multiply(log_probs, jnn.one_hot(samples, self.output_dim)), axis=2)
        logP = 0.5 * jnp.sum(logP, axis=1)
        return logP
