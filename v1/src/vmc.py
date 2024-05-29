from typing import List, Tuple, Union
from abc import ABC, abstractmethod
from helpers import get_all_interactions_jax
from dataclasses import dataclass
from definitions import VMCConfig
import jax.numpy as jnp
from jax import nn as jnn
from jax import random, jit, lax, value_and_grad
from flax import linen as nn
from functools import partial
import jax.tree_util as tree_util   
import optax

@dataclass
class VMC(ABC):
    nsamples: int
    n: int
    learning_rate: float
    num_epochs: int
    output_dim: int
    sequence_length: int
    num_hidden_units: int

    def __post_init__(self):
        self.pairs, self.multipliers = get_all_interactions_jax(self.n)
        self.Omega = 1.0
        self.delta = 1.0

    
    
    def sample(self, key, params, model) -> List[Union[float, Tuple[float, ...]]]:
        # Initialize the hidden state and inputs
        initial_hidden_state = jnp.zeros((self.nsamples, self.num_hidden_units))
        inputs = 0.0 * jnn.one_hot(jnp.zeros((self.nsamples, 1), dtype=jnp.float32), self.output_dim)

        # Pre-allocate array for samples
        samples = jnp.zeros((self.nsamples, 16), dtype=jnp.int32)

        # Pre-generate random keys for sampling
        keys = random.split(key, 16)

        @jit
        def step_fn(i, state):
            hidden_state, inputs, sample_array = state
            # Run a single RNN cell
            hidden_state, logits = model.apply(params, inputs, initial_carry=hidden_state)

            # Compute log probabilities
            log_probs = jnn.log_softmax(logits)

            # Sample from the categorical distribution
            sample = random.categorical(keys[i], log_probs)
            # Reshape sample for compatibility with sample_array
            sample = sample.reshape((self.nsamples, 1))

            # Update inputs with the sampled data
            inputs = jnn.one_hot(sample, self.output_dim)

            # Update the samples array using .at method
            sample_array = sample_array.at[:, i].set(sample[:, 0])

            return hidden_state, inputs, sample_array

        # Run the sampling loop
        _, _, samples = lax.fori_loop(0, 16, step_fn, (initial_hidden_state, inputs, samples))

        return samples
    

    def logpsi(self, samples: List[Union[float, Tuple[float, ...]]], params, model) -> List[float]:
        ss = (0, self.sequence_length - 1)
        nsamples = samples.shape[0]
        data   = jnn.one_hot(samples[:, ss[0]:ss[1]], self.output_dim)
        x0 = 0.0 * jnn.one_hot(jnp.zeros((nsamples, 1)), self.output_dim)
        inputs = jnp.concatenate([x0, data], axis = 1)

        hidden_state = jnp.zeros((nsamples, self.num_hidden_units))

        _, logits = model.apply(params, inputs, initial_carry = hidden_state)
        log_probs = nn.activation.log_softmax(logits)

        logP   = jnp.sum(jnp.multiply(log_probs, jnn.one_hot(samples, self.output_dim)), axis=2)
        logP = 0.5 * jnp.sum(logP, axis=1)
        return logP
    
    


    def get_loss(self, params, rng_key, model):
        def l2_loss(x, alpha):
            return alpha * (x ** 2).mean()

        @jit
        def all_reg():
            return sum(
                l2_loss(w, alpha=0.001) for w in tree_util.tree_leaves(params["params"])
            )

        samples = self.sample(rng_key, params, model)
        log_psi = self.logpsi(samples, params, model)
        e_loc = self.local_energy(samples, params, model, log_psi)
        e_o = e_loc.mean()

        # We expand the equation in the text above
        first_term = 2 * jnp.multiply(log_psi, e_loc)
        second_term = 2 * jnp.multiply(e_o, log_psi)

        l2_reg = all_reg()

        loss = jnp.mean(first_term - second_term)
        # loss = l2_reg(params) + loss
        loss += l2_reg
        return loss, e_loc
    

    def train(self, rng_key, params, model):
        rng_key = random.PRNGKey(0)
        
        optimizer = optax.adam(learning_rate=self.learning_rate)
        opt_state = optimizer.init(params)

        loss_fn = self.get_loss
    

        @partial(jit, static_argnums=(3,))
        def step(params, rng_key, opt_state, get_loss=loss_fn):
            rng_key, new_key = random.split(rng_key)

            value, grads = value_and_grad(get_loss, has_aux=True)(params, rng_key, model)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return new_key, params, opt_state, value

        energies = []
        for i in range(self.num_epochs):
            rng_key, params, opt_state, (loss, eloc) = step(params, rng_key, opt_state)
            energies.append(eloc)

            if i % 100 == 0:
                print(f'step {i}, loss: {loss}')

        return energies
    
    
    @abstractmethod
    def local_energy(self, samples: List[Union[float, Tuple[float, ...]]], params, model, log_psi) -> List[float]:
        """
        Method to compute the energy function for given samples.
        
        Parameters:
        - samples (List[Union[float, Tuple[float, ...]]]): List of samples.
        
        Returns:
        - float: Energy function value.
        """
        raise NotImplementedError("Energy function not implemented")