from .base import Observable
import numpy as np
from jax import numpy as jnp

class SigmaX(Observable):
    def __init__(self):
        self.name = "SigmaX"
        self.symbol = "X"

    def compute(self, nn_state, num_samples = 1000):
        samples = nn_state.sample(num_samples)
        log_samp = nn_state.logpsi(samples)
        seq_len = samples.shape[-1]
        
        flipped_samples = jnp.copy(samples)
        flip_results = []
        for i in range(seq_len):
            flipped_samples = flipped_samples.at[:, i].set(1 - flipped_samples[:, i])
            log_flipped_samples = nn_state.logpsi(flipped_samples)

            contrib = jnp.exp(log_flipped_samples - log_samp)
            flip_results.append(contrib)

        return jnp.mean(jnp.array(flip_results)).item()


class SigmaZ(Observable):
    def __init__(self):
        self.name = "SigmaZ"
        self.symbol = "Z"

    def compute(self, nn_state, num_samples: int = 1000):
        samples = nn_state.sample(num_samples)
        tmp = self.convert_to_11(samples)
        return jnp.mean(jnp.abs(jnp.mean(tmp, axis = 1))).item()

    @staticmethod
    def convert_to_11(states):
        return 2 * jnp.array(states) - 1
    
    
class SWAP(Observable):
    def __init__(self, A):
        self.name = "SWAP"
        self.symbol = "S"
        self.A = A

    def compute(self, nn_state, num_samples: int = 1000):
        # duplicate the state
        samples_1 = nn_state.sample(num_samples)
        samples_2 = nn_state.sample(num_samples)

        # perform swap operation
        swapped_1, swapped_2 = self.swap(samples_1, samples_2, self.A)

        # compute overlap
        logpsi_samples1 = nn_state.logpsi(samples_1)
        logpsi_samples2 = nn_state.logpsi(samples_2)
        logpsi_swapped_samples1 = nn_state.logpsi(swapped_1)
        logpsi_swapped_samples2 = nn_state.logpsi(swapped_2)

        overlap = logpsi_samples1 + logpsi_samples2 - logpsi_swapped_samples1 - logpsi_swapped_samples2
        overlap = jnp.exp(overlap)
        expectation_value = jnp.mean(overlap)

        # Renyi entropy
        renyi_entrop = - jnp.log(expectation_value)
        return renyi_entrop.item()


    @staticmethod
    def swap(arr1, arr2, subset):
        arr1, arr2 = np.asarray(arr1), np.asarray(arr2)
        
        swapped_state_1 = np.copy(arr1)
        swapped_state_2 = np.copy(arr2)
        

        swapped_state_1[:, subset] = arr2[:, subset]
        swapped_state_2[:, subset] = arr1[:, subset]

        return swapped_state_1, swapped_state_2
