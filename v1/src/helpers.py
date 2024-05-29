# src/helpers.py
import optax
import flax.linen as nn
import jax.numpy as jnp
from  jax import lax, jit, random, value_and_grad
from jax import nn as jnn
from functools import lru_cache, partial
import jax.tree_util as tree_util

@lru_cache(maxsize=128, typed=False)
def get_all_interactions_jax(n: int) -> tuple:
    """
    Get all to all interactions from a n by n lattice using the euclidean distances.
    Assume a unit distance (1) between nearest neighbours

    Parameters
    ---
    n: integer representing a side of the square

    Output
    ---
    tuple[unique_pairs, multipliers]
    """

    # Create a grid of coordinates
    x, y = jnp.meshgrid(jnp.arange(n), jnp.arange(n))
    coordinates = jnp.stack([x.flatten(), y.flatten()], axis=1)

    # Calculate distances between all unique pairs
    num_points = coordinates.shape[0]
    distances = jnp.sqrt(
        jnp.sum((coordinates[:, None, :] - coordinates[None, :, :]) ** 2, axis=-1)
    )

    # Mask to select only unique pairs
    mask = jnp.triu(jnp.ones((num_points, num_points), dtype=bool), k=1)

    # Extract unique pairs, distances, and calculate multipliers
    unique_pairs = jnp.argwhere(mask)
    unique_distances = distances[mask]
    multipliers = 1 / unique_distances ** 6

    return unique_pairs, multipliers



def sample(model, params, nsamples, key, *, output_dim=2, num_hidden_units=64):
    # Initialize the hidden state and inputs
    initial_hidden_state = jnp.zeros((nsamples, num_hidden_units))
    inputs = 0.0 * jnn.one_hot(jnp.zeros((nsamples, 1), dtype=jnp.float32), output_dim)

    # Pre-allocate array for samples
    samples = jnp.zeros((nsamples, 16), dtype=jnp.int32)

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
        sample = sample.reshape((nsamples, 1))

        # Update inputs with the sampled data
        inputs = jnn.one_hot(sample, output_dim)

        # Update the samples array using .at method
        sample_array = sample_array.at[:, i].set(sample[:, 0])

        return hidden_state, inputs, sample_array

    # Run the sampling loop
    _, _, samples = lax.fori_loop(0, 16, step_fn, (initial_hidden_state, inputs, samples))

    return samples



def logpsi(samples, model, params, N = 16, ss=(0, 15), K = 2, nh = 64):
    nsamples = samples.shape[0]
    data   = jnn.one_hot(samples[:, ss[0]:ss[1]], K)
    x0 = 0.0 * jnn.one_hot(jnp.zeros((nsamples, 1)), K)
    inputs = jnp.concatenate([x0, data], axis = 1)

    hidden_state = jnp.zeros((nsamples, nh))

    _, logits = model.apply(params, inputs, initial_carry = hidden_state)
    log_probs        = nn.activation.log_softmax(logits)

    logP   = jnp.sum(jnp.multiply(log_probs, jnn.one_hot(samples, K)), axis=2)
    logP = 0.5 * jnp.sum(logP, axis=1)
    return logP


@partial(jit, static_argnums = [2])
def local_energy_opt(samples, log_psi, model, params, pairs, multipliers, *, Omega=1.0, delta=1.0):

    output = jnp.zeros((samples.shape[0]), dtype=jnp.float32)

    def step_fn_chemical(i, state):
      s, output = state
      output += - delta * s[:, i]
      return s, output

    def step_fn_intr(i, state):
      samples, pairs, multipliers, output = state
      output += multipliers[i] * samples[:, pairs[i, 0]] * samples[:, pairs[i, 1]]
      return samples, pairs, multipliers, output

    def step_fn_transverse(i, state):
      s, output = state
      flipped_state = s.at[:, i].set(1 - s[:, i])
      flipped_logpsi = logpsi(flipped_state, model, params, N = 16, K = 2, nh = 64)
      output += - Omega * jnp.exp(flipped_logpsi - log_psi)
      return s, output


    # Interaction Term
    _, _, _, interaction_term = lax.fori_loop(0, 120, step_fn_intr, (samples, pairs, multipliers, output))
    # Off Diagonal Term
    _, transverse_field = lax.fori_loop(0, 16, step_fn_transverse, (samples, output))
    # _, transverse_field = lax.fori_loop(0, 16, step_fn_transverse, (samples, output))
    # Occupancy Term
    _, chemical_potential = lax.fori_loop(0, 16, step_fn_chemical, (samples, output))

    # Total energy
    loc_e = transverse_field + chemical_potential + interaction_term
    return loc_e


def get_opt_loss(params, rng_key, model, nsamples, pairs, multipliers):
    def l2_loss(x, alpha):
      return alpha * (x ** 2).mean()

    @jit
    def all_reg():
      return sum(
          l2_loss(w, alpha=0.001) for w in tree_util.tree_leaves(params["params"])
      )

    samples = sample(model, params, nsamples, rng_key)
    log_psi = logpsi(samples, model, params, N = 16, K = 2, nh = 64)
    e_loc = local_energy_opt(samples, log_psi, model, params, pairs, multipliers)
    e_o = e_loc.mean()

    # We expand the equation in the text above
    first_term = 2 * jnp.multiply(log_psi, e_loc)
    second_term = 2 * jnp.multiply(e_o, log_psi)

    l2_reg = all_reg()

    loss = jnp.mean(first_term - second_term)
    # loss = l2_reg(params) + loss
    loss += l2_reg
    return loss, e_loc


def OptimizedVMC(modelClass):
  n = 4  # lattice size
  pairs, multipliers = get_all_interactions_jax(n)

  num_hidden_units = 64
  batch_size = 500
  output_dim = 2
  sequence_length = 16
  nsamples = 1000

  model = modelClass(num_hidden_units=num_hidden_units, output_dim=output_dim)


  rng_key = random.PRNGKey(0)
  dummy_input = jnp.zeros((batch_size, sequence_length, output_dim))
  params = model.init(rng_key, dummy_input)

  optimizer = optax.adam(learning_rate=1e-02)
  opt_state = optimizer.init(params)


  num_epochs = 2000

  @partial(jit, static_argnums = [3])
  def step(params, rng_key, opt_state, nsamples):
    rng_key, new_key = random.split(rng_key)

    value, grads = value_and_grad(get_opt_loss, has_aux=True)(params, rng_key, model, nsamples, pairs, multipliers)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return new_key, params, opt_state, value

  energies = []
  for i in range(num_epochs):
    rng_key, params, opt_state, (loss, eloc) = step(params, rng_key, opt_state, nsamples)
    energies.append(eloc)

    if i % 100 == 0:
      print(f'step {i}, loss: {loss}')

  return energies