class MyVMC(VMC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def local_energy(self, samples, params, model, log_psi) -> List[float]:
        output = jnp.zeros((samples.shape[0]), dtype=jnp.float32)

        def step_fn_chemical(i, state):
            s, output = state
            output += - self.delta * s[:, i]
            return s, output

        # Compute the interaction term
        def step_fn_intr(i, state):
            samples, pairs, multipliers, output = state
            output += multipliers[i] * samples[:, pairs[i, 0]] * samples[:, pairs[i, 1]]
            return samples, pairs, multipliers, output

        # Compute the off-diagonal term
        def step_fn_transverse(i, state):
            s, output = state
            flipped_state = s.at[:, i].set(1 - s[:, i])
            flipped_logpsi = self.logpsi(flipped_state, params, model)
            output += - self.Omega * jnp.exp(flipped_logpsi - log_psi)
            return s, output

        # Interaction Term
        _, _, _, interaction_term = lax.fori_loop(0, 120, step_fn_intr, (samples, self.pairs, self.multipliers, output))
        
        # Off Diagonal Term
        _, transverse_field = lax.fori_loop(0, 16, step_fn_transverse, (samples, output))

        # Occupancy Term
        _, chemical_potential = lax.fori_loop(0, 16, step_fn_chemical, (samples, output))

        # Total energy
        loc_e = transverse_field + chemical_potential + interaction_term
        
        return loc_e


"""WARNING: DO NOT EDIT THIS PART OF THE CODE"""

my_vmc = MyVMC(
    nsamples=vmc_config.n_samples,
    n=vmc_config.nx,
    learning_rate=vmc_config.learning_rate,
    num_epochs=vmc_config.num_epochs,
    output_dim=vmc_config.output_dim,
    sequence_length=vmc_config.sequence_length,
    num_hidden_units=vmc_config.num_hidden_units
)

# print(my_vmc)

# # Initialize the model
# dummy_input = jnp.zeros((vmc_config.n_samples, vmc_config.sequence_length, vmc_config.output_dim))
# params = model.init(rng_key, dummy_input)
# e_den = my_vmc.train(rng_key, params, model)

# print('Completed!')

output_container = st.empty()
    
# Redirect stdout to capture print statements
old_stdout = sys.stdout
new_stdout = io.StringIO()
sys.stdout = new_stdout

try:
    print(my_vmc)

    # Initialize the model
    dummy_input = jnp.zeros((vmc_config.n_samples, vmc_config.sequence_length, vmc_config.output_dim))
    params = model.init(rng_key, dummy_input)
    e_den = my_vmc.train(rng_key, params, model)

    print('Completed!')
except Exception as e:
    st.error(f"Error executing code: {e}")
finally:
    sys.stdout = old_stdout

output_container.text(new_stdout.getvalue())