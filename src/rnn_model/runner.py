class MyVMC(VMC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def local_energy(self, samples, params, model, log_psi) -> List[float]:
        output = jnp.zeros((samples.shape[0]), dtype=jnp.float32)

        #** TODO_1: Somemething is missing here **#
        def step_fn_chemical(i, state):
            s, output = state
            output += - `CHANGE_ME` * s[:, i] # Do we need to multiply by the chemical potential?
            return s, output

        # Compute the interaction term
        def step_fn_intr(i, state):
            samples, pairs, multipliers, output = state
            output += multipliers[i] * samples[:, pairs[i, 0]] * samples[:, pairs[i, 1]]
            return samples, pairs, multipliers, output

        # Compute the off-diagonal term
        #** TODO_2: Somemething is missing here **#
        def step_fn_transverse(i, state):
            s, output = state
            flipped_state = s.at[:, i].set(1 - s[:, i])
            flipped_logpsi = self.logpsi(flipped_state, params, model)
            output += - 0.5 * `CHANGE_ME` * jnp.exp(flipped_logpsi - log_psi) # Something about the Rabi frequency
            return s, output

        # Interaction Term
        _, _, _, interaction_term = lax.fori_loop(0, 120, step_fn_intr, (samples, self.pairs, self.multipliers, output))
        
        # Off Diagonal Term
        _, transverse_field = lax.fori_loop(0, 16, step_fn_transverse, (samples, output))

        # Occupancy Term
        _, chemical_potential = lax.fori_loop(0, 16, step_fn_chemical, (samples, output))

        # Total energy
        #** TODO_3: Somemething very wrong here **#
        loc_e = `CHANGE_ME` + `CHANGE_ME` + `CHANGE_ME` # What should be the total energy?
        
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

# Initialize the model
dummy_input = jnp.zeros((vmc_config.n_samples, vmc_config.sequence_length, vmc_config.output_dim))
params = model.init(rng_key, dummy_input)
e_den = my_vmc.train(rng_key, params, model)

state.densities = [(i.mean() / vmc_config.sequence_length).item() for i in e_den]
state.training_completed = True