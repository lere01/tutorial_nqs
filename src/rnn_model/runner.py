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
        
        #** TODO_2 **#
        # Implement flip_state
        # It is a function that takes a flip index and state (1D array of spin configuration), and returns the flipped_state
        def flip_state(i: int, state: np.ndarray) -> np.ndarray:
            flipped_state = copy.deepcopy(state)
            return flipped_state

        # Compute the off-diagonal term
        #** TODO_3: Use  your function, flip_state and the instance method, self.logpsi to complete this function**#
        ## function get_logpsi takes (flipped_state, params, model) as arguments and returns the logpsi of the flipped state
        def step_fn_transverse(i, holder):
            state, output = holder
            flipped_state = None
            flipped_logpsi = None
            output += - 0.5 * Omega * jnp.exp(flipped_logpsi - log_psi) # Something about the Rabi frequency
            return state, output

        # Interaction Term
        _, _, _, interaction_term = lax.fori_loop(0, 120, step_fn_intr, (samples, self.pairs, self.multipliers, output))
        
        # Off Diagonal Term
        _, transverse_field = lax.fori_loop(0, 16, step_fn_transverse, (samples, output))

        # Occupancy Term
        _, chemical_potential = lax.fori_loop(0, 16, step_fn_chemical, (samples, output))

        # Total energy
        #** TODO_4: Somemething very wrong here **#
        loc_e = "" + "" + "" # What should be the total energy?
        
        return loc_e

