import numpy as np
import pdb
from typing import Callable
from numba import jit
import copy
import time
import sys


def direct_method(end_time: int,
                  end_steps: int,
                  rate_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
                  rates: np.ndarray,
                  species: np.ndarray,
                  reaction_effects: np.ndarray):
    """The direct simulation method introduced in Gillespie 1977.

    For each state s_i, define the rate at which it transitions to state s_j r_ij \in [o, \infty).
    The probability of transition from s_i to s_j is r_ij / sum_j r_ij.
    The *time* at which the transition occurs is drawn from an exponential distribution with lambda value sum_j r_ij.

    The Gillespie algorithm just draws from these two distributions and updates the state accordingly.

    The state is assumed to be in R^n, since the algorithm is meant for simulating the number of reactants in a system.

    Args:
        end_time:
        end_steps:
        rate_function: A function which computes
        rates:
        species:
        reaction_effects: A mapping of reactions to reaction effects. An effect is a vector which gets added to the current
            state to produce the next state. A very simple birth reaction would have [1] as its effect vector, since it
            increases the number of reactants by 1.

    Returns:

    """
    simulation_time = 0
    time_step = 0
    random_count = 0

    # TODO: Profile this to establish trade off between function calls and cache misses.
    log_random_values = -np.log(np.random.random(1000))
    random_values = np.random.random(1000)

    # TODO: Convert steps vs time to closures.
    while simulation_time < end_time and time_step < end_steps:

        rates = rate_function(rates, species)
        total_rate = rates.sum()

        if total_rate <= 0:
            break

        if random_count == 1000:
            log_random_values = -np.log(np.random.random(1000))
            random_values = np.random.random(1000)
            random_count = 0

        random_cutoff = random_values[random_count]
        time_to_next_reaction = log_random_values[random_count] / total_rate
        simulation_time += time_to_next_reaction

        true_cutoff = random_cutoff * total_rate
        # TODO: Compare cumsum to just normalizing
        cutoffs = np.cumsum(rates)
        selected_reaction = np.searchsorted(cutoffs, true_cutoff, side='right')
        species += reaction_effects[selected_reaction]
        time_step += 1
        random_count += 1
    return species, rates


algorithms_mapping = {
    "direct": direct_method
}
