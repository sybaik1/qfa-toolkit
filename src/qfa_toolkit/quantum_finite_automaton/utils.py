import numpy as np
import numpy.typing as npt

from .quantum_finite_automaton_base import Transition


def direct_sum(u: Transition, v: Transition) -> Transition:
    """Returns the direct sum of two matrices.

    Direct sum of U, V: (U, V) |-> [U 0; 0 V]
    """
    w1 = np.concatenate(
        (u, np.zeros((u.shape[0], v.shape[1]))), axis=1)
    w2 = np.concatenate(
        (np.zeros((v.shape[0], u.shape[1])), v), axis=1)
    w = np.concatenate((w1, w2), axis=0)
    return w


def get_real_valued_transition(
    transition: Transition
) -> npt.NDArray[np.double]:
    stacked = np.stack([
        [transition.real, transition.imag],
        [-transition.imag, transition.real]
    ])
    stacked = stacked.transpose((2, 0, 3, 1))
    stacked = stacked.reshape((2 * len(transition), 2 * len(transition)))
    return stacked
