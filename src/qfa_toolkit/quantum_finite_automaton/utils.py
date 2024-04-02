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


def get_transition_from_initial_to_superposition(
    superposition: npt.NDArray[np.cdouble],
    *,
    _normalized: bool = True
) -> npt.NDArray[np.cdouble]:
    if _normalized:
        assert np.isclose(np.linalg.norm(superposition), 1)

    if np.count_nonzero(superposition) == 1:
        i = np.nonzero(superposition)[0][0]
        states = len(superposition)
        transition = np.eye(states, dtype=np.cdouble)
        transition[0][0] = 0
        transition[0][i] = 1
        transition[i][i] = 0
        transition[i][0] = 1
        return transition

    states_1 = len(superposition) // 2

    superposition_1 = superposition[:states_1]
    superposition_2 = superposition[states_1:]

    length_1 = np.linalg.norm(superposition_1)
    length_2 = np.linalg.norm(superposition_2)

    transition_1 = get_transition_from_initial_to_superposition(
        superposition_1 / length_1)
    transition_2 = get_transition_from_initial_to_superposition(
        superposition_2 / length_2)

    initial_transition = np.eye(len(superposition))
    initial_transition[0][0] = length_1
    initial_transition[0][states_1] = length_2
    initial_transition[states_1][0] = length_2
    initial_transition[states_1][states_1] = -length_1

    return direct_sum(transition_1, transition_2) @ initial_transition
