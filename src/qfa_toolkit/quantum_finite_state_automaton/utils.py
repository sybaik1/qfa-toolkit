import numpy as np
import numpy.typing as npt

from .quantum_finite_state_automaton_base import Transition


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
        assert np.isclose(np.linalg.norm(superposition), 1), str(superposition)

    if len(superposition) == 1:
        return np.array([[1]], dtype=np.cdouble)

    states_1 = len(superposition) // 2
    states_2 = len(superposition) - states_1

    superposition_1 = superposition[:states_1]
    superposition_2 = superposition[states_1:]

    length_1 = np.linalg.norm(superposition_1)
    length_2 = np.linalg.norm(superposition_2)

    normalized_superposition_1 = (
        superposition_1 / length_1
        if length_1 > 0 else np.array([1] + [0] * (states_1 - 1))
    )
    normalized_superposition_2 = (
        superposition_2 / length_2
        if length_2 > 0 else np.array([1] + [0] * (states_2 - 1))
    )

    transition_1 = get_transition_from_initial_to_superposition(
        normalized_superposition_1)
    transition_2 = get_transition_from_initial_to_superposition(
        normalized_superposition_2)

    initial_transition = np.eye(len(superposition))
    initial_transition[0][0] = length_1
    initial_transition[0][states_1] = length_2
    initial_transition[states_1][0] = length_2
    initial_transition[states_1][states_1] = -length_1

    return direct_sum(transition_1, transition_2) @ initial_transition


def mapping_to_transition(mapping: dict[int, int]) -> npt.NDArray[np.cdouble]:
    states = len(mapping)
    transition = np.zeros((states, states), dtype=np.cdouble)
    for i, j in mapping.items():
        transition[i][j] = 1
    return transition
