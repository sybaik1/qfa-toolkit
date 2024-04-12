from functools import reduce
import abc
import math
from typing import (TypeVar, Union, Optional, )

import numpy as np
import numpy.typing as npt


Superposition = npt.NDArray[np.cdouble]  # (n, )-shape array of complex
States = npt.NDArray[np.bool_]  # (n, )-shape array of bool
Transitions = npt.NDArray[np.cdouble]  # (m, n, n)-shape of complex
Transition = npt.NDArray[np.cdouble]  # (n, n)-shape of complex
Observable = npt.NDArray[np.bool_]  # (3, n)-shape of bool


class InvalidQuantumFiniteStateAutomatonError(Exception):
    pass


class NotClosedUnderOperationException(Exception):
    pass


class TotalState():
    """
    Attila Kondacs and John Watros. On the power of quantum finite automata.
    1997. 38th Annual Symposium on Foundations of Computer Science
    """

    def __init__(
        self,
        superposition_or_list: Union[Superposition, list[complex]],
        acceptance: float = 0,
        rejection: float = 0
    ) -> None:
        superposition = np.array(superposition_or_list, dtype=np.cdouble)
        norm_square = np.linalg.norm(superposition) ** 2
        if not math.isclose(norm_square + acceptance + rejection, 1):
            raise ValueError(
                "The sum of the superposition ({}), "
                "acceptance ({}) and rejection ({}) "
                "probabilities must be 1"
                .format(norm_square, acceptance, rejection)
            )

        self.superposition = superposition
        self.acceptance = acceptance
        self.rejection = rejection

    @classmethod
    def initial(self, states: int) -> 'TotalState':
        superposition = np.zeros(states, dtype=np.cdouble)
        superposition[0] = 1
        return TotalState(superposition)

    def measure_by(self, observable: Observable) -> 'TotalState':

        superpositions = observable * self.superposition

        acccepting_superposition = superpositions[0]
        rejecting_superposition = superpositions[1]
        superposition = superpositions[2]

        delta_acceptance = np.linalg.norm(acccepting_superposition) ** 2
        delta_rejection = np.linalg.norm(rejecting_superposition) ** 2

        acceptance = (self.acceptance + delta_acceptance).item()
        rejection = (self.rejection + delta_rejection).item()

        total_state = TotalState(superposition, acceptance, rejection)
        return total_state.normalized()

    def apply(self, unitary: npt.NDArray[np.cdouble]) -> 'TotalState':
        superposition = unitary @ self.superposition
        total_state = TotalState(
            superposition, self.acceptance, self.rejection)
        return total_state.normalized()

    def to_tuple(self) -> tuple[Superposition, float, float]:
        return (self.superposition, self.acceptance, self.rejection)

    def normalized(self) -> 'TotalState':
        norm = np.linalg.norm(self.superposition)
        factor = 1 / (norm ** 2 + self.acceptance + self.rejection)
        superposition = math.sqrt(factor) * self.superposition
        acceptance = (factor * self.acceptance).item()
        rejection = (factor * self.rejection).item()
        return TotalState(superposition, acceptance, rejection)


QfaT = TypeVar('QfaT', bound='QuantumFiniteStateAutomatonBase')


class QuantumFiniteStateAutomatonBase(abc.ABC):
    start_of_string: int = 0

    def __init__(self, transitions: npt.NDArray[np.cdouble]) -> None:
        if len(transitions.shape) != 3:
            raise ValueError("Transition matrix must be 3-dimensional")
        if transitions.shape[0] < 3:
            raise ValueError("Transition matrix must have at least 3 rows")
        if transitions.shape[1] != transitions.shape[2]:
            raise ValueError(
                "Each component of transition matrix must be square")
        identity = np.eye(transitions.shape[1])
        for u in transitions:
            if not np.allclose(identity, u.dot(u.T.conj())):
                raise ValueError(
                    "Each component of transition matrix must be unitary")
        self.transitions = transitions

    @property
    def alphabet(self) -> int:
        return self.transitions.shape[0] - 2

    @property
    def states(self) -> int:
        return self.transitions.shape[1]

    @property
    def end_of_string(self) -> int:
        return self.alphabet + 1

    @property
    def initial_transition(self) -> Transition:
        return self.transitions[self.start_of_string]

    @property
    def final_transition(self) -> Transition:
        return self.transitions[self.end_of_string]

    @abc.abstractmethod
    def __call__(self, w: list[int]) -> float:
        tape = self.string_to_tape(w)
        last_total_state = self.process(tape)
        _, acceptance, rejection = last_total_state.to_tuple()
        if not math.isclose(acceptance + rejection, 1):
            raise InvalidQuantumFiniteStateAutomatonError()
        return acceptance

    @property
    @abc.abstractmethod
    def observable(self) -> Observable:
        raise NotImplementedError()

    def process(
        self,
        w: list[int],
        total_state: Optional[TotalState] = None
    ) -> TotalState:
        if total_state is None:
            return self.process(w, TotalState.initial(self.states))
        return reduce(self.step, w, total_state)

    @abc.abstractmethod
    def step(self, total_state: TotalState, c: int) -> TotalState:
        raise NotImplementedError()

    def string_to_tape(self, string: list[int]) -> list[int]:
        return [self.start_of_string] + string + [self.end_of_string]
