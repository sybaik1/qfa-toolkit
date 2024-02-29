import abc
import math
from typing import (TypeVar, Union, Generic, )

import numpy as np
import numpy.typing as npt


Superposition = npt.NDArray[np.cdouble]
Observable = npt.NDArray[bool]


class InvalidQuantumFiniteAutomatonError(Exception):
    pass


class NotClosedUnderOperationException(Exception):
    pass


class TotalState():

    def __init__(
        self,
        superposition_or_list: Union[Superposition, list[complex]],
        acceptance: float,
        rejection: float
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
        return TotalState(superposition, 0, 0)

    def measure_by(self, observable: Observable) -> 'TotalState':

        superpositions = observable * self.superposition

        acccepting_superposition = superpositions[0]
        rejecting_superposition = superpositions[1]
        superposition = superpositions[2]

        delta_acceptance = np.linalg.norm(acccepting_superposition) ** 2
        delta_rejection = np.linalg.norm(rejecting_superposition) ** 2

        acceptance = self.acceptance + delta_acceptance
        rejection = self.rejection + delta_rejection

        return TotalState(superposition, acceptance, rejection)

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
        acceptance = factor * self.acceptance
        rejection = factor * self.rejection
        return TotalState(superposition, acceptance, rejection)



QfaType = TypeVar('QfaType', bound='QuantumFiniteAutomatonBase')


class QuantumFiniteAutomatonBase(abc.ABC):
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

    def __call__(self, w: list[int]) -> float:
        tape = [0] + w + [self.end_of_string]
        last_total_state = self.process(TotalState.initial(self.states), tape)
        _, acceptance, rejection = last_total_state.to_tuple()
        if not math.isclose(acceptance + rejection, 1):
            raise InvalidQuantumFiniteAutomatonError()
        return acceptance

    @abc.abstractmethod
    def process(self, total_state: TotalState, w: list[int]) -> TotalState:
        raise NotImplementedError()

    @abc.abstractmethod
    def step(self, total_state: TotalState, c: int) -> TotalState:
        raise NotImplementedError()

    @abc.abstractmethod
    def concatenation(self: QfaType, other: QfaType) -> QfaType:
        raise NotImplementedError()

    def __concat__(self: QfaType, other: QfaType) -> QfaType:
        return self.concatenation(other)

    @abc.abstractmethod
    def union(self: QfaType, other: QfaType) -> QfaType:
        raise NotImplementedError()

    def __or__(self: QfaType, other: QfaType) -> QfaType:
        """Returns the union of the quantum finite automaton.

        See union() for details.
        """
        return self.union(other)

    @abc.abstractmethod
    def intersection(self: QfaType, other: QfaType) -> QfaType:
        raise NotImplementedError()

    def __and__(self: QfaType, other: QfaType) -> QfaType:
        """Returns the intersection of the quantum finite automaton.

        See intersection() for details.
        """
        return self.intersection(other)

    def __multiply__(self: QfaType, other: QfaType) -> QfaType:
        """Returns the intersection of the quantum finite automaton.

        See intersection() for details.
        """
        return self.intersection(other)

    @abc.abstractmethod
    def complement(self: QfaType) -> QfaType:
        raise NotImplementedError()

    def __invert__(self: QfaType) -> QfaType:
        """Returns the complement of the quantum finite automaton.

        See complement() for details.
        """
        return self.complement()

    @abc.abstractmethod
    def difference(self: QfaType, other: QfaType) -> QfaType:
        raise NotImplementedError()

    def __sub__(self: QfaType, other: QfaType) -> QfaType:
        return self.difference(other)

    @abc.abstractmethod
    def equivalence(self: QfaType, other: QfaType) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def minimize(self: QfaType) -> QfaType:
        raise NotImplementedError()

    @abc.abstractmethod
    def symmetric_difference(self: QfaType, other: QfaType) -> QfaType:
        raise NotImplementedError()

    def __xor__(self: QfaType, other: QfaType) -> QfaType:
        return self.symmetric_difference(other)

    @abc.abstractmethod
    def kleene_star(self: QfaType) -> QfaType:
        raise NotImplementedError()

    @abc.abstractmethod
    def kleene_plus(self: QfaType) -> QfaType:
        raise NotImplementedError()

    @abc.abstractmethod
    def reverse(self: QfaType) -> QfaType:
        raise NotImplementedError()

    @abc.abstractmethod
    def is_empty(self: QfaType) -> bool:
        raise NotImplementedError()


class DynamicConfiguration(Generic[QfaType]):
    def __init__(self, qfa: QfaType, q: TotalState) -> None:
        pass

    def step(self, c: str) -> None:
        if len(c) != 1:
            raise ValueError("c must be a single character")
        pass

    def process(self, w: str) -> None:
        pass
