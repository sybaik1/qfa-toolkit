import abc
import math
from typing import (TypeVar, Union, Generic, )

import numpy as np
import numpy.typing as npt


Superposition = npt.NDArray[np.cdouble]
Observable = tuple[set[int], set[int], set[int]]


class InvalidQauntumFiniteAutomatonException(Exception):
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

    def measure_by(
        self,
        observable: Observable,
    ) -> 'TotalState':
        acceptings, rejectings, non_haltings = observable

        def projection(
            superposition: Superposition,
            axis: set[int]
        ) -> Superposition:
            mask = np.zeros(self.superposition.shape, dtype=np.cdouble)
            mask[list(axis)] = 1
            return self.superposition * mask

        acccepting_superposition = projection(self.superposition, acceptings)
        delta_acceptance = np.linalg.norm(acccepting_superposition) ** 2
        assert isinstance(delta_acceptance, float)
        acceptance = self.acceptance + delta_acceptance

        rejecting_superposition = projection(self.superposition, rejectings)
        delta_rejection = np.linalg.norm(rejecting_superposition) ** 2
        assert isinstance(delta_rejection, float)
        rejection = self.rejection + delta_rejection

        superposition = projection(self.superposition, non_haltings)
        return TotalState(superposition, acceptance, rejection)

    def apply(self, unitary: npt.NDArray[np.cdouble]) -> 'TotalState':
        superposition = unitary @ self.superposition
        return TotalState(superposition, self.acceptance, self.rejection)

    def to_tuple(self) -> tuple[Superposition, float, float]:
        return (self.superposition, self.acceptance, self.rejection)


T = TypeVar('T', bound='QuantumFiniteAutomatonBase')


class QuantumFiniteAutomatonBase(abc.ABC):
    start_of_string: int = 0

    def __init__(self, transition: Superposition) -> None:
        if len(transition.shape) != 3:
            raise ValueError("Transition matrix must be 3-dimensional")
        if transition.shape[0] < 3:
            raise ValueError("Transition matrix must have at least 3 rows")
        if transition.shape[1] != transition.shape[2]:
            raise ValueError(
                "Each component of transition matrix must be square")
        identity = np.eye(transition.shape[1])
        for u in transition:
            if not np.allclose(identity, u.dot(u.T.conj())):
                raise ValueError(
                    "Each component of transition matrix must be unitary")
        self.transition = transition

    @property
    def alphabet(self) -> int:
        return self.transition.shape[0] - 2

    @property
    def states(self) -> int:
        return self.transition.shape[1]

    @property
    def end_of_string(self) -> int:
        return self.alphabet + 1

    @abc.abstractmethod
    def process(self, total_state: TotalState, w: list[int]) -> TotalState:
        raise NotImplementedError()

    @abc.abstractmethod
    def step(self, total_state: TotalState, c: int) -> TotalState:
        raise NotImplementedError()

    @abc.abstractmethod
    def __call__(self, w: list[int]) -> float:
        raise NotImplementedError()

    @abc.abstractmethod
    def concatination(self: T, other: T) -> T:
        raise NotImplementedError()

    def __concat__(self: T, other: T) -> T:
        return self.concatination(other)

    @abc.abstractmethod
    def union(self: T, other: T) -> T:
        raise NotImplementedError()

    def __or__(self: T, other: T) -> T:
        return self.union(other)

    @abc.abstractmethod
    def intersection(self: T, other: T) -> T:
        raise NotImplementedError()

    def __and__(self: T, other: T) -> T:
        return self.intersection(other)

    @abc.abstractmethod
    def complement(self: T) -> T:
        raise NotImplementedError()

    def __not__(self: T) -> T:
        return self.complement()

    @abc.abstractmethod
    def difference(self: T, other: T) -> T:
        raise NotImplementedError()

    def __sub__(self: T, other: T) -> T:
        return self.difference(other)

    @abc.abstractmethod
    def equivalence(self: T, other: T) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def minimize(self: T) -> T:
        raise NotImplementedError()

    @abc.abstractmethod
    def symmetric_difference(self: T, other: T) -> T:
        raise NotImplementedError()

    def __xor__(self: T, other: T) -> T:
        return self.symmetric_difference(other)

    @abc.abstractmethod
    def kleene_star(self: T) -> T:
        raise NotImplementedError()

    @abc.abstractmethod
    def kleene_plus(self: T) -> T:
        raise NotImplementedError()

    @abc.abstractmethod
    def reverse(self: T) -> T:
        raise NotImplementedError()

    @abc.abstractmethod
    def is_empty(self: T) -> bool:
        raise NotImplementedError()


class DynamicConfiguration(Generic[T]):
    def __init__(self, qfa: T, q: TotalState) -> None:
        pass

    def step(self, c: str) -> None:
        if len(c) != 1:
            raise ValueError("c must be a single character")
        pass

    def process(self, w: str) -> None:
        pass
