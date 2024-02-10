from __future__ import annotations

import abc
import math
from typing import (TypeVar, Union, Generic, )

import numpy as np
import numpy.typing as npt


Superposition = npt.NDArray[np.cdouble]


class InvalidQauntumFiniteAutomatonException(Exception):
    pass


class TotalState():

    def __init__(
        self,
        superposition_or_list: Union[Superposition, list[complex]],
        acceptance: float,
        rejection: float
    ) -> None:
        if isinstance(superposition_or_list, list):
            superposition = np.array(superposition_or_list, dtype=np.cdouble)
        elif isinstance(superposition_or_list, np.ndarray):
            superposition = superposition_or_list
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
    def initial(self, states: int) -> TotalState:
        superposition = np.zeros(states, dtype=np.cdouble)
        superposition[0] = 1
        return TotalState(superposition, 0, 0)

    def measure_by(
        self,
        observable: tuple[set[int], set[int], set[int]]
    ) -> TotalState:
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

    def apply(self, unitary: npt.NDArray[np.cdouble]) -> TotalState:
        superposition = unitary @ self.superposition
        return TotalState(superposition, self.acceptance, self.rejection)

    def to_tuple(self) -> tuple[Superposition, float, float]:
        return (self.superposition, self.acceptance, self.rejection)


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
    def concatination(
        self: QuantumFiniteAutomatonType,
        other: QuantumFiniteAutomatonType
    ) -> QuantumFiniteAutomatonType:
        raise NotImplementedError()

    @abc.abstractmethod
    def union(
        self: QuantumFiniteAutomatonType,
        other: QuantumFiniteAutomatonType
    ) -> QuantumFiniteAutomatonType:
        raise NotImplementedError()

    @abc.abstractmethod
    def intersection(
        self: QuantumFiniteAutomatonType,
        other: QuantumFiniteAutomatonType
    ) -> QuantumFiniteAutomatonType:
        raise NotImplementedError()

    @abc.abstractmethod
    def complement(
        self: QuantumFiniteAutomatonType
    ) -> QuantumFiniteAutomatonType:
        raise NotImplementedError()

    @abc.abstractmethod
    def difference(
        self: QuantumFiniteAutomatonType,
        other: QuantumFiniteAutomatonType
    ) -> QuantumFiniteAutomatonType:
        raise NotImplementedError()

    @abc.abstractmethod
    def equivalence(
        self: QuantumFiniteAutomatonType,
        other: QuantumFiniteAutomatonType
    ) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def minimize(
        self: QuantumFiniteAutomatonType
    ) -> QuantumFiniteAutomatonType:
        raise NotImplementedError()

    @abc.abstractmethod
    def symmetric_difference(
        self: QuantumFiniteAutomatonType, other: QuantumFiniteAutomatonType
    ) -> QuantumFiniteAutomatonType:
        raise NotImplementedError()

    @abc.abstractmethod
    def kleene_star(
        self: QuantumFiniteAutomatonType
    ) -> QuantumFiniteAutomatonType:
        raise NotImplementedError()

    @abc.abstractmethod
    def kleene_plus(
        self: QuantumFiniteAutomatonType
    ) -> QuantumFiniteAutomatonType:
        raise NotImplementedError()

    @abc.abstractmethod
    def reverse(
        self: QuantumFiniteAutomatonType
    ) -> QuantumFiniteAutomatonType:
        raise NotImplementedError()

    @abc.abstractmethod
    def is_empty(self: QuantumFiniteAutomatonType) -> bool:
        raise NotImplementedError()


QuantumFiniteAutomatonType = TypeVar(
    'QuantumFiniteAutomatonType', bound=QuantumFiniteAutomatonBase)


class DynamicConfiguration(Generic[QuantumFiniteAutomatonType]):
    def __init__(
        self,
        qfa: QuantumFiniteAutomatonType,
        q: TotalState
    ) -> None:
        pass

    def step(self, c: str) -> None:
        if len(c) != 1:
            raise ValueError("c must be a single character")
        pass

    def process(self, w: str) -> None:
        pass
