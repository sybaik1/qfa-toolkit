from __future__ import annotations
from abc import (ABC, abstractmethod, )
import numpy as np
import numpy.typing as npt


class QuantumState:
    def __init__(self) -> None:
        pass


class QuantumFiniteAutomatonBase(ABC):
    _alphabet: set
    _statesname: list
    # _transition: np.array(dim=(len(alphabet), len(statesname), len(statesname))
    _transition: npt.NDArray[np.cdouble]

    @abstractmethod
    def step(self, psi: QuantumState, string: str) -> QuantumState:
        raise NotImplementedError()

    @abstractmethod
    def process(self, psi: QuantumState, w: str) -> QuantumState:
        raise NotImplementedError()

    @abstractmethod
    def test(self, w: str) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def concatination(self, other: QuantumFiniteAutomatonBase) \
            -> QuantumFiniteAutomatonBase:
        raise NotImplementedError()

    @abstractmethod
    def union(self, other: QuantumFiniteAutomatonBase) \
            -> QuantumFiniteAutomatonBase:
        raise NotImplementedError()

    @abstractmethod
    def intersection(self, other: QuantumFiniteAutomatonBase) \
            -> QuantumFiniteAutomatonBase:
        raise NotImplementedError()

    @abstractmethod
    def complement(self) -> QuantumFiniteAutomatonBase:
        raise NotImplementedError()

    @abstractmethod
    def difference(self, other: QuantumFiniteAutomatonBase) \
            -> QuantumFiniteAutomatonBase:
        raise NotImplementedError()

    @abstractmethod
    def equivalence(self, other: QuantumFiniteAutomatonBase) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def minimize(self) -> QuantumFiniteAutomatonBase:
        raise NotImplementedError()

    @abstractmethod
    def symmetric_difference(self, other: QuantumFiniteAutomatonBase) \
            -> QuantumFiniteAutomatonBase:
        raise NotImplementedError()

    @abstractmethod
    def kleene_star(self) -> QuantumFiniteAutomatonBase:
        raise NotImplementedError()

    @abstractmethod
    def kleene_plus(self) -> QuantumFiniteAutomatonBase:
        raise NotImplementedError()

    @abstractmethod
    def reverse(self) -> QuantumFiniteAutomatonBase:
        raise NotImplementedError()

    @abstractmethod
    def is_empty(self) -> bool:
        raise NotImplementedError()


class DynamicConfiguration:
    def __init__(
        self,
        qfa: QuantumFiniteAutomatonBase,
        q: QuantumState
    ) -> None:
        pass

    def step(self, c: str) -> None:
        if len(c) != 1:
            raise ValueError("c must be a single character")
        pass

    def process(self, w: str) -> None:
        pass
