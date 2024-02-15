from typing import (TypeVar, )

from ..quantum_finite_automaton import QuantumFiniteAutomatonBase
from ..recognition_strategy import RecognitionStrategy


T = TypeVar('T', bound='QuantumFiniteAutomatonLanguage')


class QuantumFiniteAutomatonLanguage():
    def __init__(
        self,
        qfa: QuantumFiniteAutomatonBase,
        strategy: RecognitionStrategy,
    ) -> None:
        self.qfa = qfa
        self.strategy = strategy

    @property
    def alphabet(self) -> int:
        return self.qfa.alphabet

    @property
    def start_of_string(self) -> int:
        return self.qfa.start_of_string

    @property
    def end_of_string(self) -> int:
        return self.qfa.end_of_string

    def __contains__(self, w: list[int]) -> bool:
        result = self.strategy(self.qfa(w))
        if result == RecognitionStrategy.Result.INVALID:
            raise ValueError("Invalid result from recognition strategy")
        return result == RecognitionStrategy.Result.ACCEPT

    def concatination(self: T, other: T) -> T:
        raise NotImplementedError()

    def union(self: T, other: T) -> T:
        raise NotImplementedError()

    def intersection(self: T, other: T) -> T:
        raise NotImplementedError()

    def complement(self: T) -> T:
        raise NotImplementedError()

    def difference(self: T, other: T) -> T:
        raise NotImplementedError()

    def equivalence(self: T, other: T) -> bool:
        raise NotImplementedError()

    def kleene_star(self: T) -> T:
        raise NotImplementedError()

    def kleene_plus(self: T) -> T:
        raise NotImplementedError()

    def reverse(self: T) -> T:
        raise NotImplementedError()

    def is_empty(self: T) -> bool:
        raise NotImplementedError()
