from typing import (Iterator, TypeVar, Generic, )

from ..quantum_finite_automaton import QuantumFiniteAutomatonBase
from ..recognition_strategy import RecognitionStrategy
from .utils import iterate_every_string
from .utils import iterate_length_n_strings
from .utils import iterate_length_less_than_n_strings

TQfl = TypeVar('TQfl', bound='QuantumFiniteAutomatonLanguage')
TRecognitionStrategy = TypeVar(
    'TRecognitionStrategy', bound=RecognitionStrategy)


class QuantumFiniteAutomatonLanguage(Generic[TRecognitionStrategy]):
    def __init__(
        self,
        qfa: QuantumFiniteAutomatonBase,
        strategy: TRecognitionStrategy,
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

    def enumerate_length_less_than_n(self, n: int) -> Iterator[list[int]]:
        length_less_than_n_strings = (
            iterate_length_less_than_n_strings(self.alphabet, n))
        return filter(lambda w: w in self, length_less_than_n_strings)

    def enumerate_length_n(self, n: int) -> Iterator[list[int]]:
        length_n_string = iterate_length_n_strings(self.alphabet, n)
        return filter(lambda w: w in self, length_n_string)

    def enumerate(self) -> Iterator[list[int]]:
        every_string = iterate_every_string(self.alphabet)
        return filter(lambda w: w in self, every_string)

    def concatination(self: TQfl, other: TQfl) -> TQfl:
        raise NotImplementedError()

    def union(self: TQfl, other: TQfl) -> TQfl:
        raise NotImplementedError()

    def intersection(self: TQfl, other: TQfl) -> TQfl:
        raise NotImplementedError()

    def __invert__(self: TQfl) -> TQfl:
        return self.complement()

    def complement(self: TQfl) -> TQfl:
        raise NotImplementedError()

    def difference(self: TQfl, other: TQfl) -> TQfl:
        raise NotImplementedError()

    def equivalence(self: TQfl, other: TQfl) -> bool:
        raise NotImplementedError()

    def kleene_star(self: TQfl) -> TQfl:
        raise NotImplementedError()

    def reverse(self: TQfl) -> TQfl:
        raise NotImplementedError()

    def is_empty(self: TQfl) -> bool:
        raise NotImplementedError()
