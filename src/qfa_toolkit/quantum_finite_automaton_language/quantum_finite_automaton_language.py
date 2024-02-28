import itertools
from typing import (Iterator, )

from ..quantum_finite_automaton import QuantumFiniteAutomatonBase
from ..recognition_strategy import RecognitionStrategy


Qfl = 'QuantumFiniteAutomatonLanguage'


def _iterate_length_n_strings(alphabet: int, n: int) -> Iterator[list[int]]:
    return map(list, itertools.product(range(1, alphabet+1), repeat=n))


def _iterate_length_less_than_n_strings(
    alphabet: int, n: int
) -> Iterator[list[int]]:
    count = range(n)
    iterables = map(lambda n: _iterate_length_n_strings(alphabet, n), count)
    return itertools.chain.from_iterable(iterables)


def _iterate_every_string(alphabet: int) -> Iterator[list[int]]:
    count = itertools.count()
    iterables = map(lambda n: _iterate_length_n_strings(alphabet, n), count)
    return itertools.chain.from_iterable(iterables)


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

    def enumerate_length_less_than_n(self, n: int) -> Iterator[list[int]]:
        length_less_than_n_strings = (
            _iterate_length_less_than_n_strings(self.alphabet, n))
        return filter(lambda w: w in self, length_less_than_n_strings)

    def enumerate_length_n(self, n: int) -> Iterator[list[int]]:
        length_n_string = _iterate_length_n_strings(self.alphabet, n)
        return filter(lambda w: w in self, length_n_string)

    def enumerate(self) -> Iterator[list[int]]:
        every_string = _iterate_every_string(self.alphabet)
        return filter(lambda w: w in self, every_string)

    def concatination(self: Qfl, other: Qfl) -> Qfl:
        raise NotImplementedError()

    def union(self: Qfl, other: Qfl) -> Qfl:
        raise NotImplementedError()

    def intersection(self: Qfl, other: Qfl) -> Qfl:
        raise NotImplementedError()

    def __invert__(self: Qfl) -> Qfl:
        return self.complement()

    def complement(self: Qfl) -> Qfl:
        raise NotImplementedError()

    def difference(self: Qfl, other: Qfl) -> Qfl:
        raise NotImplementedError()

    def equivalence(self: Qfl, other: Qfl) -> bool:
        raise NotImplementedError()

    def kleene_star(self: Qfl) -> Qfl:
        raise NotImplementedError()

    def reverse(self: Qfl) -> Qfl:
        raise NotImplementedError()

    def is_empty(self: Qfl) -> bool:
        raise NotImplementedError()
