import itertools
import abc
from typing import (Iterator, TypeVar, Generic, )

from ..quantum_finite_state_automaton import QuantumFiniteStateAutomatonBase
from ..recognition_strategy import RecognitionStrategy

QflT = TypeVar('QflT', bound='QuantumFiniteStateAutomatonLanguageBase')
QfaT = TypeVar('QfaT', bound=QuantumFiniteStateAutomatonBase)
RecognitionStrategyT = TypeVar(
    'RecognitionStrategyT', bound=RecognitionStrategy)


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


class QuantumFiniteStateAutomatonLanguageBase(
    abc.ABC, Generic[QfaT, RecognitionStrategyT]
):
    def __init__(
        self,
        quantum_finite_state_automaton: QfaT,
        strategy: RecognitionStrategyT
    ) -> None:
        self.quantum_finite_state_automaton = quantum_finite_state_automaton
        self.strategy = strategy

    @property
    def alphabet(self) -> int:
        return self.quantum_finite_state_automaton.alphabet

    @property
    def start_of_string(self) -> int:
        return self.quantum_finite_state_automaton.start_of_string

    @property
    def end_of_string(self) -> int:
        return self.quantum_finite_state_automaton.end_of_string

    def __contains__(self, w: list[int]) -> bool:
        probability = self.quantum_finite_state_automaton(w)
        result = self.strategy(probability)
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
