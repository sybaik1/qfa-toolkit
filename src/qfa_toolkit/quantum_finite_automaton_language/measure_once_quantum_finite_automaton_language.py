from typing import (overload, Generic, TypeVar, )

from .quantum_finite_automaton_language import (
    QuantumFiniteAutomatonLanguage as Qfl)
from ..quantum_finite_automaton import (
    MeasureOnceQuantumFiniteAutomaton as Moqfa)
from ..recognition_strategy import RecognitionStrategy
from ..recognition_strategy import IsolatedCutPoint


Moqfl = 'MeasureOnceQuantumFiniteAutomatonLanguage'

_T = TypeVar('_T', bound=RecognitionStrategy)


class MeasureOnceQuantumFiniteAutomatonLanguage(Qfl, Generic[_T]):
    def __init__(self, moqfa: Moqfa, strategy: _T) -> None:
        super().__init__(moqfa, strategy)

    def concatenation(self, other: Moqfl) -> Moqfl:
        raise NotImplementedError()

    def union(self, other: Moqfl) -> Moqfl:
        raise NotImplementedError()

    def intersection(self, other: Moqfa) -> Moqfa:
        raise NotImplementedError()
