from typing import (Generic, TypeVar, )

from .quantum_finite_automaton_language_base import (
    QuantumFiniteAutomatonLanguageBase as Qfl)
from ..quantum_finite_automaton import (
    MeasureOnceQuantumFiniteAutomaton as Moqfa)
from ..recognition_strategy import RecognitionStrategy


TMoqfl = TypeVar('TMoqfl', bound='MeasureOnceQuantumFiniteAutomatonLanguage')
TRecognitionStrategy = TypeVar(
    'TRecognitionStrategy', bound=RecognitionStrategy)


class MeasureOnceQuantumFiniteAutomatonLanguage(
    Qfl[Moqfa, TRecognitionStrategy],
    Generic[TRecognitionStrategy]
):

    def concatenation(self, other: TMoqfl) -> TMoqfl:
        raise NotImplementedError()

    def union(self, other: TMoqfl) -> TMoqfl:
        raise NotImplementedError()

    def intersection(self, other: TMoqfl) -> TMoqfl:
        raise NotImplementedError()
