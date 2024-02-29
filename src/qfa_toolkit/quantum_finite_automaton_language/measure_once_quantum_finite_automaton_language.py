from typing import (Generic, TypeVar, )

from .quantum_finite_automaton_language import (
    QuantumFiniteAutomatonLanguage as Qfl)
from ..quantum_finite_automaton import (
    MeasureOnceQuantumFiniteAutomaton as Moqfa)
from ..recognition_strategy import RecognitionStrategy


TMoqfl = TypeVar('TMoqfl', bound='MeasureOnceQuantumFiniteAutomatonLanguage')
TRecognitionStrategy = TypeVar(
    'TRecognitionStrategy', bound=RecognitionStrategy)


class MeasureOnceQuantumFiniteAutomatonLanguage(Qfl[TRecognitionStrategy]):
    def __init__(self, moqfa: Moqfa, strategy: TRecognitionStrategy) -> None:
        super().__init__(moqfa, strategy)

    def concatenation(self, other: TMoqfl) -> TMoqfl:
        raise NotImplementedError()

    def union(self, other: TMoqfl) -> TMoqfl:
        raise NotImplementedError()

    def intersection(self, other: TMoqfl) -> TMoqfl:
        raise NotImplementedError()
