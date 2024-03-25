from typing import (Generic, TypeVar, )

from .quantum_finite_automaton_language_base import (
    QuantumFiniteAutomatonLanguageBase as Qfl)
from ..quantum_finite_automaton import (
    MeasureOnceQuantumFiniteAutomaton as Moqfa)
from ..recognition_strategy import RecognitionStrategy


MoqflT = TypeVar('MoqflT', bound='MeasureOnceQuantumFiniteAutomatonLanguage')
RecognitionStrategyT = TypeVar(
    'RecognitionStrategyT', bound=RecognitionStrategy)


class MeasureOnceQuantumFiniteAutomatonLanguage(
    Qfl[Moqfa, RecognitionStrategyT],
    Generic[RecognitionStrategyT]
):

    def concatenation(self, other: MoqflT) -> MoqflT:
        raise NotImplementedError()

    def union(self, other: MoqflT) -> MoqflT:
        raise NotImplementedError()

    def intersection(self, other: MoqflT) -> MoqflT:
        raise NotImplementedError()
