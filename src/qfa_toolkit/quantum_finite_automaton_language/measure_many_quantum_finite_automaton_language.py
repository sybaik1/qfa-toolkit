from typing import (overload, Generic, TypeVar, )

from .quantum_finite_automaton_language import (
    QuantumFiniteAutomatonLanguage as Qfl)
from ..quantum_finite_automaton import (
    MeasureManyQuantumFiniteAutomaton as Mmqfa)
from ..recognition_strategy import RecognitionStrategy


_T = TypeVar('_T', bound=RecognitionStrategy)

Mmqfl = 'MeasureManyQuantumFiniteAutomatonLanguage'


class MeasureManyQuantumFiniteAutomatonLanguage(Qfl, Generic[_T]):
    def __init__(self, mmqfa: Mmqfa, strategy: RecognitionStrategy) -> None:
        super().__init__(mmqfa, strategy)
