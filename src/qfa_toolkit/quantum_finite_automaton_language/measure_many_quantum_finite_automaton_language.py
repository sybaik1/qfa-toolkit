from typing import (overload, Generic, TypeVar, )

from .quantum_finite_automaton_language import (
    QuantumFiniteAutomatonLanguage as Qfl)
from ..quantum_finite_automaton import (
    MeasureManyQuantumFiniteAutomaton as Mmqfa)
from ..recognition_strategy import RecognitionStrategy
from ..recognition_strategy import IsolatedCutPoint


_T = TypeVar('_T', bound=RecognitionStrategy)

Mmqfl = 'MeasureManyQuantumFiniteAutomatonLanguage'


class MeasureManyQuantumFiniteAutomatonLanguage(Qfl, Generic[_T]):
    def __init__(self, mmqfa: Mmqfa, strategy: RecognitionStrategy) -> None:
        super().__init__(mmqfa, strategy)

    def complement(self) -> Mmqfl:
        threshold = 1.0 - self.strategy.threshold
        epsilon = self.strategy.epsilon
        isolated_cut_point = IsolatedCutPoint(threshold, epsilon)
        return self.__class__(~self.qfa, isolated_cut_point)
