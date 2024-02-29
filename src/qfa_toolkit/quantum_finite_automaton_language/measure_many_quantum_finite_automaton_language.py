from typing import (TypeVar, )

from .quantum_finite_automaton_language import (
    QuantumFiniteAutomatonLanguage as Qfl)
from ..quantum_finite_automaton import (
    MeasureManyQuantumFiniteAutomaton as Mmqfa)
from ..recognition_strategy import RecognitionStrategy
from ..recognition_strategy import IsolatedCutPoint


TMmqfl = TypeVar('TMmqfl', bound='MeasureManyQuantumFiniteAutomatonLanguage')
TRecognitionStrategy = TypeVar(
    'TRecognitionStrategy', bound=RecognitionStrategy)


class MeasureManyQuantumFiniteAutomatonLanguage(Qfl[TRecognitionStrategy]):
    def __init__(self, mmqfa: Mmqfa, strategy: TRecognitionStrategy) -> None:
        self.qfa: Mmqfa
        super().__init__(mmqfa, strategy)

    def complement(self: TMmqfl) -> TMmqfl:
        if not isinstance(self.strategy, IsolatedCutPoint):
            raise ValueError(
                "Complement is only defined for IsolatedCutPoint strategies")
        threshold = 1.0 - self.strategy.threshold
        epsilon = self.strategy.epsilon
        isolated_cut_point = IsolatedCutPoint(threshold, epsilon)
        return self.__class__(~self.qfa, isolated_cut_point)
