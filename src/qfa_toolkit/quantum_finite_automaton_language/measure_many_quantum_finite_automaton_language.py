from .quantum_finite_automaton_language import (
    QuantumFiniteAutomatonLanguage as Qfl)
from ..quantum_finite_automaton import (
    MeasureManyQuantumFiniteAutomaton as Moqfa)
from ..recognition_strategy import RecognitionStrategy


class MeasureManyQuantumFiniteAutomatonLanguage(Qfl):
    def __init__(self, mmqfa: Moqfa, strategy: RecognitionStrategy) -> None:
        super().__init__(mmqfa, strategy)
