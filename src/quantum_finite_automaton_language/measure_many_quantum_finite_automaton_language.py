from .quantum_finite_automaton_language import (
    QuantumFiniteAutomatonLanguage as QfaLanguage)
from quantum_finite_automaton import MeasureManyQuantumFiniteAutomaton as Mmqfa
from recognition_strategy import RecognitionStrategy


class MeasureManyQuantumFiniteAutomatonLanguage(QfaLanguage):
    def __init__(self, mmqfa: Mmqfa, strategy: RecognitionStrategy) -> None:
        super().__init__(mmqfa, strategy)
