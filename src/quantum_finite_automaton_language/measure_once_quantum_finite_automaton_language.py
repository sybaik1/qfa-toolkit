from .quantum_finite_automaton_language import (
    QuantumFiniteAutomatonLanguage as QfaLanguage)
from quantum_finite_automaton import MeasureOnceQuantumFiniteAutomaton as Moqfa
from recognition_strategy import RecognitionStrategy


class MeasureOnceQuantumFiniteAutomatonLanguage(QfaLanguage):
    def __init__(self, moqfa: Moqfa, strategy: RecognitionStrategy) -> None:
        super().__init__(moqfa, strategy)
