from .quantum_finite_automaton_language_base import (
    QuantumFiniteAutomatonLanguageBase as QfaLangBase)
from quantum_finite_automaton import MeasureManyQuantumFiniteAutomaton as Mmqfa
from recognition_strategy import RecognitionStrategy


class MeasureManyQuantumFiniteAutomatonLanguage(QfaLangBase):
    def __init__(self, mmqfa: Mmqfa, strategy: RecognitionStrategy) -> None:
        super().__init__(mmqfa, strategy)
