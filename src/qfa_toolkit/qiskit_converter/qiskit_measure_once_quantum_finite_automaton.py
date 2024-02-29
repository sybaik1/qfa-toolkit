from .qiskit_base import QiskitQuantumFiniteAutomaton
from ..quantum_finite_automaton import MeasureOnceQuantumFiniteAutomaton

Moqfa = MeasureOnceQuantumFiniteAutomaton
QMoqfa = 'QiskitMeasureOnceQuantumFiniteAutomaton'


class QiskitMeasureOnceQuantumFiniteAutomaton(QiskitQuantumFiniteAutomaton):
    def __init__(self, qfa: Moqfa):
        super().__init__(qfa)
