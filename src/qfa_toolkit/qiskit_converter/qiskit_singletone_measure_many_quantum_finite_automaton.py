from functools import reduce

from ..quantum_finite_automaton.singleton_measure_many_quantum_finite_automaton import (
    SingletoneMeasureManyQuantumFiniteAutomaton)
from .qiskit_measure_many_quantum_finite_automaton import (
    QiskitMeasureManyQuantumFiniteAutomaton)


class QiskitSingletoneMeasureManyQuantumFiniteAutomaton(
        QiskitMeasureManyQuantumFiniteAutomaton):
    def __init__(self, n: list[int]):
        mmqfa = reduce(lambda x, y: x.__or__(y),
                       [SingletoneMeasureManyQuantumFiniteAutomaton(n[i])
                        for i in range(1, len(n))],
                       SingletoneMeasureManyQuantumFiniteAutomaton(n[0]))
        super().__init__(mmqfa)
