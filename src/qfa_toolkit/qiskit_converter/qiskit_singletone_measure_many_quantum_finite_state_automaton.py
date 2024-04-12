from functools import reduce

from ..quantum_finite_state_automaton.singleton_measure_many_quantum_finite_state_automaton import (
    SingletoneMeasureManyQuantumFiniteStateAutomaton)
from .qiskit_measure_many_quantum_finite_state_automaton import (
    QiskitMeasureManyQuantumFiniteStateAutomaton)


class QiskitSingletoneMeasureManyQuantumFiniteStateAutomaton(
        QiskitMeasureManyQuantumFiniteStateAutomaton):
    def __init__(self, n: list[int]):
        mmqfa = reduce(lambda x, y: x.__or__(y),
                       [SingletoneMeasureManyQuantumFiniteStateAutomaton(n[i])
                        for i in range(1, len(n))],
                       SingletoneMeasureManyQuantumFiniteStateAutomaton(n[0]))
        super().__init__(mmqfa)
