import numpy as np
from qiskit.circuit import QuantumCircuit  # type: ignore
from qiskit import QuantumRegister, ClassicalRegister  # type: ignore

from .qiskit_base import QiskitQuantumFiniteAutomaton
from ..quantum_finite_automaton import MeasureOnceQuantumFiniteAutomaton

Moqfa = MeasureOnceQuantumFiniteAutomaton
QMoqfa = 'QiskitMeasureOnceQuantumFiniteAutomaton'


class QiskitMeasureOnceQuantumFiniteAutomaton(QiskitQuantumFiniteAutomaton):
    def __init__(self, qfa: Moqfa):
        self.qfa = qfa
        self.get_size()
        self.get_mapping()
        self.transitions_to_circuit(qfa.transitions)

    def get_mapping(self):
        super().get_mapping()

        self.accepting_states = set(np.flatnonzero(self.qfa.accepting_states))
        self.rejecting_states = set(np.flatnonzero(self.qfa.rejecting_states))

    def get_circuit_for_string(self, w: list[int]):
        qreg_states = QuantumRegister(self.size, 'q_state')
        creg_states = ClassicalRegister(self.size, 'c_state')

        circuit = QuantumCircuit(qreg_states, creg_states)

        circuit.append(self.circuit[self.qfa.start_of_string], qreg_states)
        for c in [self.qfa.start_of_string] + w + [self.qfa.end_of_string]:
            circuit.append(self.circuit[c], qreg_states)
        circuit.measure_all(add_bits=False)
        return circuit
