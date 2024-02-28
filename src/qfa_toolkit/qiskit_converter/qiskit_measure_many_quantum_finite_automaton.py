import math

import numpy as np
import numpy.typing as npt
from qiskit.circuit import QuantumCircuit  # type: ignore
from qiskit import QuantumRegister, ClassicalRegister  # type: ignore

from .qiskit_base import QiskitQuantumFiniteAutomaton
from ..quantum_finite_automaton import (
        MeasureManyQuantumFiniteAutomaton as Mmqfa)


class QiskitMeasureManyQuantumFiniteAutomaton(QiskitQuantumFiniteAutomaton):
    def __init__(self, qfa: Mmqfa):
        """
        -- Properties --
        qfa: Mmqfa
        accepting_states: set[int]
        rejecting_states: set[int]
        circuit: list[QuantumCircuit] # indexed by the alphabet
        size: int                     # size of the qubit register for states
        mapping: dict[int, int]       # mapping from qubit register index
                                        to qfa state
        """
        self.qfa: Mmqfa = qfa
        self.accepting_states: set[int] = qfa.accepting_states
        self.rejecting_states: set[int] = qfa.rejecting_states
        self.non_halting_states: set[int] = self.qfa.non_halting_states
        self.mapping: dict[int, int] = {}
        self.circuit = self.transitions_to_circuit(qfa.transition)

    @property
    def halting_states(self):
        return self.accepting_states | self.rejecting_states

    def transitions_to_circuit(self, transitions: npt.NDArray[np.cdouble]):
        self.size = 1 + math.ceil(
                math.log2(
                    max(len(self.halting_states),
                        len(self.non_halting_states))))

        # Create a mapping from the qubit register index to the qfa state
        non_s_index = 0
        non_f_index = len(self.non_halting_states)
        non_index = list(self.non_halting_states)
        halt_s_index = 2 ** (self.size - 1)
        halt_f_index = 2 ** (self.size - 1) + len(self.halting_states)
        halt_index = list(self.halting_states)

        circit_accepting_states: set[int] = set()
        circit_rejecting_states: set[int] = set()
        for index, state in enumerate(non_index):
            self.mapping[state] = index
        for index, state in enumerate(halt_index):
            self.mapping[state] = index + halt_s_index
            if state in self.accepting_states:
                circit_accepting_states.add(index + halt_s_index)
            else:
                circit_rejecting_states.add(index + halt_s_index)

        self.accepting_states = circit_accepting_states
        self.rejecting_states = circit_rejecting_states

        circuit_unitary = []
        for trans in transitions:
            circuit_trans = np.eye(2 ** self.size, dtype=complex)
            circuit_trans[non_s_index:non_f_index,
                          non_s_index:non_f_index] = (
                                  trans[non_index][:, non_index])
            circuit_trans[halt_s_index:halt_f_index,
                          non_s_index:non_f_index] = (
                                  trans[halt_index][:, non_index])
            circuit_trans[halt_s_index:halt_f_index,
                          halt_s_index:halt_f_index] = (
                                  trans[halt_index][:, halt_index])
            circuit_trans[non_s_index:non_f_index,
                          halt_s_index:halt_f_index] = (
                                  trans[non_index][:, halt_index])

            circuit_unitary.append(circuit_trans)

        return super().transitions_to_circuit(circuit_unitary)

    def get_circuit_for_string(self, w: list[int]):
        qreg_q = QuantumRegister(self.size, 'q_state')
        creg_c = ClassicalRegister(self.size - 1, 'c_state')
        creg_m = ClassicalRegister(1, 'halt')

        circuit = QuantumCircuit(qreg_q, creg_c, creg_m)

        circuit.append(self.circuit[0], qreg_q)
        circuit.measure(qreg_q[-1], creg_m)
        circuit.measure(qreg_q[:-1], creg_c).c_if(creg_m, 1)
        for c in w:
            circuit.append(self.circuit[c], qreg_q).c_if(creg_m, 0)
            circuit.measure(qreg_q[-1], creg_m)
            circuit.measure(qreg_q[:-1], creg_c).c_if(creg_m, 1)
        circuit.append(self.circuit[-1], qreg_q).c_if(creg_m, 0)
        circuit.measure(qreg_q[:-1], creg_c)
        circuit.measure(qreg_q[-1], creg_m)
        return circuit
