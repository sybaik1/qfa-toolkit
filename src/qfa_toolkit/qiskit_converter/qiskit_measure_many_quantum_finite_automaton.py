import math

import numpy as np
import itertools as it
from qiskit.circuit import QuantumCircuit  # type: ignore
from qiskit import QuantumRegister, ClassicalRegister  # type: ignore

from .qiskit_base import QiskitQuantumFiniteAutomaton
from ..quantum_finite_automaton import (
    MeasureManyQuantumFiniteAutomaton as Mmqfa)


class QiskitMeasureManyQuantumFiniteAutomaton(QiskitQuantumFiniteAutomaton):
    def __init__(self, qfa: Mmqfa, use_entropy_mapping: bool = True):
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
        self.get_size()
        self.mapping = {k: k for k in range(2 ** self.size)}
        if use_entropy_mapping:
            self.get_entropy_mapping()
        else:
            self.get_mapping()
        self.transitions_to_circuit(qfa.transitions)

    @property
    def halting_states(self):
        return self.accepting_states | self.rejecting_states

    def get_size(self):
        self.size = 1 + math.ceil(
            math.log2(
                max(len(np.flatnonzero(self.qfa.halting_states)),
                    len(np.flatnonzero(self.qfa.non_halting_states)))))

    def get_entropy_mapping(self):
        state_order = [
                int('1'+''.join(map(str, tp)), base=2)
                for tp in sorted(it.product([0, 1], repeat=self.size-1))]
        new_mapping = dict()
        # non-halting states mapped with the first qubit as 0
        for index, state in enumerate(np.flatnonzero(
                self.qfa.non_halting_states)):
            new_mapping[state] = index

        # halting states
        # bits with less 1s
        for index, state in enumerate(np.flatnonzero(
                self.qfa.accepting_states)):
            new_mapping[state] = state_order[index]
        # bits with more 1s
        for index, state in enumerate(np.flatnonzero(
                self.qfa.rejecting_states)):
            new_mapping[state] = state_order[-index-1]

        half_size = 2 ** (self.size - 1)
        for index, state in enumerate(self.undefined_states):
            if index + len(np.flatnonzero(
                    self.qfa.non_halting_states)) < half_size:
                new_mapping[state] = (index
                                      + len(np.flatnonzero(
                                          self.qfa.non_halting_states)))
            else:
                new_mapping[state] = (index
                                      + len(np.flatnonzero(
                                          self.qfa.accepting_states))
                                      + half_size)

        # State status mapping
        self.mapping = new_mapping
        self.accepting_states = {self.mapping[state] for state in
                                 set(np.flatnonzero(
                                     self.qfa.accepting_states))}
        self.rejecting_states = {self.mapping[state] for state in
                                 set(np.flatnonzero(
                                     self.qfa.rejecting_states))}
        self.non_halting_states = {self.mapping[state] for state in
                                   set(np.flatnonzero(
                                       self.qfa.non_halting_states))}

    def get_mapping(self):
        # Create a mapping from the qubit register index to the qfa state
        half_size = 2 ** (self.size - 1)
        new_mapping = dict()
        for index, state in enumerate(np.flatnonzero(
                self.qfa.non_halting_states)):
            new_mapping[state] = index
        for index, state in enumerate(np.flatnonzero(
                self.qfa.halting_states)):
            new_mapping[state] = index + half_size
        for index, state in enumerate(self.undefined_states):
            if index + len(np.flatnonzero(
                    self.qfa.non_halting_states)) < half_size:
                new_mapping[state] = (index
                                      + len(np.flatnonzero(
                                          self.qfa.non_halting_states)))
            else:
                new_mapping[state] = (index
                                      + len(np.flatnonzero(
                                          self.qfa.halting_states))
                                      + half_size)

        # State status mapping
        self.mapping = new_mapping
        self.accepting_states = {self.mapping[state] for state in
                                 set(np.flatnonzero(
                                     self.qfa.accepting_states))}
        self.rejecting_states = {self.mapping[state] for state in
                                 set(np.flatnonzero(
                                     self.qfa.rejecting_states))}
        self.non_halting_states = {self.mapping[state] for state in
                                   set(np.flatnonzero(
                                       self.qfa.non_halting_states))}

    def get_circuit_for_string(self, w: list[int]):
        qreg_states = QuantumRegister(self.size, 'q_state')
        creg_states = ClassicalRegister(self.size, 'c_state')

        circuit = QuantumCircuit(qreg_states, creg_states)

        circuit.append(self.circuit[self.qfa.start_of_string], qreg_states)
        for c in w + [self.qfa.end_of_string]:
            circuit.measure(qreg_states[-1], creg_states[-1])
            circuit.append(
                self.circuit[c], qreg_states).c_if(creg_states[-1], 0)
        circuit.measure_all(add_bits=False)
        return circuit
