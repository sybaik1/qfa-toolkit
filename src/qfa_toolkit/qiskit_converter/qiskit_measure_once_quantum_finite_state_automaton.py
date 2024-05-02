import numpy as np
import itertools as it
from qiskit.circuit import QuantumCircuit  # type: ignore
from qiskit import QuantumRegister, ClassicalRegister  # type: ignore

from .qiskit_base import QiskitQuantumFiniteStateAutomaton
from ..quantum_finite_state_automaton import MeasureOnceQuantumFiniteStateAutomaton

Moqfa = MeasureOnceQuantumFiniteStateAutomaton
QMoqfa = 'QiskitMeasureOnceQuantumFiniteStateAutomaton'


class QiskitMeasureOnceQuantumFiniteStateAutomaton(
    QiskitQuantumFiniteStateAutomaton
):
    def __init__(self, qfa: Moqfa, use_mapping_type: int = 0):
        self.qfa: Moqfa = qfa
        self.get_size()
        self.mapping = {k: k for k in range(2 ** self.size)}

        # temporary fix TODO FIXME
        if type(use_mapping_type) == bool:
            use_mapping_type = 1 if use_mapping_type else 0

        if use_mapping_type == 0:
            self.get_mapping()
        elif use_mapping_type == 1:
            self.get_entropy_mapping()
        elif use_mapping_type == 2:
            self.get_random_mapping()
        else:
            raise ValueError('Invalid use_mapping_type')
        assert len(self.accepting_states & self.rejecting_states) == 0
        self.transitions_to_circuit(qfa.transitions)

    def get_mapping(self):
        super().get_mapping()

        self.accepting_states = set(np.flatnonzero(self.qfa.accepting_states))
        self.rejecting_states = set(np.flatnonzero(self.qfa.rejecting_states))

    def get_random_mapping(self, seed: int = 42):
        np.random.seed(seed)
        state_list = list(range(2 ** self.size))
        state_order = list(range(2 ** self.size))
        np.random.shuffle(state_order)

        new_mapping = dict()
        for index, state in enumerate(state_list):
            new_mapping[state] = state_order[index]

#        accepting_states_num = len(np.flatnonzero(self.qfa.accepting_states))
#        rejecting_states_num = len(np.flatnonzero(self.qfa.rejecting_states))
#        for index, state in enumerate(np.flatnonzero(
#                self.qfa.accepting_states)):
#            new_mapping[state] = state_order[index]
#        for index, state in enumerate(np.flatnonzero(
#                self.qfa.rejecting_states)):
#            new_mapping[state] = state_order[
#                    index + accepting_states_num]
#        for index, state in enumerate(self.undefined_states):
#            new_mapping[state] = state_order[
#                    index + accepting_states_num + rejecting_states_num]

        print(new_mapping)

        # State status mapping
        self.mapping = new_mapping
        self.accepting_states = {
                self.mapping[state] for state in
                set(np.flatnonzero(
                    self.qfa.accepting_states))}
        self.rejecting_states = {
                self.mapping[state] for state in
                set(np.flatnonzero(
                    self.qfa.rejecting_states))}

    def get_entropy_mapping(self):
        state_order = [
                int(''.join(map(str, tp)), base=2)
                for tp in sorted(
                    it.product([0, 1], repeat=self.size),
                    key=sum)]
        new_mapping = dict()
        for index, state in enumerate(np.flatnonzero(
                self.qfa.accepting_states)):
            new_mapping[state] = state_order[index]
        for index, state in enumerate(np.flatnonzero(
                self.qfa.rejecting_states)):
            new_mapping[state] = state_order[-index-1]
        for index, state in enumerate(self.undefined_states):
            new_mapping[state] = state_order[
                    index
                    + len(np.flatnonzero(
                        self.qfa.accepting_states))]

        # State status mapping
        self.mapping = new_mapping
        self.accepting_states = {
                self.mapping[state] for state in
                set(np.flatnonzero(
                    self.qfa.accepting_states))}
        self.rejecting_states = {
                self.mapping[state] for state in
                set(np.flatnonzero(
                    self.qfa.rejecting_states))}

    def get_circuit_for_string(self, w: list[int]):
        qreg_states = QuantumRegister(self.size, 'q_state')
        creg_states = ClassicalRegister(self.size, 'c_state')

        circuit = QuantumCircuit(qreg_states, creg_states)

        for c in [self.qfa.start_of_string] + w + [self.qfa.end_of_string]:
            circuit.append(self.circuit[c], qreg_states)
        circuit.measure_all(add_bits=False)
        return circuit
