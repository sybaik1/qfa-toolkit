import abc
import math
import numpy as np

from .utils import unitary_matrix_to_circuit
from ..quantum_finite_automaton.quantum_finite_automaton_base import (
        QuantumFiniteAutomatonBase)


class QiskitQuantumFiniteAutomaton(abc.ABC):
    def __init__(self, qfa: QuantumFiniteAutomatonBase):
        """
        -- Properties --
        qfa: QuantumFiniteAutomaton
        size: int
        mapping: dict[int, int]
        defined_states: set[int]
        undefined_states: set[int]
        circuit: list[QuantumCircuit]
        """
        self.qfa = qfa
        self.get_size()
        self.get_mapping()
        self.transitions_to_circuit(qfa.transitions)

    @property
    def alphabet(self):
        return self.qfa.alphabet

    @property
    def states(self):
        return 2 ** self.size

    @property
    def defined_states(self) -> set[int]:
        return set(range(self.qfa.states))

    @property
    def undefined_states(self) -> set[int]:
        return set(range(self.size ** 2)) - self.defined_states

    # mapping from qiskit states to qfa states
    @property
    def reverse_mapping(self) -> dict[int, int]:
        return {v: k for k, v in self.mapping.items()}

    def get_size(self):
        self.size = math.ceil(math.log2(self.qfa.states))

    def get_mapping(self):
        self.mapping = dict()
        for state in self.defined_states:
            self.mapping[state] = state
        for state in self.undefined_states:
            self.mapping[state] = state

    def _transition_to_circuit(self, transition, character: str):
        # make unitary matrix to be a square matrix of size 2^n
        unitary_matrix = np.eye(self.states, dtype=complex)
        unitary_matrix[:unitary_matrix.shape[0],
                       :unitary_matrix.shape[1]] = unitary_matrix

        rev_index = [self.reverse_mapping[state]
                     for state in range(self.states)]

        # apply the mapping to the unitary matrix
        unitary_matrix[:] = unitary_matrix[rev_index][:, rev_index]

        return unitary_matrix_to_circuit(unitary_matrix, character)

    def transitions_to_circuit(self, transitions):
        self.circuit = dict()
        for character in range(self.qfa.alphabet + 2):
            self.circuit[character] = self._transition_to_circuit(
                transitions[character], str(character))

    @abc.abstractmethod
    def get_circuit_for_string(self, w: list[int]):
        raise NotImplementedError()

#   @abc.abstractmethod
#   def get_qfa_result(self):
#       raise NotImplementedError()

#   @abc.abstractmethod
#   def get_qfa_result_probability(self):
#       raise NotImplementedError()

#   @abc.abstractmethod
#   def get_qfa_result_state(self):
#       raise NotImplementedError()
