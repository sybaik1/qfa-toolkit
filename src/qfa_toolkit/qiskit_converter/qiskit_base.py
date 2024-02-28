import abc
import math

from .utils import unitary_matrix_to_circuit
from ..quantum_finite_automaton.quantum_finite_automaton_base import (
        QuantumFiniteAutomatonBase)


class QiskitQuantumFiniteAutomaton(abc.ABC):
    def __init__(self, qfa: QuantumFiniteAutomatonBase):
        self.qfa = qfa
        self.size = math.ceil(math.log2(self.qfa.states))
        self.circuit = self.transitions_to_circuit(qfa.transition)

    @property
    def quantum_finite_automaton(self):
        return self.qfa

    @property
    def states(self):
        return 2 ** self.size

    def transitions_to_circuit(self, transition):
        return [unitary_matrix_to_circuit(transition, str(index))
                for index, transition in enumerate(transition)]

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
