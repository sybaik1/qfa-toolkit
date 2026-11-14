import math
from typing import Optional

import numpy as np
import itertools as it
from qiskit.circuit import QuantumCircuit  # type: ignore
from qiskit import QuantumRegister, ClassicalRegister  # type: ignore

from .qiskit_base import QiskitQuantumFiniteStateAutomaton
from ..quantum_finite_state_automaton import (
    MeasureManyQuantumFiniteStateAutomaton as Mmqfa)


class QiskitMeasureManyQuantumFiniteStateAutomaton(
    QiskitQuantumFiniteStateAutomaton
):
    def __init__(
        self,
        qfa: Mmqfa,
        use_density_mapping: bool = True,
        *,
        use_entropy_mapping: Optional[bool] = None,
    ):
        """
        -- Properties --
        qfa: Mmqfa
        accepting_states: set[int]
        rejecting_states: set[int]
        circuit: list[QuantumCircuit] # indexed by the alphabet
        size: int                     # size of the qubit register for states
        mapping: dict[int, int]       # mapping from QFA state index
                                        to qubit basis index
        """
        self.qfa: Mmqfa = qfa
        self.get_size()
        self.mapping = {k: k for k in range(2 ** self.size)}
        if use_entropy_mapping is not None:
            use_density_mapping = use_entropy_mapping
        if use_density_mapping:
            self.get_density_mapping()
        else:
            self.get_mapping()
        assert len(self.accepting_states & self.rejecting_states) == 0
        self.transitions_to_circuit(qfa.transitions)

    @property
    def halting_states(self):
        return self.accepting_states | self.rejecting_states

    def get_size(self) -> None:
        self.size = 1 + math.ceil(
            math.log2(
                max(len(np.flatnonzero(self.qfa.halting_states)),
                    len(np.flatnonzero(self.qfa.non_halting_states)))))

    def get_density_mapping(self) -> None:
        state_order = [
                int('1'+''.join(map(str, tp)), base=2)
                for tp in sorted(
                    it.product([0, 1], repeat=self.size-1),
                    key=sum)]
        non_halting_indices = [
            int(s) for s in np.flatnonzero(self.qfa.non_halting_states)
        ]
        accepting_indices = [
            int(s) for s in np.flatnonzero(self.qfa.accepting_states)
        ]
        rejecting_indices = [
            int(s) for s in np.flatnonzero(self.qfa.rejecting_states)
        ]
        new_mapping: dict[int, int] = dict()
        # non-halting states mapped with the first qubit as 0
        for index, state in enumerate(non_halting_indices):
            new_mapping[state] = index

        # halting states
        # bits with less 1s
        for index, state in enumerate(accepting_indices):
            new_mapping[state] = state_order[index]
        # bits with more 1s
        for index, state in enumerate(rejecting_indices):
            new_mapping[state] = state_order[-index-1]

        half_size = 2 ** (self.size - 1)
        for index, state in enumerate(range(self.qfa.states, 2 ** self.size)):
            if index + len(non_halting_indices) < half_size:
                new_mapping[state] = index + len(non_halting_indices)
            else:
                new_mapping[state] = state_order[
                    index
                    + len(non_halting_indices)
                    + len(accepting_indices)
                    - half_size]

        # State status mapping
        self.mapping = new_mapping
        self.accepting_states = {
            self.mapping[state] for state in accepting_indices
        }
        self.rejecting_states = {
            self.mapping[state] for state in rejecting_indices
        }
        self.non_halting_states = {
            self.mapping[state] for state in non_halting_indices
        }

    def get_entropy_mapping(self) -> None:
        self.get_density_mapping()

    def get_mapping(self) -> None:
        # Create a mapping from the qubit register index to the qfa state
        half_size = 2 ** (self.size - 1)
        non_halting_indices = [
            int(s) for s in np.flatnonzero(self.qfa.non_halting_states)
        ]
        halting_indices = [
            int(s) for s in np.flatnonzero(self.qfa.halting_states)
        ]
        accepting_indices = [
            int(s) for s in np.flatnonzero(self.qfa.accepting_states)
        ]
        rejecting_indices = [
            int(s) for s in np.flatnonzero(self.qfa.rejecting_states)
        ]
        new_mapping: dict[int, int] = dict()
        for index, state in enumerate(non_halting_indices):
            new_mapping[state] = index
        for index, state in enumerate(halting_indices):
            new_mapping[state] = index + half_size
        for index, state in enumerate(range(self.qfa.states, 2 ** self.size)):
            if index + len(non_halting_indices) < half_size:
                new_mapping[state] = index + len(non_halting_indices)
            else:
                new_mapping[state] = index + self.qfa.states

        # State status mapping
        self.mapping = new_mapping
        self.accepting_states = {
            self.mapping[state] for state in accepting_indices
        }
        self.rejecting_states = {
            self.mapping[state] for state in rejecting_indices
        }
        self.non_halting_states = {
            self.mapping[state] for state in non_halting_indices
        }

    def get_circuit_for_string(self, w: list[int]) -> QuantumCircuit:
        qreg_states = QuantumRegister(self.size, 'q_state')
        creg_states = ClassicalRegister(self.size, 'c_state')

        circuit = QuantumCircuit(qreg_states, creg_states)
        self._prepare_initial_state(circuit, qreg_states)

        circuit.append(self.circuit[self.qfa.start_of_string], qreg_states)
        for c in w + [self.qfa.end_of_string]:
            circuit.measure(qreg_states[-1], creg_states[-1])
            with circuit.if_test((creg_states[-1], 0)):
                circuit.append(self.circuit[c], qreg_states)
        circuit.measure_all(add_bits=False)
        return circuit
