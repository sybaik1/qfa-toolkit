from typing import Optional
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
    def __init__(
        self,
        qfa: Moqfa,
        use_density_mapping: bool = True,
        *,
        use_entropy_mapping: Optional[bool] = None,
    ):
        self.qfa: Moqfa = qfa
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

    def get_mapping(self) -> None:
        super().get_mapping()

        self.accepting_states = {
            int(s) for s in np.flatnonzero(self.qfa.accepting_states)
        }
        self.rejecting_states = {
            int(s) for s in np.flatnonzero(self.qfa.rejecting_states)
        }

    def get_random_mapping(self, seed: int = 42) -> None:
        np.random.seed(seed)
        state_order = list(range(2 ** self.size))
        np.random.shuffle(state_order)
        accepting_indices = [
            int(s) for s in np.flatnonzero(self.qfa.accepting_states)
        ]
        rejecting_indices = [
            int(s) for s in np.flatnonzero(self.qfa.rejecting_states)
        ]
        accepting_states_num = len(accepting_indices)
        rejecting_states_num = len(rejecting_indices)

        new_mapping: dict[int, int] = dict()
        for index, state in enumerate(accepting_indices):
            new_mapping[state] = state_order[index]
        for index, state in enumerate(rejecting_indices):
            new_mapping[state] = state_order[
                    index + accepting_states_num]
        for index, state in enumerate(self.undefined_states):
            new_mapping[state] = state_order[
                    index + accepting_states_num + rejecting_states_num]

        # State status mapping
        self.mapping = new_mapping
        self.accepting_states = {
            self.mapping[state] for state in accepting_indices
        }
        self.rejecting_states = {
            self.mapping[state] for state in rejecting_indices
        }

    def get_density_mapping(self) -> None:
        state_order = [
                int(''.join(map(str, tp)), base=2)
                for tp in sorted(
                    it.product([0, 1], repeat=self.size),
                    key=sum)]
        accepting_indices = [
            int(s) for s in np.flatnonzero(self.qfa.accepting_states)
        ]
        rejecting_indices = [
            int(s) for s in np.flatnonzero(self.qfa.rejecting_states)
        ]
        new_mapping: dict[int, int] = dict()
        for index, state in enumerate(accepting_indices):
            new_mapping[state] = state_order[index]
        for index, state in enumerate(rejecting_indices):
            new_mapping[state] = state_order[-index-1]
        for index, state in enumerate(self.undefined_states):
            new_mapping[state] = state_order[
                    index + len(accepting_indices)]

        # State status mapping
        self.mapping = new_mapping
        self.accepting_states = {
            self.mapping[state] for state in accepting_indices
        }
        self.rejecting_states = {
            self.mapping[state] for state in rejecting_indices
        }

    def get_entropy_mapping(self) -> None:
        self.get_density_mapping()

    def get_circuit_for_string(self, w: list[int]) -> QuantumCircuit:
        qreg_states = QuantumRegister(self.size, 'q_state')
        creg_states = ClassicalRegister(self.size, 'c_state')

        circuit = QuantumCircuit(qreg_states, creg_states)

        for c in [self.qfa.start_of_string] + w + [self.qfa.end_of_string]:
            circuit.append(self.circuit[c], qreg_states)
        circuit.measure_all(add_bits=False)
        return circuit
