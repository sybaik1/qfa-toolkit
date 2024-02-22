from functools import reduce
from itertools import product

import numpy as np
import numpy.typing as npt

from .quantum_finite_automaton_base import TotalState
from .quantum_finite_automaton_base import Observable
from .quantum_finite_automaton_base import QuantumFiniteAutomatonBase
from ..quantum_finite_automaton.measure_many_quantum_finite_automaton import (
    MeasureManyQuantumFiniteAutomaton as Mmqfa)

Moqfa = 'MeasureOnceQuantumFiniteAutomaton'


class MeasureOnceQuantumFiniteAutomaton(QuantumFiniteAutomatonBase):
    def __init__(
        self,
        transition: npt.NDArray[np.cdouble],
        accepting_states: set[int]
    ) -> None:
        super().__init__(transition)

        if not all(state < self.states for state in accepting_states):
            raise ValueError("Accepting states must be a subset of states")

        self.accepting_states = accepting_states

    @property
    def rejecting_states(self) -> set[int]:
        return set(range(self.states)) - self.accepting_states

    @property
    def observable(self) -> Observable:
        return (self.accepting_states, self.rejecting_states, set())

    def process(self, total_state: TotalState, w: list[int]) -> TotalState:
        total_state = reduce(self.step, w, total_state)
        return total_state.measure_by(self.observable)

    def step(self, total_state: TotalState, c: int) -> TotalState:
        return total_state.apply(self.transition[c])

    def concatenation(self, other: Moqfa) -> Moqfa:
        raise NotImplementedError()

    def union(self, other: Moqfa) -> Moqfa:
        """Returns the union of the quantum finite automaton.

        For a quantum finite automaton M and N, the union is defined as the
        quantum finite automaton M' such that 1 - M'(w) = (1 - M(w)) * (1 -
        N(w)) for all w.
        """
        return ~(~self & ~other)

    def intersection(self, other: Moqfa) -> Moqfa:
        """Returns the intersection of the quantum finite automaton.

        For a quantum finite automaton M and N, the intersection, also known as
        Hadamard product, is defined as the quantum finite automaton M' such
        that M'(w) = M(w) * N(w) for all w.

        Alberto Bertoni, Carlo Mereghetti, and Beatrice Palano. 2003. Quantum
        Computing: 1-Way Quantum Automata. In Proceedings of the 8th
        International Conference on Developments in Language Theory (DLT'04).
        """
        transition = np.stack([
            np.kron(u, v) for u, v
            in zip(self.transition, other.transition)
        ])
        accepting_states = set(
            i * other.states + j for i, j
            in product(self.accepting_states, other.accepting_states)
        )

        return self.__class__(transition, accepting_states)

    def complement(self: Moqfa) -> Moqfa:
        """Returns the complement of the quantum finite automaton.

        For a quantum finite automaton M, the complement is defined as the
        quantum finite automaton M' such that M'(w) = 1 - M(w) for all w.

        Alberto Bertoni, Carlo Mereghetti, and Beatrice Palano. 2003. Quantum
        Computing: 1-Way Quantum Automata. In Proceedings of the 8th
        International Conference on Developments in Language Theory (DLT'04).
        """
        return self.__class__(self.transition, self.rejecting_states)

    def linear_combination(self, other: Moqfa, c: float) -> Moqfa:
        """Returns the linear combination of two MMQFA.

        For a quantum finite automaton M, N and 0 <= c <= 1, the linear
        combination M' is an MMQFA such that M'(w) = c * M(w) + (1 - c) * N(w)
        for all w.

        Alberto Bertoni, Carlo Mereghetti, and Beatrice Palano. 2003. Quantum
        Computing: 1-Way Quantum Automata. In Proceedings of the 8th
        International Conference on Developments in Language Theory (DLT'04).
        """
        if not 0 <= c and c <= 1:
            raise ValueError("c must be in [0, 1]")

        # (U, V) -> [U 0; 0 V]
        def direct_sum(
            u: npt.NDArray[np.cdouble],
            v: npt.NDArray[np.cdouble]
        ) -> npt.NDArray[np.cdouble]:
            w1 = np.concatenate(
                (u, np.zeros((u.shape[0], v.shape[1]))), axis=1)
            w2 = np.concatenate(
                (np.zeros((v.shape[0], u.shape[1])), v), axis=1)
            w = np.concatenate((w1, w2), axis=0)
            return w

        f = np.sqrt(1 - c)
        d = np.sqrt(c)

        states = self.states + other.states
        initial_transition = np.eye(states)
        initial_transition[0][0] = d
        initial_transition[0][self.states] = f
        initial_transition[self.states][0] = -f
        initial_transition[self.states][self.states] = d
        transition = np.stack([
            direct_sum(u, v) for u, v in zip(self.transition, other.transition)
        ])
        transition[0] = transition[0] @ initial_transition
        accepting_states = (
            self.accepting_states
            | set(state + self.states for state in other.accepting_states)
        )
        return self.__class__(transition, accepting_states)

    def difference(self, other: Moqfa) -> Moqfa:
        raise NotImplementedError()

    def equivalence(self, other: Moqfa) -> bool:
        raise NotImplementedError()

    def minimize(self: Moqfa) -> Moqfa:
        raise NotImplementedError()

    def symmetric_difference(self, other: Moqfa) -> Moqfa:
        raise NotImplementedError()

    def kleene_star(self: Moqfa) -> Moqfa:
        raise NotImplementedError()

    def kleene_plus(self: Moqfa) -> Moqfa:
        raise NotImplementedError()

    def reverse(self: Moqfa) -> Moqfa:
        raise NotImplementedError()

    def is_empty(self) -> bool:
        raise NotImplementedError()

    def to_measure_many_quantum_finite_automaton(self) -> Mmqfa:
        raise NotImplementedError()
