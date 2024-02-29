from typing import (TypeVar, )
from functools import reduce

import numpy as np

from .quantum_finite_automaton_base import TotalState
from .quantum_finite_automaton_base import (States, Transition, Observable, )
from .quantum_finite_automaton_base import QuantumFiniteAutomatonBase
from .quantum_finite_automaton_base import NotClosedUnderOperationException

TMmqfa = TypeVar('TMmqfa', bound='MeasureManyQuantumFiniteAutomaton')


def direct_sum(u: Transition, v: Transition) -> Transition:
    """Returns the direct sum of two matrices.

    Direct sum of U, V: (U, V) |-> [U 0; 0 V]
    """
    w1 = np.concatenate(
        (u, np.zeros((u.shape[0], v.shape[1]))), axis=1)
    w2 = np.concatenate(
        (np.zeros((v.shape[0], u.shape[1])), v), axis=1)
    w = np.concatenate((w1, w2), axis=0)
    return w


class MeasureManyQuantumFiniteAutomaton(QuantumFiniteAutomatonBase):
    def __init__(
        self,
        transition: Transition,
        accepting_states: States,
        rejecting_states: States,
    ) -> None:
        super().__init__(transition)

        if not len(accepting_states) == self.states:
            raise ValueError(
                "accepting_states must be the same size as states")
        if not len(rejecting_states) == self.states:
            raise ValueError(
                "rejecting_states must be the same size as states")
        if np.any(accepting_states & rejecting_states):
            raise ValueError("Accepting and rejecting states must be disjoint")

        self.accepting_states = accepting_states
        self.rejecting_states = rejecting_states

    @property
    def halting_states(self) -> States:
        return self.accepting_states | self.rejecting_states

    @property
    def non_halting_states(self) -> States:
        return ~self.halting_states

    @property
    def observable(self) -> Observable:
        return np.stack([
            self.accepting_states,
            self.rejecting_states,
            self.non_halting_states
        ])

    def process(self, total_state: TotalState, w: list[int]) -> TotalState:
        return reduce(self.step, w, total_state)

    def step(self, total_state: TotalState, c: int) -> TotalState:
        total_state = total_state.apply(self.transitions[c])
        total_state = total_state.measure_by(self.observable)
        return total_state

    def concatenation(self, other: TMmqfa) -> TMmqfa:
        if self.alphabet != other.alphabet:
            raise ValueError("Alphabets must be the same")
        raise NotImplementedError()

    def union(self: TMmqfa, other: TMmqfa) -> TMmqfa:
        """Returns the union of two measure-many quantum finite automata.

        For a quantum finite automaton M and N, the union is defined as the
        quantum finite automaton M' such that 1 - M'(w) = (1 - M(w)) * (1 -
        N(w)) for all w.

        Generally, MMQFA is not closed under the union. See intersection() for
        details.

        Maria Paola Bianchi and Beatrice Palano. 2010. Behaviours of Unary
        Quantum Automata. Fundamenta Informaticae.

        Raises: NotClosedUnderOperationError
        """
        if self.alphabet != other.alphabet:
            raise ValueError("Alphabets must be the same")

        if not self.is_co_end_decisive() or not other.is_co_end_decisive():
            raise NotClosedUnderOperationException()

        return ~((~self) & (~other))

    def intersection(self: TMmqfa, other: TMmqfa) -> TMmqfa:
        """Returns the intersection of two measure-many quantum finite
        automata.

        For a quantum finite automaton M and N, the union is defined as the
        quantum finite automaton M' such that M'(w) = M(w) * N(w) for all w.

        Generally, MMQFA is not closed under the intersection. However,
        end-decisive MMQFA with pure states are closed under the intersection.
        Note that this is not a necessary condition.

        Maria Paola Bianchi and Beatrice Palano. 2010. Behaviours of Unary
        Quantum Automata. Fundamenta Informaticae.

        Raises: NotClosedUnderOperationError
        """
        if self.alphabet != other.alphabet:
            raise ValueError("Alphabets must be the same")

        if not self.is_end_decisive() or not other.is_end_decisive():
            raise NotClosedUnderOperationException()
        transition = np.stack([
            np.kron(u, v) for u, v
            in zip(self.transitions, other.transitions)
        ])
        non_halting_states = np.kron(
            self.non_halting_states, other.non_halting_states)
        halting_states = ~non_halting_states
        accepting_states = np.kron(
            self.accepting_states, other.accepting_states)
        rejecting_states = halting_states & ~accepting_states
        return self.__class__(transition, accepting_states, rejecting_states)

    def complement(self: TMmqfa) -> TMmqfa:
        """Returns the complement of the quantum finite automaton.

        For a quantum finite automaton M, the complement is defined as the
        quantum finite automaton M' such that M'(w) = 1 - M(w) for all w.

        Alberto Bertoni, Carlo Mereghetti, and Beatrice Palano. 2003. Quantum
        Computing: 1-Way Quantum Automata. In Proceedings of the 8th
        International Conference on Developments in Language Theory (DLT'04).
        """
        return self.__class__(
            self.transitions, self.rejecting_states, self.accepting_states)

    def linear_combination(self: TMmqfa, other: TMmqfa, c: float) -> TMmqfa:
        """Returns the linear combination of two MMQFA.

        For a quantum finite automaton M, N and 0 <= c <= 1, the linear
        combination M' is an MMQFA such that M'(w) = c * M(w) + (1 - c) * N(w)
        for all w.

        Alberto Bertoni, Carlo Mereghetti, and Beatrice Palano. 2003. Quantum
        Computing: 1-Way Quantum Automata. In Proceedings of the 8th
        International Conference on Developments in Language Theory (DLT'04).
        """
        if self.alphabet != other.alphabet:
            raise ValueError("Alphabets must be the same")
        if not 0 <= c and c <= 1:
            raise ValueError("c must be in [0, 1]")

        states = self.states + other.states
        f = np.sqrt(1 - c)
        d = np.sqrt(c)

        initial_transition = np.eye(states)
        initial_transition[0][0] = d
        initial_transition[0][self.states] = f
        initial_transition[self.states][0] = -f
        initial_transition[self.states][self.states] = d

        transition = np.stack([
            direct_sum(u, v)
            for u, v in zip(self.transitions, other.transitions)
        ])
        transition[self.start_of_string] = (
            transition[self.start_of_string] @ initial_transition)
        accepting_states = np.concatenate(
            (self.accepting_states, other.accepting_states))
        rejecting_states = np.concatenate(
            (self.rejecting_states, other.rejecting_states))
        return self.__class__(transition, accepting_states, rejecting_states)

    def word_quotient(self: TMmqfa, w: list[int]) -> TMmqfa:
        raise NotImplementedError()

    def difference(self, other: TMmqfa) -> TMmqfa:
        if self.alphabet != other.alphabet:
            raise ValueError("Alphabets must be the same")
        raise NotImplementedError()

    def equivalence(self, other: TMmqfa) -> bool:
        if self.alphabet != other.alphabet:
            raise ValueError("Alphabets must be the same")
        raise NotImplementedError()

    def minimize(self: TMmqfa) -> TMmqfa:
        raise NotImplementedError()

    def symmetric_difference(self, other: TMmqfa) -> TMmqfa:
        if self.alphabet != other.alphabet:
            raise ValueError("Alphabets must be the same")
        raise NotImplementedError()

    def kleene_star(self: TMmqfa) -> TMmqfa:
        raise NotImplementedError()

    def kleene_plus(self: TMmqfa) -> TMmqfa:
        raise NotImplementedError()

    def reverse(self: TMmqfa) -> TMmqfa:
        raise NotImplementedError()

    def is_empty(self) -> bool:
        raise NotImplementedError()

    def is_end_decisive(self) -> bool:
        """Returns whether the quantum finite automaton is end-decisive.

        A quantum finite automaton is end-decisive if it accepts only after
        read end-of-string.

        Alex Brodsky and Nicholas Pippenger. 2002. Characterizations of 1-way
        Quantum Finite Automata. SIAM Jornal on Computing.
        """
        adjacency = sum(
            abs(self.transitions[:-1]) + np.eye(self.states)).astype(bool)
        connected = np.linalg.matrix_power(adjacency, self.states)[0]
        return not any(self.accepting_states & connected)

    def is_co_end_decisive(self) -> bool:
        """Returns whether the quantum finite automaton is co-end-decisive.

        A quantum finite automaton is end-decisive if it rejects only after
        read end-of-string.

        Alex Brodsky and Nicholas Pippenger. 2002. Characterizations of 1-way
        Quantum Finite Automata. SIAM Jornal on Computing.
        """
        adjacency = sum(
            abs(self.transitions[:-1]) + np.eye(self.states)).astype(bool)
        connected = np.linalg.matrix_power(adjacency, self.states)[0]
        return not any(self.rejecting_states & connected)
