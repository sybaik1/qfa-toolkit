from typing import (TypeVar, Optional, )

import numpy as np

from .quantum_finite_automaton_base import NotClosedUnderOperationException
from .quantum_finite_automaton_base import Observable
from .quantum_finite_automaton_base import QuantumFiniteAutomatonBase
from .quantum_finite_automaton_base import States
from .quantum_finite_automaton_base import TotalState
from .quantum_finite_automaton_base import Transition
from .utils import direct_sum
from .utils import get_real_valued_transition

MmqfaT = TypeVar('MmqfaT', bound='MeasureManyQuantumFiniteAutomaton')


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

    def step(self, total_state: TotalState, c: int) -> TotalState:
        total_state = total_state.apply(self.transitions[c])
        total_state = total_state.measure_by(self.observable)
        return total_state

    def __call__(self, w):
        tape = self.string_to_tape(w)
        return self.process(tape).acceptance

    def concatenation(self, other: MmqfaT) -> MmqfaT:
        if self.alphabet != other.alphabet:
            raise ValueError("Alphabets must be the same")
        raise NotImplementedError()

    def union(self: MmqfaT, other: MmqfaT) -> MmqfaT:
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

    def intersection(self: MmqfaT, other: MmqfaT) -> MmqfaT:
        """Returns the intersection of two measure-many quantum finite
        automata.

        For a quantum finite automaton M and N, the intersection is defined as
        the quantum finite automaton M' such that M'(w) = M(w) * N(w) for all
        w.

        Generally, MMQFA is not closed under the intersection. However,
        end-decisive MMQFAs with pure states are closed under the intersection.
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

    def complement(self: MmqfaT) -> MmqfaT:
        """Returns the complement of the quantum finite automaton.

        For a quantum finite automaton M, the complement is defined as the
        quantum finite automaton M' such that M'(w) = 1 - M(w) for all w.

        Alberto Bertoni, Carlo Mereghetti, and Beatrice Palano. 2003. Quantum
        Computing: 1-Way Quantum Automata. In Proceedings of the 8th
        International Conference on Developments in Language Theory (DLT'04).
        """
        return self.__class__(
            self.transitions, self.rejecting_states, self.accepting_states)

    def linear_combination(self: MmqfaT, other: MmqfaT, c: float) -> MmqfaT:
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
        initial_transition[self.states][0] = f
        initial_transition[self.states][self.states] = -d

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

    def word_quotient(self: MmqfaT, w: list[int]) -> MmqfaT:
        raise NotImplementedError()

    def inverse_homomorphism(self: MmqfaT, phi: list[list[int]]) -> MmqfaT:
        """Returns the inverse homomorphism of the measure-many quantum finite
        automaton.

        For a quantum finite automaton M and a homomorphism phi, the inverse
        homomorphism M' of M with respect to phi is an MMQFA M' such that M'(w)
        = M(phi(w)).

        Alex Brodsky and Nicholas Pippenger 2002. Characterizations of 1-way
        Quantum Finite Automata. SIAM Journal on Computing.
        """
        if len(phi) == 0:
            raise ValueError("phi must be non-empty")
        if phi[self.start_of_string] != [self.start_of_string]:
            raise ValueError("phi[start_of_string] must be [start_of_string]")
        raise NotImplementedError()

    def difference(self, other: MmqfaT) -> MmqfaT:
        if self.alphabet != other.alphabet:
            raise ValueError("Alphabets must be the same")
        raise NotImplementedError()

    def equivalence(self, other: MmqfaT) -> bool:
        """Returns whether the measure-many quantum finite automaton is equal.

        For quantum finite automata M and M', the equivalence is defined as
        whether M(w) = M'(w) for all w.

        See also counter_example().

        """
        return self.counter_example(other) is None

    def counter_example(self, other: MmqfaT) -> Optional[list[int]]:
        """Returns a counter example of the equivalence of the measure-many
        quantum finite automaton.

        For quantum finite automata M and M', the counter example is defined as
        a word w such that M(w) != M'(w).
        """
        if self.alphabet != other.alphabet:
            raise ValueError("Alphabets must be the same")
        raise NotImplementedError()

    def minimize(self: MmqfaT) -> MmqfaT:
        raise NotImplementedError()

    def symmetric_difference(self, other: MmqfaT) -> MmqfaT:
        if self.alphabet != other.alphabet:
            raise ValueError("Alphabets must be the same")
        raise NotImplementedError()

    def kleene_star(self: MmqfaT) -> MmqfaT:
        raise NotImplementedError()

    def kleene_plus(self: MmqfaT) -> MmqfaT:
        raise NotImplementedError()

    def reverse(self: MmqfaT) -> MmqfaT:
        raise NotImplementedError()

    def is_empty(self) -> bool:
        raise NotImplementedError()

    def is_end_decisive(self) -> bool:
        """Returns whether the quantum finite automaton is end-decisive.

        A quantum finite automaton is end-decisive if it accepts only after
        read end-of-string.

        Alex Brodsky and Nicholas Pippenger. 2002. Characterizations of 1-way
        Quantum Finite Automata. SIAM Journal on Computing.
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
        Quantum Finite Automata. SIAM Journal on Computing.
        """
        adjacency = sum(
            abs(self.transitions[:-1]) + np.eye(self.states)).astype(bool)
        connected = np.linalg.matrix_power(adjacency, self.states)[0]
        return not any(self.rejecting_states & connected)

    def to_real_valued(self: MmqfaT) -> MmqfaT:
        transitions = np.stack([
            get_real_valued_transition(transition)
            for transition in self.transitions
        ]).astype(complex)

        stacked_accepting = np.stack([self.accepting_states] * 2)
        accepting_states = stacked_accepting.T.reshape(2 * self.states)
        stacked_rejecting = np.stack([self.rejecting_states] * 2)
        rejecting_states = stacked_rejecting.T.reshape(2 * self.states)
        return self.__class__(transitions, accepting_states, rejecting_states)
